mod data;
mod guard;
mod parse;
mod score;
mod sim;
mod stats;
mod text;

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use data::{Incident, Manual, RawAnswer, Res};
use guard::Route;
use parse::{Answer, Severity};
use score::{Evidence, Fixed, Mark, Verdict, SOURCES};

const RESAMPLES: usize = 20_000;
const SHIFTS: usize = 100_000;
const SEED: u64 = 2026;
const PAIRS: [(&str, &str); 3] = [
    ("A_короткая", "B_структура"),
    ("A_короткая", "C_управляемый_AI"),
    ("B_структура", "C_управляемый_AI"),
];

struct Cli {
    data: PathBuf,
    model: String,
}

fn cli() -> Res<Cli> {
    let mut cli = Cli {
        data: PathBuf::from("data"),
        model: "qwen2.5:7b".into(),
    };
    let args: Vec<String> = env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| format!("после {flag} нужно значение"))?;
        match flag.as_str() {
            "--data" => cli.data = PathBuf::from(value),
            "--model" => cli.model.clone_from(value),
            _ => return Err(format!("непонятный флаг {flag}\nзапуск: kt2-incident-guard [--data data] [--model qwen2.5:7b]").into()),
        }
    }
    Ok(cli)
}

struct Row {
    inc: Incident,
    raw: RawAnswer,
    answer: Answer,
    verdict: Verdict,
    route: Route,
    why: Vec<&'static str>,
}

fn path_str(dir: &Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

fn fix_answer(a: &mut Answer, m: &Manual) -> Res<()> {
    let v = m.value.trim().to_string();
    match m.field.as_str() {
        "model_severity" => {
            a.severity = Severity::from_value(&v);
            a.severity_raw = v;
        }
        "model_owner" => a.owner = v,
        "model_first_action" => a.first_action = v,
        "evidence_quote" => a.evidence_quote = v,
        "unknowns" => a.unknowns = v,
        "model_human_review" => a.human_review = parse::yes_no(&v),
        "owner_ok" | "first_action_ok" | "evidence_ok" | "error_source" | "dangerous_error"
        | "comment" => {}
        other => return Err(format!("ручная проверка: непонятное поле {other}").into()),
    }
    Ok(())
}

fn fixed_marks(fixes: &[&Manual]) -> Res<Fixed> {
    let mut f = Fixed::default();
    for m in fixes {
        let bad = || {
            format!(
                "ручная проверка {} {}: {}={}",
                m.incident_id, m.prompt_version, m.field, m.value
            )
        };
        match m.field.as_str() {
            "owner_ok" => f.owner = Some(Mark::from_label(&m.value).ok_or_else(bad)?),
            "first_action_ok" => f.action = Some(Mark::from_label(&m.value).ok_or_else(bad)?),
            "evidence_ok" => f.evidence = Some(Evidence::from_label(&m.value).ok_or_else(bad)?),
            _ => {}
        }
    }
    Ok(f)
}

fn fix_verdict(v: &mut Verdict, fixes: &[&Manual]) -> Res<()> {
    for m in fixes {
        let val = m.value.trim();
        match m.field.as_str() {
            "error_source" => {
                v.sources = val
                    .split(';')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                if let Some(bad) = v.sources.iter().find(|s| !SOURCES.contains(&s.as_str())) {
                    return Err(format!(
                        "ручная проверка {}: неизвестный источник риска {bad}",
                        m.incident_id
                    )
                    .into());
                }
            }
            "dangerous_error" if val == "нет" => v.sources.clear(),
            "comment" => v.notes = vec![val.to_string()],
            _ => {}
        }
        if !m.reason.trim().is_empty() {
            v.notes
                .push(format!("ручная проверка: {}", m.reason.trim()));
        }
    }
    Ok(())
}

fn build_rows(dir: &Path) -> Res<Vec<Row>> {
    let incidents: HashMap<String, Incident> =
        data::incidents(&path_str(dir, "КТ2_отчёты_об_инцидентах.csv"))?
            .into_iter()
            .map(|i| (i.incident_id.clone(), i))
            .collect();
    let rubric = data::rubric(&path_str(dir, "рубрика.csv"))?;
    let manual = data::manual(&path_str(dir, "ручная_проверка.csv"))?;
    let mut rows = Vec::new();
    for raw in data::answers(&path_str(dir, "llm_raw.jsonl"))? {
        let Some(inc) = incidents.get(&raw.incident_id) else {
            eprintln!("пропускаю {}: такого случая нет в csv", raw.incident_id);
            continue;
        };
        let fixes: Vec<&Manual> = manual
            .iter()
            .filter(|m| m.incident_id == raw.incident_id && m.prompt_version == raw.prompt_version)
            .filter(|m| m.model.is_empty() || m.model == raw.model)
            .collect();
        let mut answer = parse::parse(&raw.prompt_version, &raw.response, &inc.report_text);
        for m in &fixes {
            fix_answer(&mut answer, m)?;
        }
        let mut verdict = score::score(inc, &answer, &rubric, fixed_marks(&fixes)?)?;
        fix_verdict(&mut verdict, &fixes)?;
        let (route, why) = guard::route(
            &inc.report_text,
            &answer,
            Evidence::of(&answer.evidence_quote, &inc.report_text),
        );
        rows.push(Row {
            inc: inc.clone(),
            raw,
            answer,
            verdict,
            route,
            why,
        });
    }
    rows.sort_by(|a, b| {
        (&a.raw.model, &a.raw.incident_id, &a.raw.prompt_version).cmp(&(
            &b.raw.model,
            &b.raw.incident_id,
            &b.raw.prompt_version,
        ))
    });
    Ok(rows)
}

fn yn(b: bool) -> &'static str {
    if b {
        "да"
    } else {
        "нет"
    }
}

fn write_journal(path: &str, rows: &[Row]) -> Res<()> {
    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "incident_id",
        "prompt_version",
        "model_severity",
        "model_owner",
        "model_first_action",
        "evidence_quote",
        "unknowns",
        "dangerous_error",
        "comment",
        "model",
        "gold_severity",
        "severity_ok",
        "owner_ok",
        "first_action_ok",
        "evidence_ok",
        "unknowns_ok",
        "model_human_review",
        "gold_human_review",
        "human_review_ok",
        "error_source",
        "guard_route",
        "guard_reason",
        "json_ok",
        "seconds",
    ])?;
    for r in rows {
        let (a, v) = (&r.answer, &r.verdict);
        let severity = a
            .severity
            .map(Severity::label)
            .map_or_else(|| a.severity_raw.clone(), String::from);
        w.write_record([
            r.raw.incident_id.clone(),
            r.raw.prompt_version.clone(),
            severity,
            a.owner.clone(),
            a.first_action.clone(),
            a.evidence_quote.clone(),
            a.unknowns.clone(),
            yn(v.dangerous()).into(),
            v.comment(),
            r.raw.model.clone(),
            r.inc.gold_severity.clone(),
            yn(v.severity_ok).into(),
            v.owner.label().into(),
            v.action.label().into(),
            v.evidence.label().into(),
            yn(v.unknowns_ok).into(),
            a.human_review.clone(),
            r.inc.gold_human_review.clone(),
            yn(v.human_review_ok).into(),
            v.sources.join("; "),
            r.route.label().into(),
            r.why.join("; "),
            yn(a.structured).into(),
            r.raw.seconds.map(|s| format!("{s:.1}")).unwrap_or_default(),
        ])?;
    }
    w.flush()?;
    Ok(())
}

type Metric = (&'static str, fn(&Row) -> Option<f64>);

fn f(b: bool) -> f64 {
    f64::from(u8::from(b))
}

fn all_ok(r: &Row) -> bool {
    let v = &r.verdict;
    v.severity_ok
        && v.owner == Mark::Yes
        && v.action == Mark::Yes
        && v.evidence == Evidence::Verbatim
        && v.human_review_ok
}

fn metrics() -> Vec<Metric> {
    vec![
        ("json_ok", |r| Some(f(r.answer.structured))),
        ("severity_ok", |r| Some(f(r.verdict.severity_ok))),
        ("owner_ok", |r| Some(f(r.verdict.owner == Mark::Yes))),
        ("first_action_ok", |r| {
            Some(f(r.verdict.action == Mark::Yes))
        }),
        ("first_action_partly", |r| {
            Some(f(r.verdict.action != Mark::No))
        }),
        ("evidence_ok", |r| {
            Some(f(r.verdict.evidence == Evidence::Verbatim))
        }),
        ("unknowns_ok", |r| Some(f(r.verdict.unknowns_ok))),
        ("human_review_ok", |r| Some(f(r.verdict.human_review_ok))),
        ("escalation_recall", |r| {
            (r.inc.gold_human_review == "да").then(|| f(r.answer.human_review == "да"))
        }),
        ("all_ok", |r| Some(f(all_ok(r)))),
        ("dangerous", |r| Some(f(r.verdict.dangerous()))),
        ("route_auto", |r| Some(f(r.route == Route::Auto))),
        ("route_suggest", |r| Some(f(r.route == Route::Suggest))),
        ("route_human", |r| Some(f(r.route == Route::Human))),
        ("dangerous_auto", |r| {
            Some(f(r.verdict.dangerous() && r.route == Route::Auto))
        }),
    ]
}

fn clusters<'a>(rows: impl Iterator<Item = (&'a str, Option<f64>)>) -> Vec<Vec<f64>> {
    let mut by_text: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (text, value) in rows {
        if let Some(v) = value {
            by_text.entry(text).or_default().push(v);
        }
    }
    by_text.into_values().collect()
}

fn stability(rows: &[&Row]) -> f64 {
    let mut by_text: BTreeMap<&str, Vec<Option<Severity>>> = BTreeMap::new();
    for r in rows {
        by_text
            .entry(&r.inc.report_text)
            .or_default()
            .push(r.answer.severity);
    }
    let multi: Vec<_> = by_text.values().filter(|v| v.len() > 1).collect();
    stats::mean(multi.iter().map(|v| f(v.iter().all(|s| *s == v[0]))))
}

fn groups(rows: &[Row]) -> BTreeMap<(&str, &str), Vec<&Row>> {
    let mut g: BTreeMap<(&str, &str), Vec<&Row>> = BTreeMap::new();
    for r in rows {
        g.entry((r.raw.model.as_str(), r.raw.prompt_version.as_str()))
            .or_default()
            .push(r);
    }
    g
}

fn write_summary(path: &str, rows: &[Row]) -> Res<()> {
    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "model",
        "prompt_version",
        "metric",
        "value",
        "ci_lo",
        "ci_hi",
    ])?;
    for (gi, ((model, version), g)) in groups(rows).into_iter().enumerate() {
        let mut out = |metric: &str, value: f64, ci: Option<(f64, f64)>| {
            let (lo, hi) = ci.map_or((String::new(), String::new()), |(l, h)| {
                (format!("{l:.4}"), format!("{h:.4}"))
            });
            w.write_record([model, version, metric, &format!("{value:.4}"), &lo, &hi])
        };
        out("n", g.len() as f64, None)?;
        for (mi, (name, fun)) in metrics().into_iter().enumerate() {
            let values: Vec<Option<f64>> = g.iter().map(|r| fun(r)).collect();
            let c = clusters(
                g.iter()
                    .map(|r| r.inc.report_text.as_str())
                    .zip(values.iter().copied()),
            );
            let value = stats::mean(values.iter().flatten().copied());
            let ci = stats::cluster_ci(&c, RESAMPLES, SEED + (gi * 100 + mi) as u64);
            out(name, value, Some(ci))?;
        }
        out(
            "dangerous_n",
            g.iter().filter(|r| r.verdict.dangerous()).count() as f64,
            None,
        )?;
        out("stable_severity", stability(&g), None)?;
        out(
            "mean_seconds",
            stats::mean(g.iter().filter_map(|r| r.raw.seconds)),
            None,
        )?;
        for src in SOURCES {
            let n = g
                .iter()
                .filter(|r| r.verdict.sources.iter().any(|s| s == src))
                .count();
            out(&format!("src:{src}"), n as f64, None)?;
        }
    }
    w.flush()?;
    Ok(())
}

fn write_compare(path: &str, rows: &[Row]) -> Res<()> {
    let mut w = csv::Writer::from_path(path)?;
    w.write_record(["model", "pair", "metric", "diff", "ci_lo", "ci_hi"])?;
    let g = groups(rows);
    let models: Vec<&str> = g
        .keys()
        .map(|(m, _)| *m)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let chosen = [
        "severity_ok",
        "owner_ok",
        "first_action_ok",
        "evidence_ok",
        "all_ok",
        "dangerous",
    ];
    for model in models {
        for (pi, (from, to)) in PAIRS.iter().enumerate() {
            let (Some(a), Some(b)) = (g.get(&(model, *from)), g.get(&(model, *to))) else {
                continue;
            };
            let by_id: HashMap<&str, &&Row> =
                a.iter().map(|r| (r.raw.incident_id.as_str(), r)).collect();
            for (mi, (name, fun)) in metrics()
                .into_iter()
                .filter(|(n, _)| chosen.contains(n))
                .enumerate()
            {
                let diffs: Vec<(&str, Option<f64>)> = b
                    .iter()
                    .filter_map(|rb| by_id.get(rb.raw.incident_id.as_str()).map(|ra| (rb, ra)))
                    .map(|(rb, ra)| {
                        (
                            rb.inc.report_text.as_str(),
                            fun(rb).zip(fun(ra)).map(|(x, y)| x - y),
                        )
                    })
                    .collect();
                let value = stats::mean(diffs.iter().filter_map(|(_, d)| *d));
                let (lo, hi) = stats::cluster_ci(
                    &clusters(diffs.into_iter()),
                    RESAMPLES,
                    SEED + 7 + (pi * 10 + mi) as u64,
                );
                w.write_record([
                    model,
                    &format!("{to} − {from}"),
                    name,
                    &format!("{value:.4}"),
                    &format!("{lo:.4}"),
                    &format!("{hi:.4}"),
                ])?;
            }
        }
    }
    w.flush()?;
    Ok(())
}

fn pct(x: f64) -> String {
    format!("{:.0}%", 100.0 * x)
}

fn print_summary(rows: &[Row]) {
    println!(
        "{:<14} {:<18} {:>5} {:>9} {:>8} {:>8} {:>7} {:>7} {:>7} {:>8}   авто / предл. / человек",
        "модель",
        "версия",
        "JSON",
        "приоритет",
        "владелец",
        "1-й шаг",
        "цитата",
        "неизв.",
        "эскал.",
        "опасных"
    );
    println!("{}", "-".repeat(130));
    for ((model, version), g) in groups(rows) {
        let share = |fun: fn(&Row) -> bool| pct(stats::mean(g.iter().map(|r| f(fun(r)))));
        let need: Vec<&&Row> = g
            .iter()
            .filter(|r| r.inc.gold_human_review == "да")
            .collect();
        let esc = stats::mean(need.iter().map(|r| f(r.answer.human_review == "да")));
        let routes: Vec<String> = Route::ALL
            .iter()
            .map(|rt| g.iter().filter(|r| r.route == *rt).count().to_string())
            .collect();
        println!(
            "{:<14} {:<18} {:>5} {:>9} {:>8} {:>8} {:>7} {:>7} {:>7} {:>8}   {}",
            model.chars().take(14).collect::<String>(),
            version,
            share(|r| r.answer.structured),
            share(|r| r.verdict.severity_ok),
            share(|r| r.verdict.owner == Mark::Yes),
            share(|r| r.verdict.action == Mark::Yes),
            share(|r| r.verdict.evidence == Evidence::Verbatim),
            share(|r| r.verdict.unknowns_ok),
            pct(esc),
            format!(
                "{}/{}",
                g.iter().filter(|r| r.verdict.dangerous()).count(),
                g.len()
            ),
            routes.join(" / ")
        );
    }
}

fn simulate(dir: &Path, rows: &[Row], model: &str) -> Res<()> {
    let c: Vec<&Row> = rows
        .iter()
        .filter(|r| r.raw.model == model && r.raw.prompt_version.starts_with('C'))
        .collect();
    if c.is_empty() {
        println!("\nнет ответов версии C у {model}, симуляцию пропускаю");
        return Ok(());
    }
    let outcomes: Vec<sim::Outcome> = c
        .iter()
        .map(|r| sim::Outcome {
            gold: r.verdict.gold.rank() as u8,
            model: r.answer.severity.map_or(3, |s| s.rank() as u8),
            route: r.route,
            error: r.verdict.error(),
            dangerous: r.verdict.dangerous(),
        })
        .collect();
    let latency: Vec<f64> = c
        .iter()
        .filter_map(|r| r.raw.seconds)
        .map(|s| s / 60.0)
        .collect();
    let latency = if latency.is_empty() {
        vec![0.5]
    } else {
        latency
    };
    let params = sim::Params::new(&data::assumptions(&path_str(dir, "допущения_времени.csv"))?)?;
    let t0 = Instant::now();
    let res = sim::run(&params, &outcomes, &latency, SHIFTS, SEED);
    let secs = t0.elapsed().as_secs_f64();

    println!(
        "\nСимуляция смены диспетчера ({model}, версия C + guard): {} смен на сценарий, {:.1} с",
        SHIFTS, secs
    );
    println!(
        "{:>9} {:<11} {:>10} {:>9} {:>10} {:>11} {:>11} {:>10} {:>12}",
        "сообщ/ч",
        "режим",
        "занят, мин",
        "загрузка",
        "критич P50",
        "критич P90",
        "позже 15м",
        "сэконом.",
        "часов/мес"
    );
    for r in &res {
        println!(
            "{:>9} {:<11} {:>10.0} {:>9} {:>9.1}м {:>10.1}м {:>11} {:>9} {:>12}",
            r.lambda_per_hour,
            r.mode,
            r.busy_min,
            pct(r.utilization),
            r.crit_first_action_p50,
            r.crit_first_action_p90,
            pct(r.crit_late_share),
            r.saved_min_p50
                .map_or(String::new(), |s| format!("{s:.0} мин")),
            r.saved_hours_month_p50
                .map_or(String::new(), |h| format!("{h:.0}"))
        );
    }
    let mut w = csv::Writer::from_path(path_str(dir, "rust_sim.csv"))?;
    for r in &res {
        w.serialize(r)?;
    }
    w.flush()?;
    Ok(())
}

fn main() -> Res<()> {
    let cli = cli()?;
    let dir = cli.data.as_path();
    let rows = build_rows(dir)?;
    if rows.is_empty() {
        return Err("в llm_raw.jsonl нет ответов".into());
    }
    println!("ответов разобрано: {}\n", rows.len());
    print_summary(&rows);

    write_journal(&path_str(dir, "rust_journal.csv"), &rows)?;
    let t0 = Instant::now();
    write_summary(&path_str(dir, "rust_summary.csv"), &rows)?;
    write_compare(&path_str(dir, "rust_compare.csv"), &rows)?;
    println!(
        "\nбутстрэп по текстам ({RESAMPLES} повторов на метрику): {:.1} с",
        t0.elapsed().as_secs_f64()
    );
    simulate(dir, &rows, &cli.model)?;
    println!(
        "\nзаписано: rust_journal.csv, rust_summary.csv, rust_compare.csv, rust_sim.csv в {}",
        dir.display()
    );
    Ok(())
}
