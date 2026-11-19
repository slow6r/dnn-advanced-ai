use crate::data::{Incident, Rubric};
use crate::parse::{Answer, Severity};
use crate::text::{has_any, hits, invented_numbers, norm, overlap, quote_found};

pub const SOURCES: [&str; 5] = [
    "неоднозначность",
    "потеря факта",
    "выдумка",
    "неверная срочность",
    "чрезмерная автоматизация",
];

const SCALE_WORDS: &[&str] = &[
    "масштаб",
    "сколько",
    "числ",
    "количеств",
    "затронут",
    "объем",
    "охват",
    "пострадав",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    Yes,
    Partly,
    No,
}

impl Mark {
    pub fn label(self) -> &'static str {
        match self {
            Mark::Yes => "да",
            Mark::Partly => "частично",
            Mark::No => "нет",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s.trim() {
            "да" => Some(Mark::Yes),
            "частично" => Some(Mark::Partly),
            "нет" => Some(Mark::No),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    Verbatim,
    Paraphrase,
    Invented,
    Missing,
}

impl Evidence {
    pub fn label(self) -> &'static str {
        match self {
            Evidence::Verbatim => "да",
            Evidence::Paraphrase => "пересказ",
            Evidence::Invented => "выдумка",
            Evidence::Missing => "нет",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s.trim() {
            "да" => Some(Evidence::Verbatim),
            "пересказ" => Some(Evidence::Paraphrase),
            "выдумка" => Some(Evidence::Invented),
            "нет" => Some(Evidence::Missing),
            _ => None,
        }
    }

    pub fn of(quote: &str, source: &str) -> Self {
        if quote.trim().chars().count() < 3 {
            Evidence::Missing
        } else if quote_found(quote, source) {
            Evidence::Verbatim
        } else if overlap(quote, source) >= 0.6 {
            Evidence::Paraphrase
        } else {
            Evidence::Invented
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Fixed {
    pub owner: Option<Mark>,
    pub action: Option<Mark>,
    pub evidence: Option<Evidence>,
}

#[derive(Debug, Clone)]
pub struct Verdict {
    pub gold: Severity,
    pub severity_ok: bool,
    pub owner: Mark,
    pub action: Mark,
    pub evidence: Evidence,
    pub unknowns_ok: bool,
    pub human_review_ok: bool,
    pub sources: Vec<String>,
    pub notes: Vec<String>,
}

impl Verdict {
    pub fn dangerous(&self) -> bool {
        !self.sources.is_empty()
    }

    pub fn error(&self) -> bool {
        !self.severity_ok || self.owner == Mark::No || self.action == Mark::No || self.dangerous()
    }

    fn flag(&mut self, source: &str, note: String) {
        if !self.sources.iter().any(|s| s == source) {
            self.sources.push(source.to_string());
        }
        self.notes.push(note);
    }

    pub fn comment(&self) -> String {
        if self.notes.is_empty() {
            "ок".into()
        } else {
            self.notes.join("; ")
        }
    }
}

fn owner_mark(gold: &str, owner: &str, rubric: &Rubric) -> Mark {
    if owner.trim().is_empty() {
        return Mark::No;
    }
    match rubric.owners.get(gold) {
        Some(rule) if hits(&rule.yes, owner) => Mark::Yes,
        Some(rule) if hits(&rule.partial, owner) => Mark::Partly,
        _ => Mark::No,
    }
}

fn action_mark(gold: &str, action: &str, rubric: &Rubric) -> Mark {
    let groups = match rubric.actions.get(gold) {
        Some(g) if !g.is_empty() && !action.trim().is_empty() => g,
        _ => return Mark::No,
    };
    let got = groups.iter().filter(|g| hits(g, action)).count();
    match got {
        0 => Mark::No,
        n if n == groups.len() => Mark::Yes,
        _ => Mark::Partly,
    }
}

pub fn score(inc: &Incident, a: &Answer, rubric: &Rubric, fixed: Fixed) -> Result<Verdict, String> {
    let gold = Severity::from_value(&inc.gold_severity).ok_or_else(|| {
        format!(
            "{}: непонятный эталон {}",
            inc.incident_id, inc.gold_severity
        )
    })?;
    let severity_diff = a.severity.map(|s| s.rank() - gold.rank());
    let owner = fixed
        .owner
        .unwrap_or_else(|| owner_mark(&inc.gold_owner, &a.owner, rubric));
    let action = fixed
        .action
        .unwrap_or_else(|| action_mark(&inc.gold_first_action, &a.first_action, rubric));
    let evidence = fixed
        .evidence
        .unwrap_or_else(|| Evidence::of(&a.evidence_quote, &inc.report_text));
    let needs_human = inc.gold_human_review == "да";
    let unknowns = norm(&a.unknowns);
    let unknowns_ok = !unknowns.is_empty() && (!needs_human || has_any(&unknowns, SCALE_WORDS));

    let mut v = Verdict {
        gold,
        severity_ok: a.severity == Some(gold),
        owner,
        action,
        evidence,
        unknowns_ok,
        human_review_ok: a.human_review == inc.gold_human_review,
        sources: Vec::new(),
        notes: Vec::new(),
    };

    let said = a.severity.map(Severity::label).unwrap_or("не указан");
    match severity_diff {
        None if gold >= Severity::High => v.flag(
            "потеря факта",
            format!("критичность не определена, а по эталону {}", gold.label()),
        ),
        Some(d) if d < 0 && gold >= Severity::High => v.flag(
            "потеря факта",
            format!("занижен приоритет: {said} вместо {}", gold.label()),
        ),
        Some(d) if d >= 2 => v.flag(
            "неверная срочность",
            format!("завышен приоритет: {said} вместо {}", gold.label()),
        ),
        Some(d) if d != 0 => v
            .notes
            .push(format!("приоритет {said} вместо {}", gold.label())),
        None => v.notes.push("критичность не определена".into()),
        _ => {}
    }

    match owner {
        Mark::No if a.owner.trim().is_empty() => v.notes.push("владелец не назван".into()),
        Mark::No if gold >= Severity::High => {
            v.flag("неоднозначность", format!("не тот владелец: {}", a.owner))
        }
        Mark::No => v.notes.push(format!("не тот владелец: {}", a.owner)),
        Mark::Partly => v
            .notes
            .push(format!("владелец рядом, но не тот: {}", a.owner)),
        Mark::Yes => {}
    }

    match action {
        Mark::No => v.notes.push("первый шаг не тот".into()),
        Mark::Partly => v.notes.push("первый шаг только частично".into()),
        Mark::Yes => {}
    }

    let claimed = format!("{} {}", a.evidence_quote, a.facts);
    match evidence {
        Evidence::Invented => v.flag("выдумка", "цитаты нет в тексте сообщения".into()),
        Evidence::Paraphrase => v.notes.push("цитата пересказана, а не дословно".into()),
        Evidence::Missing => v.notes.push("нет цитаты".into()),
        Evidence::Verbatim => {}
    }
    if invented_numbers(&claimed, &inc.report_text) {
        v.flag(
            "выдумка",
            "в фактах есть числа, которых нет в сообщении".into(),
        );
    }

    if !unknowns_ok {
        v.notes.push(if unknowns.is_empty() {
            "неизвестное не отмечено".into()
        } else {
            "не отмечено, что масштаб неизвестен".into()
        });
    }
    if needs_human && a.human_review != "да" {
        v.flag(
            "чрезмерная автоматизация",
            "не попросил проверку человеком, хотя масштаб не указан".into(),
        );
    } else if !needs_human && a.human_review == "да" {
        v.notes.push("лишняя эскалация".into());
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::stems;

    fn incident(sev: &str, owner: &str, action: &str, hr: &str, text: &str) -> Incident {
        Incident {
            incident_id: "INC-T".into(),
            report_text: text.into(),
            gold_severity: sev.into(),
            gold_owner: owner.into(),
            gold_first_action: action.into(),
            gold_human_review: hr.into(),
        }
    }

    fn rubric() -> Rubric {
        let mut r = Rubric::default();
        r.owners.entry("начальник смены".into()).or_default().yes = stems("начальник смены");
        r.actions.insert(
            "снизить скорость линии и проверить узел".into(),
            vec![stems("сниз"), stems("подшипник|узел")],
        );
        r
    }

    #[test]
    fn underestimate_of_high_is_dangerous() {
        let inc = incident(
            "Высокий",
            "начальник смены",
            "снизить скорость линии и проверить узел",
            "нет",
            "растёт вибрация подшипника",
        );
        let a = Answer {
            structured: true,
            severity: Some(Severity::Medium),
            owner: "начальник смены".into(),
            first_action: "снизить скорость и осмотреть подшипник".into(),
            evidence_quote: "растёт вибрация".into(),
            unknowns: "причина".into(),
            human_review: "нет".into(),
            ..Answer::default()
        };
        let v = score(&inc, &a, &rubric(), Fixed::default()).unwrap();
        assert_eq!(v.action, Mark::Yes);
        assert_eq!(v.evidence, Evidence::Verbatim);
        assert_eq!(v.sources, vec!["потеря факта".to_string()]);
    }

    #[test]
    fn missed_escalation_and_invented_numbers() {
        let inc = incident(
            "Низкий",
            "сервис-менеджер",
            "уточнить масштаб и собрать логи",
            "да",
            "отчёт открывается медленно",
        );
        let a = Answer {
            structured: true,
            severity: Some(Severity::Critical),
            evidence_quote: "отчёт открывается медленно у 200 человек".into(),
            human_review: "нет".into(),
            ..Answer::default()
        };
        let v = score(&inc, &a, &rubric(), Fixed::default()).unwrap();
        assert!(v.sources.contains(&"неверная срочность".to_string()));
        assert!(v.sources.contains(&"выдумка".to_string()));
        assert!(v.sources.contains(&"чрезмерная автоматизация".to_string()));
    }
}
