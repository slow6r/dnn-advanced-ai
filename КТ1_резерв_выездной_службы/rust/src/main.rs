// КТ1, часть на Rust.
// Питоновский ноутбук делает сплит и сохраняет data/train.csv и data/test.csv,
// эта программа учит на них baseline (linfa) и MLP (candle), а метрики пишет
// в data/rust_metrics.csv и data/rust_predictions.csv. Ноутбук их потом читает.
//
// Запуск из папки rust:
//   cargo run --release -- ../data/train.csv ../data/test.csv \
//       ../data/rust_metrics.csv ../data/rust_predictions.csv

mod data;
mod logreg;
mod metrics;
mod mlp;

use std::env;
use std::fs;

type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

fn usage(prog: &str) {
    eprintln!("запуск: {prog} <train.csv> <test.csv> <metrics_out.csv> <predictions_out.csv>");
}

fn main() -> DynResult<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        usage(&args[0]);
        std::process::exit(1);
    }

    let (train, test) = data::load(&args[1], &args[2])?;
    let base_rate = test.y.iter().filter(|&&v| v > 0.5).count() as f64 / test.y.len() as f64;
    println!(
        "train: {} заявок ({:.1}% нарушений), test: {} заявок ({:.1}% нарушений)",
        train.x.len(),
        100.0 * train.y.iter().filter(|&&v| v > 0.5).count() as f64 / train.y.len() as f64,
        test.x.len(),
        100.0 * base_rate
    );

    println!("\n[1/2] baseline: логистическая регрессия (linfa)");
    let p_logreg = logreg::train_and_predict(&train.x, &train.y, &test.x)?;
    let m_logreg = metrics::evaluate(&p_logreg, &test.y);

    println!("\n[2/2] MLP {}-{}-1 (candle), {} эпох", data::N_IN, mlp::DEFAULT.hidden, mlp::DEFAULT.epochs);
    let p_mlp = mlp::train_and_predict(&train.x, &train.y, &test.x, &mlp::DEFAULT)?;
    let m_mlp = metrics::evaluate(&p_mlp, &test.y);

    println!("\nИтоги на тесте при очереди 20%:");
    println!(
        "{:<28} {:>9} {:>9} {:>11} {:>8}",
        "модель", "очередь", "recall", "precision", "AUC"
    );
    println!("{:-<70}", "");
    println!(
        "{:<28} {:>8.1}% {:>8.1}% {:>10.1}% {:>8.3}",
        "случайный отбор (для сравнения)", 20.0, 100.0 * base_rate, 100.0 * base_rate, 0.500
    );
    println!(
        "{:<28} {:>8.1}% {:>8.1}% {:>10.1}% {:>8.3}",
        "логрегрессия (linfa)",
        100.0 * m_logreg.queue,
        100.0 * m_logreg.recall,
        100.0 * m_logreg.precision,
        m_logreg.auc
    );
    println!(
        "{:<28} {:>8.1}% {:>8.1}% {:>10.1}% {:>8.3}",
        "MLP 9-12-1 (candle)",
        100.0 * m_mlp.queue,
        100.0 * m_mlp.recall,
        100.0 * m_mlp.precision,
        m_mlp.auc
    );

    // топ очереди: что MLP отправил бы в резерв
    let mut idx: Vec<usize> = (0..p_mlp.len()).collect();
    idx.sort_by(|&a, &b| p_mlp[b].partial_cmp(&p_mlp[a]).unwrap());
    println!("\nТоп-8 заявок по скору MLP (факт — что вышло на самом деле):");
    for &i in idx.iter().take(8) {
        println!(
            "  {}  риск {:5.1}%  факт: {}",
            test.ids[i],
            100.0 * p_mlp[i],
            if test.y[i] > 0.5 { "нарушение SLA" } else { "обошлось" }
        );
    }

    // метрики для ноутбука
    let mut csv = String::from("model,queue_pct,recall_pct,precision_pct,auc\n");
    csv.push_str(&format!(
        "случайный отбор,20.0,{:.1},{:.1},0.500\n",
        100.0 * base_rate,
        100.0 * base_rate
    ));
    csv.push_str(&format!(
        "логрегрессия (linfa),{:.1},{:.1},{:.1},{:.3}\n",
        100.0 * m_logreg.queue,
        100.0 * m_logreg.recall,
        100.0 * m_logreg.precision,
        m_logreg.auc
    ));
    csv.push_str(&format!(
        "MLP 9-12-1 (candle),{:.1},{:.1},{:.1},{:.3}\n",
        100.0 * m_mlp.queue,
        100.0 * m_mlp.recall,
        100.0 * m_mlp.precision,
        m_mlp.auc
    ));
    fs::write(&args[3], csv)?;
    println!("\nметрики записаны: {}", args[3]);

    // решения MLP по заявкам
    let queue = metrics::queue_indices(&p_mlp);
    let mut in_queue = vec![false; test.x.len()];
    for &i in &queue {
        in_queue[i] = true;
    }
    let mut csv = String::from("request_id,risk,fact,decision\n");
    for i in idx.iter() {
        csv.push_str(&format!(
            "{},{:.4},{},{}\n",
            test.ids[*i],
            p_mlp[*i],
            if test.y[*i] > 0.5 { 1 } else { 0 },
            if in_queue[*i] { 1 } else { 0 }
        ));
    }
    fs::write(&args[4], csv)?;
    println!("решения MLP записаны: {}", args[4]);

    Ok(())
}
