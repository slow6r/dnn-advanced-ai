// Метрики: очередь 20%, recall, precision, AUC. Посчитал сам, формулы простые.

pub struct Metrics {
    pub queue: f64,
    pub recall: f64,
    pub precision: f64,
    pub auc: f64,
}

pub fn evaluate(p: &[f64], y: &[f64]) -> Metrics {
    let n = p.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());

    // очередь: каждая пятая заявка
    let k = ((0.20 * n as f64).ceil() as usize).min(n);
    let hits = idx[..k].iter().filter(|&&i| y[i] > 0.5).count();
    let total_pos = y.iter().filter(|&&v| v > 0.5).count();

    // AUC: сколько раз случайное нарушение получило скор выше случайной
    // спокойной заявки. Ничьи считаю за полпобеды.
    let mut wins = 0.0f64;
    let mut pairs = 0.0f64;
    for i in 0..n {
        if y[i] > 0.5 {
            for j in 0..n {
                if y[j] <= 0.5 {
                    pairs += 1.0;
                    wins += if p[i] > p[j] {
                        1.0
                    } else if (p[i] - p[j]).abs() < 1e-12 {
                        0.5
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    Metrics {
        queue: k as f64 / n as f64,
        recall: hits as f64 / total_pos.max(1) as f64,
        precision: hits as f64 / k as f64,
        auc: wins / pairs.max(1.0),
    }
}

// индексы заявок, которые попали в очередь (для файла с решениями)
pub fn queue_indices(p: &[f64]) -> Vec<usize> {
    let n = p.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
    let k = ((0.20 * n as f64).ceil() as usize).min(n);
    idx[..k].to_vec()
}
