// Чтение CSV через polars и подготовка признаков.
// Сплит train/test делает питоновский ноутбук, я читаю готовые файлы,
// чтобы метрики можно было сравнить со sklearn напрямую.

use polars::prelude::*;

pub const N_NUM: usize = 5; // числовые признаки
pub const N_IN: usize = N_NUM + 4; // + one-hot приоритета (3) + наличие запчасти

pub struct Split {
    pub ids: Vec<String>,
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

fn read_csv(path: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()
}

// в csv все числа целые, поэтому сначала перевожу в f64
fn col_f64(df: &DataFrame, name: &str) -> PolarsResult<Vec<f64>> {
    let c = df.column(name)?.cast(&DataType::Float64)?;
    Ok(c.f64()?.into_no_null_iter().collect())
}

fn col_str(df: &DataFrame, name: &str) -> PolarsResult<Vec<String>> {
    // в строковых колонках into_no_null_iter не работает, беру обычный iter
    Ok(df.column(name)?
        .str()?
        .iter()
        .flatten()
        .map(|s| s.to_string())
        .collect())
}

struct Raw {
    ids: Vec<String>,
    nums: Vec<Vec<f64>>, // 5 числовых признаков
    priority: Vec<usize>, // 0 низкий, 1 средний, 2 высокий
    spare: Vec<f64>,
    y: Vec<f64>,
}

fn read_raw(path: &str) -> PolarsResult<Raw> {
    let df = read_csv(path)?;
    let cols = [
        "distance_km",
        "engineer_experience_years",
        "equipment_age_years",
        "repeat_request",
        "regional_load_pct",
    ];
    let mut nums = Vec::with_capacity(cols.len());
    for c in cols {
        nums.push(col_f64(&df, c)?);
    }
    // переставляю колонки в строки
    let n = df.height();
    let nums: Vec<Vec<f64>> = (0..n)
        .map(|i| nums.iter().map(|c| c[i]).collect())
        .collect();

    let priority = col_str(&df, "priority")?
        .iter()
        .map(|p| match p.as_str() {
            "низкий" => 0,
            "средний" => 1,
            "высокий" => 2,
            other => panic!("неизвестный приоритет: {other}"),
        })
        .collect();

    Ok(Raw {
        ids: col_str(&df, "request_id")?,
        nums,
        priority,
        spare: col_f64(&df, "spare_part_available")?,
        y: col_f64(&df, "sla_breach")?,
    })
}

// среднее и std считаю по train. По тесту нельзя, будет утечка.
fn num_stats(nums: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = nums.len() as f64;
    let mut mean = vec![0.0; N_NUM];
    for row in nums {
        for j in 0..N_NUM {
            mean[j] += row[j];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut std = vec![0.0; N_NUM];
    for row in nums {
        for j in 0..N_NUM {
            std[j] += (row[j] - mean[j]).powi(2);
        }
    }
    for s in &mut std {
        *s = (*s / n).sqrt().max(1e-12);
    }
    (mean, std)
}

fn features(r: &Raw, i: usize, mean: &[f64], std: &[f64]) -> Vec<f64> {
    let mut x = Vec::with_capacity(N_IN);
    for j in 0..N_NUM {
        x.push((r.nums[i][j] - mean[j]) / std[j]);
    }
    // one-hot, руками
    for k in 0..3 {
        x.push(if r.priority[i] == k { 1.0 } else { 0.0 });
    }
    x.push(r.spare[i]);
    x
}

pub fn load(train_path: &str, test_path: &str) -> PolarsResult<(Split, Split)> {
    let train = read_raw(train_path)?;
    let test = read_raw(test_path)?;

    let (mean, std) = num_stats(&train.nums);
    let train_split = Split {
        ids: train.ids.clone(),
        x: (0..train.nums.len()).map(|i| features(&train, i, &mean, &std)).collect(),
        y: train.y.clone(),
    };
    let test_split = Split {
        ids: test.ids.clone(),
        x: (0..test.nums.len()).map(|i| features(&test, i, &mean, &std)).collect(),
        y: test.y.clone(),
    };
    Ok((train_split, test_split))
}
