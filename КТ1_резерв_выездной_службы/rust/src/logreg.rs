// Baseline на linfa. То же, что LogisticRegression в sklearn.

use linfa::prelude::Fit;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2};

pub fn train_and_predict(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_test: &[Vec<f64>],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let d = x_train[0].len();

    let flat_train: Vec<f64> = x_train.iter().flatten().copied().collect();
    let x_tr = Array2::from_shape_vec((x_train.len(), d), flat_train)?;
    let y_tr: Array1<bool> = y_train.iter().map(|&v| v > 0.5).collect();

    let flat_test: Vec<f64> = x_test.iter().flatten().copied().collect();
    let x_te = Array2::from_shape_vec((x_test.len(), d), flat_test)?;

    let dataset = linfa::Dataset::from((x_tr, y_tr));
    let model = LogisticRegression::default()
        .alpha(0.001) // почти без регуляризации
        .max_iterations(500)
        .fit(&dataset)?;

    // тут я долго тупил: linfa сам решает, какой класс считать положительным
    // (для bool он выбрал false), и predict_probabilities возвращает вероятность
    // именно этого класса. Поэтому смотрю labels() и разворачиваю вероятности.
    let raw = model.predict_probabilities(&x_te);
    let p: Vec<f64> = if model.labels().pos.class {
        raw.to_vec()
    } else {
        raw.iter().map(|p| 1.0 - p).collect()
    };
    Ok(p)
}
