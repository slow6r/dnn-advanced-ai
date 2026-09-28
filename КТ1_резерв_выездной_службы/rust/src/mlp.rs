// MLP на candle: 9 -> 12 (tanh) -> 1.
// Учусь полным батчем. Шаг SGD с инерцией написал сам, чтобы понять, как это работает.
// Seed фиксированный, чтобы результат повторялся.

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{loss, Linear, Module};
use rand::{Rng, SeedableRng};

pub struct Params {
    pub hidden: usize,
    pub epochs: usize,
    pub lr: f64,
    pub momentum: f64,
    pub seed: u64,
}

pub const DEFAULT: Params = Params { hidden: 12, epochs: 3000, lr: 0.1, momentum: 0.9, seed: 42 };

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

// v = m*v + g; w = w - lr*v
fn sgd_step(v: &Var, grad: Option<&Tensor>, vel: &mut Tensor, lr: f64, m: f64) -> candle_core::Result<()> {
    if let Some(g) = grad {
        let new_vel = ((&*vel * m)? + g)?;
        let upd = (&new_vel * lr)?;
        v.set(&(v.as_tensor() - &upd)?)?;
        *vel = new_vel;
    }
    Ok(())
}

pub fn train_and_predict(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_test: &[Vec<f64>],
    p: &Params,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let n_in = x_train[0].len();

    let flat: Vec<f64> = x_train.iter().flatten().copied().collect();
    let x = Tensor::from_vec(flat, (x_train.len(), n_in), &dev)?;
    let y = Tensor::from_vec(y_train.to_vec(), x_train.len(), &dev)?;

    // веса: случайные числа от -1/sqrt(n) до 1/sqrt(n)
    let mut rng = rand::rngs::StdRng::seed_from_u64(p.seed);
    let s1 = 1.0 / (n_in as f64).sqrt();
    let s2 = 1.0 / (p.hidden as f64).sqrt();
    let mut rnd = |len: usize, s: f64| -> Vec<f64> {
        (0..len).map(|_| rng.gen_range(-s..s)).collect()
    };
    let w1 = Var::from_tensor(&Tensor::from_vec(rnd(p.hidden * n_in, s1), (p.hidden, n_in), &dev)?)?;
    let b1 = Var::from_tensor(&Tensor::zeros((p.hidden,), DType::F64, &dev)?)?;
    let w2 = Var::from_tensor(&Tensor::from_vec(rnd(p.hidden, s2), (1, p.hidden), &dev)?)?;
    let b2 = Var::from_tensor(&Tensor::zeros((1,), DType::F64, &dev)?)?;

    let l1 = Linear::new(w1.as_tensor().clone(), Some(b1.as_tensor().clone()));
    let l2 = Linear::new(w2.as_tensor().clone(), Some(b2.as_tensor().clone()));

    // скорости для инерции
    let mut vel_w1 = Tensor::zeros((p.hidden, n_in), DType::F64, &dev)?;
    let mut vel_b1 = Tensor::zeros((p.hidden,), DType::F64, &dev)?;
    let mut vel_w2 = Tensor::zeros((1, p.hidden), DType::F64, &dev)?;
    let mut vel_b2 = Tensor::zeros((1,), DType::F64, &dev)?;

    for ep in 0..p.epochs {
        let h = l1.forward(&x)?.tanh()?;
        let logits = l2.forward(&h)?;
        let loss = loss::binary_cross_entropy_with_logit(&logits.flatten_to(1)?, &y)?;

        if ep % 500 == 0 {
            println!("  эпоха {:5}  logloss = {:.5}", ep, loss.to_scalar::<f64>()?);
        }

        let grads = loss.backward()?;
        sgd_step(&w1, grads.get(w1.as_tensor()), &mut vel_w1, p.lr, p.momentum)?;
        sgd_step(&b1, grads.get(b1.as_tensor()), &mut vel_b1, p.lr, p.momentum)?;
        sgd_step(&w2, grads.get(w2.as_tensor()), &mut vel_w2, p.lr, p.momentum)?;
        sgd_step(&b2, grads.get(b2.as_tensor()), &mut vel_b2, p.lr, p.momentum)?;
    }

    // прогноз на тесте: логиты прогоняю через сигмоиду
    let flat_test: Vec<f64> = x_test.iter().flatten().copied().collect();
    let xt = Tensor::from_vec(flat_test, (x_test.len(), n_in), &dev)?;
    let logits = l2.forward(&l1.forward(&xt)?.tanh()?)?;
    let raw = logits.flatten_all()?.to_vec1::<f64>()?;
    Ok(raw.iter().map(|z| sigmoid(*z)).collect())
}
