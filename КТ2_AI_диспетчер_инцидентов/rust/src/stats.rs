use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

pub fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.retain(|x| x.is_finite());
    v.sort_by(f64::total_cmp);
    v
}

pub fn mean(values: impl IntoIterator<Item = f64>) -> f64 {
    let (sum, n) = values
        .into_iter()
        .fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

pub fn cluster_ci(clusters: &[Vec<f64>], resamples: usize, seed: u64) -> (f64, f64) {
    let k = clusters.len();
    if k == 0 {
        return (f64::NAN, f64::NAN);
    }
    let stats: Vec<f64> = (0..resamples)
        .into_par_iter()
        .map(|b| {
            let mut rng =
                SmallRng::seed_from_u64(seed.wrapping_mul(1_000_003).wrapping_add(b as u64));
            let (mut sum, mut n) = (0.0, 0usize);
            for _ in 0..k {
                let c = &clusters[rng.random_range(0..k)];
                sum += c.iter().sum::<f64>();
                n += c.len();
            }
            if n == 0 {
                f64::NAN
            } else {
                sum / n as f64
            }
        })
        .collect();
    let s = sorted(stats);
    (quantile(&s, 0.025), quantile(&s, 0.975))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_interpolates() {
        let s = sorted(vec![3.0, 1.0, 2.0, 4.0]);
        assert_eq!(quantile(&s, 0.0), 1.0);
        assert_eq!(quantile(&s, 1.0), 4.0);
        assert!((quantile(&s, 0.5) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn ci_of_constant_clusters_is_degenerate() {
        let c = vec![vec![1.0, 1.0], vec![1.0]];
        assert_eq!(cluster_ci(&c, 200, 1), (1.0, 1.0));
    }
}
