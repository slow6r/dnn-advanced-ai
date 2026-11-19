use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp, Triangular};
use rayon::prelude::*;
use serde::Serialize;

use crate::data::{Assumptions, Res};
use crate::guard::Route;
use crate::stats::{mean, quantile, sorted};

#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    pub gold: u8,
    pub model: u8,
    pub route: Route,
    pub error: bool,
    pub dangerous: bool,
}

pub struct Params {
    t_manual: Triangular<f64>,
    t_auto: Triangular<f64>,
    t_suggest: Triangular<f64>,
    t_human_extra: Triangular<f64>,
    t_fix: Triangular<f64>,
    p_catch_suggest: Triangular<f64>,
    p_catch_human: Triangular<f64>,
    lambdas: [f64; 3],
    shift: f64,
    shifts_per_month: f64,
    crit_target: f64,
}

fn get(a: &Assumptions, key: &str) -> Res<(f64, f64, f64)> {
    a.get(key)
        .copied()
        .ok_or_else(|| format!("в допущениях нет {key}").into())
}

fn tri(a: &Assumptions, key: &str) -> Res<Triangular<f64>> {
    let (lo, mode, hi) = get(a, key)?;
    Triangular::new(lo, hi, mode).map_err(|e| format!("{key}: {e}").into())
}

impl Params {
    pub fn new(a: &Assumptions) -> Res<Self> {
        let (l1, l2, l3) = get(a, "lambda_per_hour")?;
        if [l1, l2, l3].iter().any(|l| *l <= 0.0) {
            return Err("lambda_per_hour должна быть больше нуля".into());
        }
        Ok(Self {
            t_manual: tri(a, "t_manual")?,
            t_auto: tri(a, "t_auto")?,
            t_suggest: tri(a, "t_suggest")?,
            t_human_extra: tri(a, "t_human_extra")?,
            t_fix: tri(a, "t_fix")?,
            p_catch_suggest: tri(a, "p_catch_suggest")?,
            p_catch_human: tri(a, "p_catch_human")?,
            lambdas: [l1, l2, l3],
            shift: get(a, "shift_min")?.1,
            shifts_per_month: get(a, "shifts_per_month")?.1,
            crit_target: get(a, "crit_target_min")?.1,
        })
    }
}

#[derive(Default)]
struct Shift {
    messages: usize,
    busy: [f64; 2],
    overtime: [f64; 2],
    crit: [Vec<f64>; 2],
    uncaught_errors: u32,
    uncaught_dangerous: u32,
}

#[derive(Debug, Serialize)]
pub struct SimRow {
    pub lambda_per_hour: f64,
    pub mode: &'static str,
    pub messages_per_shift: f64,
    pub busy_min: f64,
    pub utilization: f64,
    pub overtime_min: f64,
    pub crit_first_action_p50: f64,
    pub crit_first_action_p90: f64,
    pub crit_late_share: f64,
    pub uncaught_errors_per_shift: Option<f64>,
    pub uncaught_dangerous_per_shift: Option<f64>,
    pub saved_min_p10: Option<f64>,
    pub saved_min_p50: Option<f64>,
    pub saved_min_p90: Option<f64>,
    pub saved_hours_month_p50: Option<f64>,
}

fn serve(ready: &[f64], prio: &[u8], service: &[f64]) -> Vec<f64> {
    let n = ready.len();
    let mut done = vec![0.0; n];
    let mut queue = BinaryHeap::new();
    let (mut t, mut next) = (0.0_f64, 0);
    while next < n || !queue.is_empty() {
        if queue.is_empty() {
            t = t.max(ready[next]);
        }
        while next < n && ready[next] <= t {
            queue.push((prio[next], Reverse(next)));
            next += 1;
        }
        if let Some((_, Reverse(i))) = queue.pop() {
            t += service[i];
            done[i] = t;
        }
    }
    done
}

fn simulate_shift(
    p: &Params,
    lambda: f64,
    outcomes: &[Outcome],
    latency: &[f64],
    rng: &mut SmallRng,
) -> Shift {
    let gap = Exp::new(lambda / 60.0).expect("lambda проверена в Params::new");
    let (mut arrival, mut picked) = (Vec::new(), Vec::new());
    let mut t = gap.sample(rng);
    while t <= p.shift {
        arrival.push(t);
        picked.push(outcomes[rng.random_range(0..outcomes.len())]);
        t += gap.sample(rng);
    }
    let n = arrival.len();
    let mut s = Shift {
        messages: n,
        ..Shift::default()
    };

    let manual: Vec<f64> = (0..n).map(|_| p.t_manual.sample(rng)).collect();
    let done_manual = serve(&arrival, &vec![0; n], &manual);

    let catch_suggest = p.p_catch_suggest.sample(rng);
    let catch_human = p.p_catch_human.sample(rng);
    let mut llm_free = 0.0_f64;
    let (mut ready, mut work) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for (o, &at) in picked.iter().zip(&arrival) {
        llm_free = llm_free.max(at) + latency[rng.random_range(0..latency.len())];
        ready.push(llm_free);
        let mut w = match o.route {
            Route::Auto => p.t_auto.sample(rng),
            Route::Suggest => p.t_suggest.sample(rng),
            Route::Human => p.t_manual.sample(rng) + p.t_human_extra.sample(rng),
        };
        if o.error {
            let caught = match o.route {
                Route::Auto => false,
                Route::Suggest => rng.random_bool(catch_suggest),
                Route::Human => rng.random_bool(catch_human),
            };
            if caught {
                w += p.t_fix.sample(rng);
            } else {
                s.uncaught_errors += 1;
                s.uncaught_dangerous += u32::from(o.dangerous);
            }
        }
        work.push(w);
    }
    let prio: Vec<u8> = picked.iter().map(|o| o.model).collect();
    let done_ai = serve(&ready, &prio, &work);

    for (i, o) in picked.iter().enumerate().filter(|(_, o)| o.gold == 3) {
        s.crit[0].push(done_manual[i] - arrival[i]);
        let first_action = if o.route == Route::Auto {
            ready[i]
        } else {
            done_ai[i]
        };
        s.crit[1].push(first_action - arrival[i]);
    }
    let last = |d: &[f64]| d.iter().copied().fold(0.0, f64::max);
    s.busy = [manual.iter().sum(), work.iter().sum()];
    s.overtime = [
        (last(&done_manual) - p.shift).max(0.0),
        (last(&done_ai) - p.shift).max(0.0),
    ];
    s
}

pub fn run(
    p: &Params,
    outcomes: &[Outcome],
    latency_min: &[f64],
    shifts: usize,
    seed: u64,
) -> Vec<SimRow> {
    let mut rows = Vec::new();
    for (li, &lambda) in p.lambdas.iter().enumerate() {
        let res: Vec<Shift> = (0..shifts)
            .into_par_iter()
            .map(|r| {
                let mut rng = SmallRng::seed_from_u64(seed ^ ((li as u64) << 40) ^ r as u64);
                simulate_shift(p, lambda, outcomes, latency_min, &mut rng)
            })
            .collect();
        let saved = sorted(res.iter().map(|s| s.busy[0] - s.busy[1]).collect());
        let messages = mean(res.iter().map(|s| s.messages as f64));
        for (m, mode) in ["вручную", "AI + guard"].into_iter().enumerate() {
            let crit = sorted(res.iter().flat_map(|s| s.crit[m].iter().copied()).collect());
            let late = crit.iter().filter(|&&x| x > p.crit_target).count() as f64
                / crit.len().max(1) as f64;
            let busy = mean(res.iter().map(|s| s.busy[m]));
            let ai = m == 1;
            rows.push(SimRow {
                lambda_per_hour: lambda,
                mode,
                messages_per_shift: messages,
                busy_min: busy,
                utilization: busy / p.shift,
                overtime_min: mean(res.iter().map(|s| s.overtime[m])),
                crit_first_action_p50: quantile(&crit, 0.5),
                crit_first_action_p90: quantile(&crit, 0.9),
                crit_late_share: late,
                uncaught_errors_per_shift: ai
                    .then(|| mean(res.iter().map(|s| f64::from(s.uncaught_errors)))),
                uncaught_dangerous_per_shift: ai
                    .then(|| mean(res.iter().map(|s| f64::from(s.uncaught_dangerous)))),
                saved_min_p10: ai.then(|| quantile(&saved, 0.1)),
                saved_min_p50: ai.then(|| quantile(&saved, 0.5)),
                saved_min_p90: ai.then(|| quantile(&saved, 0.9)),
                saved_hours_month_p50: ai
                    .then(|| quantile(&saved, 0.5) * p.shifts_per_month / 60.0),
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_queue_serves_critical_first() {
        let done = serve(&[0.0, 0.0, 0.0], &[0, 3, 1], &[1.0, 1.0, 1.0]);
        assert_eq!(done, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn fifo_waits_for_late_arrivals() {
        let done = serve(&[0.0, 10.0], &[0, 0], &[2.0, 2.0]);
        assert_eq!(done, vec![2.0, 12.0]);
    }
}
