use std::collections::HashMap;
use std::error::Error;
use std::fs;

use serde::de::DeserializeOwned;
use serde::Deserialize;

use crate::text::{self, Stem};

pub type Res<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, Deserialize)]
pub struct Incident {
    pub incident_id: String,
    pub report_text: String,
    pub gold_severity: String,
    pub gold_owner: String,
    pub gold_first_action: String,
    pub gold_human_review: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawAnswer {
    pub incident_id: String,
    pub prompt_version: String,
    pub model: String,
    pub response: String,
    #[serde(default)]
    pub seconds: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manual {
    pub incident_id: String,
    pub prompt_version: String,
    pub model: String,
    pub field: String,
    pub value: String,
    pub reason: String,
}

#[derive(Default)]
pub struct OwnerRule {
    pub yes: Vec<Stem>,
    pub partial: Vec<Stem>,
}

#[derive(Default)]
pub struct Rubric {
    pub owners: HashMap<String, OwnerRule>,
    pub actions: HashMap<String, Vec<Vec<Stem>>>,
}

pub type Assumptions = HashMap<String, (f64, f64, f64)>;

#[derive(Deserialize)]
struct RubricRow {
    kind: String,
    gold: String,
    level: String,
    stems: String,
}

#[derive(Deserialize)]
struct AssumptionRow {
    param: String,
    min: f64,
    mode: f64,
    max: f64,
}

fn read_csv<T: DeserializeOwned>(path: &str) -> Res<Vec<T>> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut rdr = csv::Reader::from_reader(text.trim_start_matches('\u{feff}').as_bytes());
    let rows = rdr
        .deserialize()
        .collect::<Result<Vec<T>, _>>()
        .map_err(|e| format!("{path}: {e}"))?;
    Ok(rows)
}

pub fn incidents(path: &str) -> Res<Vec<Incident>> {
    read_csv(path)
}

pub fn manual(path: &str) -> Res<Vec<Manual>> {
    read_csv(path)
}

pub fn answers(path: &str) -> Res<Vec<RawAnswer>> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l).map_err(|e| format!("{path}, строка {}: {e}", i + 1).into())
        })
        .collect()
}

pub fn rubric(path: &str) -> Res<Rubric> {
    let mut r = Rubric::default();
    for row in read_csv::<RubricRow>(path)? {
        let stems = text::stems(&row.stems);
        match (row.kind.as_str(), row.level.as_str()) {
            ("owner", "да") => r.owners.entry(row.gold).or_default().yes = stems,
            ("owner", "частично") => r.owners.entry(row.gold).or_default().partial = stems,
            ("action", _) => r.actions.entry(row.gold).or_default().push(stems),
            (kind, level) => return Err(format!("{path}: непонятная строка {kind}/{level}").into()),
        }
    }
    Ok(r)
}

pub fn assumptions(path: &str) -> Res<Assumptions> {
    Ok(read_csv::<AssumptionRow>(path)?
        .into_iter()
        .map(|r| (r.param, (r.min, r.mode, r.max)))
        .collect())
}
