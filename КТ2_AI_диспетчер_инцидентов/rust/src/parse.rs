use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::text::{lower, norm, quote_found};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

const SEVERITY_STEMS: [(&str, Severity); 9] = [
    ("критическ", Severity::Critical),
    ("critical", Severity::Critical),
    ("высок", Severity::High),
    ("high", Severity::High),
    ("средн", Severity::Medium),
    ("умерен", Severity::Medium),
    ("medium", Severity::Medium),
    ("низк", Severity::Low),
    ("low", Severity::Low),
];

impl Severity {
    pub fn label(self) -> &'static str {
        match self {
            Severity::Low => "Низкий",
            Severity::Medium => "Средний",
            Severity::High => "Высокий",
            Severity::Critical => "Критический",
        }
    }

    pub fn rank(self) -> i32 {
        self as i32
    }

    pub fn from_value(s: &str) -> Option<Self> {
        let n = norm(s);
        if n.starts_with("некрит") {
            return Some(Severity::Low);
        }
        earliest_severity(&n).or_else(|| n.contains("крит").then_some(Severity::Critical))
    }
}

fn earliest_severity(n: &str) -> Option<Severity> {
    SEVERITY_STEMS
        .iter()
        .filter_map(|(stem, sev)| n.find(stem).map(|pos| (pos, *sev)))
        .min_by_key(|(pos, _)| *pos)
        .map(|(_, sev)| sev)
}

#[derive(Debug, Clone, Default)]
pub struct Answer {
    pub structured: bool,
    pub severity: Option<Severity>,
    pub severity_raw: String,
    pub owner: String,
    pub first_action: String,
    pub evidence_quote: String,
    pub unknowns: String,
    pub facts: String,
    pub human_review: String,
}

const REQUIRED: [&str; 6] = [
    "severity",
    "owner",
    "first_action",
    "evidence_quote",
    "unknowns",
    "human_review",
];

pub fn parse(version: &str, response: &str, source: &str) -> Answer {
    if !version.starts_with('A') {
        if let Some(map) = json_object(response) {
            return from_json(&map);
        }
    }
    from_free_text(response, source)
}

fn json_object(text: &str) -> Option<Map<String, Value>> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    match serde_json::from_str::<Value>(&text[start..=end]) {
        Ok(Value::Object(map)) => Some(map),
        _ => None,
    }
}

fn flat(v: &Value) -> String {
    match v {
        Value::String(s) => s.trim().to_string(),
        Value::Array(items) => join_non_empty(items.iter().map(flat), "; "),
        Value::Object(obj) => join_non_empty(obj.values().map(flat), "; "),
        Value::Bool(true) => "да".into(),
        Value::Bool(false) => "нет".into(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn join_non_empty(parts: impl Iterator<Item = String>, sep: &str) -> String {
    parts
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(sep)
}

fn field(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.iter()
        .find(|(k, _)| k.trim().eq_ignore_ascii_case(key))
        .map(|(_, v)| flat(v))
}

pub fn yes_no(s: &str) -> String {
    let n = norm(s);
    let no = n.starts_with("нет") || n.starts_with("не ") || n == "false" || n == "no";
    let yes = n.starts_with("да")
        || n == "true"
        || n == "yes"
        || n.contains("требуется")
        || n.contains("нужн");
    match (no, yes) {
        (true, _) => "нет".into(),
        (false, true) => "да".into(),
        _ => String::new(),
    }
}

fn empty_if_nothing(s: String) -> String {
    match norm(&s).trim_matches(|c: char| !c.is_alphanumeric()) {
        "" | "нет" | "none" | "отсутствуют" | "неизвестно" | "n/a" => {
            String::new()
        }
        _ => s,
    }
}

fn from_json(map: &Map<String, Value>) -> Answer {
    let get = |k: &str| field(map, k).unwrap_or_default();
    let severity_raw = get("severity");
    let severity = Severity::from_value(&severity_raw);
    let human_review = yes_no(&get("human_review"));
    let complete = REQUIRED.iter().all(|k| field(map, k).is_some());
    Answer {
        structured: complete && severity.is_some() && !human_review.is_empty(),
        severity,
        severity_raw,
        owner: get("owner"),
        first_action: get("first_action"),
        evidence_quote: get("evidence_quote"),
        unknowns: empty_if_nothing(get("unknowns")),
        facts: get("facts"),
        human_review,
    }
}

static SEVERITY_LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"критичност\w*|приоритет\w*|уров\w+\s+(опасност|важност)\w*").expect("regex")
});

static ACTION_LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(рекомендуем\w*\s+|первоочередн\w*\s+|немедленн\w*\s+|первое\s+|первый\s+)?(действи\w*|шаг\w*|меры|рекомендаци\w*)")
        .expect("regex")
});

static QUOTED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"["«“]([^"»”]{6,})["»”]"#).expect("regex"));

static UNKNOWN_SENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"неизвестн|не\s+указан|не\s+ясн|неясн|нет\s+(данных|информации|сведений)|не\s+удалось|не\s+сообща").expect("regex")
});

static ESCALATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"уточн\w*|недостаточно\s+(данных|информации|сведений)|не\s+хватает|нет\s+(данных|информации)|эскал\w*|проверк\w*\s+человек\w*")
        .expect("regex")
});

static OWNERS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"дежурн\w*", "дежурный инженер"),
        (r"(служб|отдел|специалист)\w*\s+(по\s+)?(информационной\s+)?безопасност\w*|\bиб\b|кибербезопасност\w*|информационн\w*\s+безопасност\w*", "служба безопасности"),
        (r"\b(ит|it)[- ]?(отдел|служб|специалист|подразделени|поддержк|команд|департамент)\w*|отдел\w*\s+(ит|it)\b", "ИТ-отдел"),
        (r"техподдержк\w*|техническ\w*\s+поддержк\w*|служб\w*\s+поддержк\w*|поддержк\w*\s+пользовател\w*", "техподдержка"),
        (r"начальник\w*\s+(смены|цеха|производства|участка|линии)|мастер\w*\s+(смены|цеха|участка)|сменн\w*\s+мастер\w*", "начальник смены"),
        (r"механик\w*|ремонтн\w*\s+(служб|бригад|персонал)\w*", "ремонтная служба"),
        (r"(руководител|начальник|заведующ|сотрудник|работник|персонал|менеджер)\w*\s+склад\w*", "персонал склада"),
        (r"диспетчер\w*|логистическ\w*\s+(отдел|служб)\w*|отдел\w*\s+логистик\w*|логист(ы|ов|ам)\b", "диспетчер логистики"),
        (r"сервис[- ]?менеджер\w*", "сервис-менеджер"),
        (r"администратор\w*", "администратор"),
        (r"разработчик\w*|команд\w*\s+разработк\w*", "разработчики"),
        (r"инженер\w*", "инженер"),
    ]
    .into_iter()
    .map(|(p, label)| (Regex::new(p).expect("regex"), label))
    .collect()
});

fn clean_line(line: &str) -> String {
    let s = line.trim().trim_start_matches(|c: char| {
        matches!(c, '-' | '•' | '*' | '#' | '>') || c.is_whitespace()
    });
    let digits = s.chars().take_while(char::is_ascii_digit).count();
    let s = match s[digits..].strip_prefix(['.', ')']) {
        Some(rest) if digits > 0 => rest,
        _ => s,
    };
    s.replace("**", "").trim().to_string()
}

fn free_severity(n: &str) -> Option<Severity> {
    SEVERITY_LABEL
        .find_iter(n)
        .find_map(|m| earliest_severity(&n[m.end()..].chars().take(40).collect::<String>()))
        .or_else(|| earliest_severity(n))
}

fn free_action(text: &str) -> String {
    let lines: Vec<String> = text.lines().map(clean_line).collect();
    for (i, line) in lines.iter().enumerate() {
        let low = lower(line);
        let Some(m) = ACTION_LABEL.find(&low) else {
            continue;
        };
        let rest = if line.is_char_boundary(m.end()) {
            &line[m.end()..]
        } else {
            ""
        };
        let rest = rest
            .trim_start_matches(|c: char| matches!(c, ':' | '-' | '—' | '–') || c.is_whitespace());
        if rest.chars().count() > 3 {
            return rest.trim().to_string();
        }
        let next: Vec<&str> = lines[i + 1..]
            .iter()
            .map(String::as_str)
            .filter(|l| !l.is_empty())
            .take(2)
            .collect();
        if !next.is_empty() {
            return next.join("; ");
        }
    }
    lines
        .iter()
        .find(|l| {
            ["необходимо", "следует", "рекоменд", "нужно"]
                .iter()
                .any(|w| lower(l).contains(w))
        })
        .cloned()
        .unwrap_or_default()
}

fn free_owner(n: &str) -> String {
    OWNERS
        .iter()
        .filter_map(|(re, label)| re.find(n).map(|m| (m.start(), *label)))
        .min_by_key(|(pos, _)| *pos)
        .map(|(_, label)| label.to_string())
        .unwrap_or_default()
}

fn free_evidence(text: &str, source: &str) -> String {
    let found: Vec<&str> = QUOTED
        .captures_iter(text)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim())
        .filter(|q| quote_found(q, source))
        .collect();
    found.join(" ... ")
}

fn free_unknowns(text: &str) -> String {
    let parts: Vec<&str> = text
        .split(['.', '!', '?', '\n'])
        .map(str::trim)
        .filter(|s| UNKNOWN_SENT.is_match(&norm(s)))
        .take(2)
        .collect();
    parts.join("; ")
}

fn from_free_text(text: &str, source: &str) -> Answer {
    let n = norm(text);
    let severity = free_severity(&n);
    Answer {
        structured: false,
        severity,
        severity_raw: severity
            .map(Severity::label)
            .unwrap_or_default()
            .to_string(),
        owner: free_owner(&n),
        first_action: free_action(text),
        evidence_quote: free_evidence(text, source),
        unknowns: free_unknowns(text),
        facts: String::new(),
        human_review: if ESCALATION.is_match(&n) {
            "да".into()
        } else {
            "нет".into()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_inside_code_fence() {
        let r = "```json\n{\"severity\": \"Высокий\", \"owner\": \"начальник смены\", \"first_action\": \"снизить скорость\", \
                 \"evidence_quote\": \"растёт вибрация\", \"unknowns\": [\"причина\"], \"human_review\": false}\n```";
        let a = parse("B_структура", r, "На линии растёт вибрация");
        assert!(a.structured);
        assert_eq!(a.severity, Some(Severity::High));
        assert_eq!(a.human_review, "нет");
        assert_eq!(a.unknowns, "причина");
    }

    #[test]
    fn free_text_label_wins_over_other_words() {
        let r = "Высокая температура подшипника опасна.\n**Критичность:** средняя\n**Действие:** снизить скорость линии";
        let a = parse("A_короткая", r, "");
        assert_eq!(a.severity, Some(Severity::Medium));
        assert_eq!(a.first_action, "снизить скорость линии");
    }

    #[test]
    fn free_text_action_list_and_owner() {
        let r = "Критичность: высокая.\n\nДействия:\n1. Сообщить в службу информационной безопасности.\n2. Заблокировать учётную запись.";
        let a = parse("A_короткая", r, "");
        assert_eq!(
            a.first_action,
            "Сообщить в службу информационной безопасности.; Заблокировать учётную запись."
        );
        assert_eq!(a.owner, "служба безопасности");
    }

    #[test]
    fn yes_no_variants() {
        assert_eq!(yes_no("не требуется"), "нет");
        assert_eq!(yes_no("Да"), "да");
        assert_eq!(yes_no("требуется"), "да");
        assert_eq!(yes_no("может быть"), "");
    }
}
