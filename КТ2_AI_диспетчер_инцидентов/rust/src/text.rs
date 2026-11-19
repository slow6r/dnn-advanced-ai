use std::collections::HashSet;

const TRIM: &[char] = &['"', '.', ',', ';', ':', ' ', '!', '?', '(', ')'];

pub fn lower(s: &str) -> String {
    s.chars()
        .flat_map(char::to_lowercase)
        .map(|c| if c == 'ё' { 'е' } else { c })
        .collect()
}

pub fn norm(s: &str) -> String {
    let mapped: String = lower(s)
        .chars()
        .map(|c| match c {
            '«' | '»' | '“' | '”' | '„' | '\'' => '"',
            '–' | '—' => '-',
            '*' | '#' | '`' => ' ',
            c => c,
        })
        .collect();
    mapped.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn tokens(normed: &str) -> Vec<&str> {
    normed
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect()
}

pub fn has_any(normed: &str, words: &[&str]) -> bool {
    words.iter().any(|w| normed.contains(w))
}

#[derive(Debug, Clone)]
pub struct Stem {
    text: String,
    exact: bool,
}

impl Stem {
    fn parse(raw: &str) -> Option<Self> {
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }
        Some(match raw.strip_prefix('=') {
            Some(rest) => Self {
                text: norm(rest),
                exact: true,
            },
            None => Self {
                text: norm(raw),
                exact: false,
            },
        })
    }
}

pub fn stems(field: &str) -> Vec<Stem> {
    field.split('|').filter_map(Stem::parse).collect()
}

pub fn hits(stems: &[Stem], text: &str) -> bool {
    let n = norm(text);
    let toks = tokens(&n);
    stems.iter().any(|s| {
        if s.exact {
            toks.contains(&s.text.as_str())
        } else {
            n.contains(&s.text)
        }
    })
}

pub fn quote_found(quote: &str, source: &str) -> bool {
    let src = norm(source);
    let q = norm(quote);
    let parts: Vec<&str> = q
        .split("...")
        .flat_map(|p| p.split('…'))
        .map(|p| p.trim_matches(TRIM))
        .filter(|p| p.chars().count() >= 3)
        .collect();
    !parts.is_empty() && parts.iter().all(|p| src.contains(p))
}

fn stem5(w: &str) -> String {
    w.chars().take(5).collect()
}

pub fn overlap(quote: &str, source: &str) -> f64 {
    let src_n = norm(source);
    let src: HashSet<String> = tokens(&src_n).into_iter().map(stem5).collect();
    let q = norm(quote);
    let words: Vec<&str> = tokens(&q)
        .into_iter()
        .filter(|w| w.chars().count() >= 3)
        .collect();
    if words.is_empty() {
        return 0.0;
    }
    words.iter().filter(|w| src.contains(&stem5(w))).count() as f64 / words.len() as f64
}

fn numbers(s: &str) -> HashSet<&str> {
    s.split(|c: char| !c.is_ascii_digit())
        .filter(|x| !x.is_empty())
        .collect()
}

pub fn invented_numbers(answer: &str, source: &str) -> bool {
    let src = numbers(source);
    numbers(answer).iter().any(|n| !src.contains(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_with_other_quotes_and_case() {
        let src = "Один пользователь сообщает о медленном открытии отчёта. Автор пишет: «срочно».";
        assert!(quote_found("\"медленном открытии отчета\"", src));
        assert!(quote_found(
            "Один пользователь ... автор пишет: \"срочно\"",
            src
        ));
        assert!(!quote_found("пользователи не могут войти", src));
    }

    #[test]
    fn exact_stem_does_not_match_inside_word() {
        let s = stems("=иб|склад");
        assert!(hits(&s, "служба ИБ"));
        assert!(!hits(&s, "ошибка"));
        assert!(hits(&s, "руководителю склада"));
    }

    #[test]
    fn numbers_not_from_source() {
        assert!(invented_numbers(
            "затронуто 50 клиентов",
            "затронуты корпоративные клиенты"
        ));
        assert!(!invented_numbers("клиенты затронуты", "затронуты клиенты"));
    }
}
