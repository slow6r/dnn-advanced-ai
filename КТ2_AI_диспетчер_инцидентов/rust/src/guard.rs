use crate::parse::{Answer, Severity};
use crate::score::Evidence;
use crate::text::{has_any, norm};

const URGENCY: &[&str] = &["срочн", "немедленно", "asap", "горит", "!!!"];

const RISK: &[&str] = &[
    "не отвеча",
    "перестал",
    "недоступ",
    "не работает",
    "остановил",
    "авари",
    "пожар",
    "дым",
    "утечк",
    "травм",
    "вибрац",
    "температур",
    "перегрев",
    "давлени",
    "искрит",
    "взлом",
    "вход",
    "неуспешн",
    "подозрит",
    "атак",
    "вирус",
    "шифрова",
];

const DISRUPTIVE: &[&str] = &[
    "останов",
    "отключ",
    "изолир",
    "заблок",
    "блокир",
    "откат",
    "перезагруз",
    "перезапуст",
    "удал",
    "сниз",
    "переключ",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Route {
    Auto,
    Suggest,
    Human,
}

impl Route {
    pub const ALL: [Route; 3] = [Route::Auto, Route::Suggest, Route::Human];

    pub fn label(self) -> &'static str {
        match self {
            Route::Auto => "авто",
            Route::Suggest => "предложить",
            Route::Human => "человек",
        }
    }
}

pub fn route(report_text: &str, a: &Answer, evidence: Evidence) -> (Route, Vec<&'static str>) {
    let text = norm(report_text);
    let low_priority = a.severity.is_none_or(|s| s <= Severity::Medium);
    let checks = [
        (!a.structured, "нет структуры"),
        (evidence != Evidence::Verbatim, "цитата не найдена в тексте"),
        (a.human_review == "да", "модель просит человека"),
        (has_any(&text, URGENCY), "автор давит срочностью"),
        (
            has_any(&text, RISK) && low_priority,
            "признаки опасности при низком приоритете",
        ),
    ];
    let human: Vec<&'static str> = checks
        .iter()
        .filter(|(hit, _)| *hit)
        .map(|(_, why)| *why)
        .collect();
    if !human.is_empty() {
        return (Route::Human, human);
    }
    if a.severity >= Some(Severity::High) {
        return (Route::Suggest, vec!["высокий приоритет"]);
    }
    if has_any(&norm(&a.first_action), DISRUPTIVE) {
        return (Route::Suggest, vec!["шаг меняет работу системы"]);
    }
    (Route::Auto, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(sev: Severity, hr: &str, action: &str) -> Answer {
        Answer {
            structured: true,
            severity: Some(sev),
            human_review: hr.into(),
            first_action: action.into(),
            ..Answer::default()
        }
    }

    #[test]
    fn danger_words_with_low_priority_go_to_human() {
        let (r, why) = route(
            "перестал отвечать основной сервис",
            &answer(Severity::Medium, "нет", "проверить"),
            Evidence::Verbatim,
        );
        assert_eq!(r, Route::Human);
        assert_eq!(why, vec!["признаки опасности при низком приоритете"]);
    }

    #[test]
    fn urgency_always_goes_to_human() {
        let text = "отчёт открывается медленно. Автор пишет: «срочно»";
        let (r, _) = route(
            text,
            &answer(Severity::Low, "нет", "уточнить масштаб"),
            Evidence::Verbatim,
        );
        assert_eq!(r, Route::Human);
    }

    #[test]
    fn clean_medium_case_is_auto_and_high_is_suggest() {
        let text = "Несколько машин задерживаются из-за перекрытия трассы";
        assert_eq!(
            route(
                text,
                &answer(Severity::Medium, "нет", "проверить маршрут"),
                Evidence::Verbatim
            )
            .0,
            Route::Auto
        );
        assert_eq!(
            route(
                text,
                &answer(Severity::High, "нет", "проверить маршрут"),
                Evidence::Verbatim
            )
            .0,
            Route::Suggest
        );
        assert_eq!(
            route(
                text,
                &answer(Severity::Medium, "нет", "проверить маршрут"),
                Evidence::Missing
            )
            .0,
            Route::Human
        );
    }
}
