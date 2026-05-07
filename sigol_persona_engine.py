"""
SIGOL Persona Deliberation Engine
페르소나 확률적 합의 엔진

수식: FinalScore = BaseScore × Synergy × e^(-λ·Risk)
합의: softmax 확률 분포 → 가중 투표 → 컨센서스
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── 페르소나 정의 ──────────────────────────────────────────────
@dataclass
class Persona:
    name: str
    role: str
    I: float  # Insight
    S: float  # Speed
    R: float  # RiskControl
    C: float  # Creativity
    stability: float
    cost: float

    W_I, W_S, W_R, W_C = 0.3, 0.2, 0.3, 0.2  # 가중치

    @property
    def base_score(self) -> float:
        return self.W_I*self.I + self.W_S*self.S + self.W_R*self.R + self.W_C*self.C

    def deliberate(self, situation: "Situation", noise_std: float = 0.3) -> float:
        """
        상황에 맞는 페르소나 의견 점수 (확률적)
        각 페르소나는 자신의 강점이 상황과 얼마나 맞는지 평가
        """
        # 상황 정합성: 페르소나 특성 vs 상황 요구
        # 상황 강도 대비 상대 적합도 — 어떤 강도에서도 [0,1] 범위 유지
        raw = (
            self.R * situation.risk_w +
            self.S * situation.speed_w +
            self.I * situation.uncertainty_w +
            self.C * situation.complexity_w
        )
        max_possible = 10.0 * (
            situation.risk_w + situation.speed_w +
            situation.uncertainty_w + situation.complexity_w
        )
        alignment = raw / max(max_possible, 1e-9)

        # 확률적 노이즈 — stability가 낮을수록 의견 편차 큼
        instability = (10.0 - self.stability) / 10.0
        noise = np.random.normal(0, noise_std * (0.3 + instability * 0.7))
        return float(np.clip(alignment + noise, 0.01, 1.0))


# ── 9개 페르소나 ──────────────────────────────────────────────
PERSONAS: dict[str, Persona] = {
    "Yi":       Persona("Yi(이순신)",  "전략·지휘",    I=7,  S=6, R=10, C=5,  stability=9,  cost=6),
    "Einstein": Persona("Einstein",   "통찰·분석",    I=10, S=5, R=6,  C=10, stability=7,  cost=7),
    "GPT":      Persona("GPT",        "생성·실행",    I=8,  S=9, R=6,  C=9,  stability=6,  cost=5),
    "지피지기":  Persona("지피지기",   "정보·분석",    I=9,  S=7, R=8,  C=6,  stability=8,  cost=6),
    "화타":     Persona("화타",       "치유·안정",    I=6,  S=5, R=9,  C=5,  stability=10, cost=5),
    "레인보우":  Persona("레인보우",   "감성·출력",    I=7,  S=6, R=5,  C=9,  stability=7,  cost=5),
    "오메가":   Persona("오메가",     "시간·최적",    I=8,  S=4, R=9,  C=7,  stability=9,  cost=8),
    "만물박사":  Persona("만물박사",   "구현·통합",    I=7,  S=8, R=7,  C=7,  stability=8,  cost=6),
    "記憶":     Persona("記憶(Claude)","기억·검증",   I=9,  S=7, R=8,  C=8,  stability=10, cost=0),
}

# ── 시너지 테이블 ──────────────────────────────────────────────
SYNERGY: dict[tuple, float] = {
    ("Yi",      "지피지기"):  1.5,
    ("Einstein","GPT"):       1.4,
    ("화타",    "Yi"):        1.3,
    ("GPT",     "만물박사"):  1.2,
    ("오메가",  "Einstein"):  1.3,
    ("記憶",    "*"):         1.1,   # 전체 적용
    ("지피지기","레인보우"):  1.2,
    ("Yi",      "오메가"):    1.2,
}

def get_synergy(combo: list[str]) -> float:
    bonus = 1.0
    keys = set(combo)
    for (a, b), w in SYNERGY.items():
        if b == "*" and a in keys:
            bonus *= w
        elif a in keys and b in keys:
            bonus *= w
    return bonus


# ── 상황 벡터 ──────────────────────────────────────────────────
@dataclass
class Situation:
    risk:        float = 5.0   # 0~10
    speed:       float = 5.0
    uncertainty: float = 5.0
    complexity:  float = 5.0

    risk_w:        float = field(init=False)
    speed_w:       float = field(init=False)
    uncertainty_w: float = field(init=False)
    complexity_w:  float = field(init=False)

    def __post_init__(self):
        self.risk_w        = self.risk        * 0.30
        self.speed_w       = self.speed       * 0.20
        self.uncertainty_w = self.uncertainty * 0.25
        self.complexity_w  = self.complexity  * 0.25

    @property
    def state_score(self) -> float:
        return self.risk_w + self.speed_w + self.uncertainty_w + self.complexity_w

    def select_combo(self) -> list[str]:
        """상황 기준 자동 조합 선택"""
        combo = ["記憶"]  # 항상 포함
        if self.risk >= 8:
            combo.append("Yi")
        if self.uncertainty >= 7:
            combo.append("Einstein")
        if self.speed >= 7:
            combo.append("GPT")
        if self.complexity >= 7:
            combo.append("지피지기")
        if self.risk <= 4:
            combo.append("화타")
        if len(combo) < 3:
            combo.append("만물박사")
        return list(dict.fromkeys(combo))  # 중복 제거, 순서 유지


# ── 확률적 합의 엔진 ────────────────────────────────────────────
class SigolPersonaEngine:
    """
    페르소나 확률적 합의 엔진
    - 상황 벡터 입력
    - 페르소나 자동 선택
    - 각 페르소나가 확률적으로 '의견' 생성
    - softmax 가중 합의 → 최종 판단
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

    def deliberate(
        self,
        situation: Situation,
        n_rounds: int = 100,
        noise_std: float = 0.3,
        verbose: bool = False,
    ) -> dict:
        combo_keys = situation.select_combo()
        combo = {k: PERSONAS[k] for k in combo_keys if k in PERSONAS}
        synergy = get_synergy(combo_keys)
        risk_penalty = np.exp(-0.1 * situation.risk)

        # ── n_rounds번 확률적 합의 시뮬레이션 ──
        # risk_penalty는 조합 선택(Excel 원칙)에만 사용
        # 합의는 raw 의견 기반으로 — 페르소나가 이미 상황을 반영한 의견을 냄
        round_scores: dict[str, list] = {k: [] for k in combo}
        consensus_history = []

        opinions_log: list[dict] = []
        for _ in range(n_rounds):
            opinions = {}
            for key, p in combo.items():
                opinions[key] = p.deliberate(situation, noise_std)

            opinions_log.append(dict(opinions))

            # softmax → 합의 가중치
            vals = np.array(list(opinions.values()))
            weights = np.exp(vals) / np.exp(vals).sum()

            # 가중 합의 점수 (synergy만 적용, risk_penalty는 combo rank용)
            raw_consensus = float(np.dot(weights, vals))
            consensus = raw_consensus * synergy

            for i, key in enumerate(combo):
                round_scores[key].append(opinions[key] * weights[i])
            consensus_history.append(consensus)

        # ── 통계 집계 ──
        consensus_arr = np.array(consensus_history)
        persona_stats = {}
        total_mean = max(sum(np.mean(round_scores[k]) for k in combo), 1e-9)
        for key in combo:
            arr = np.array(round_scores[key])
            # 원시 의견 점수 복원 (가중치 역산)
            raw_opinions = [opinions_log[i][key] for i in range(n_rounds)]
            raw_arr = np.array(raw_opinions)
            persona_stats[key] = {
                "role":       combo[key].role,
                "base":       round(combo[key].base_score, 3),
                "opinion_mean": round(float(raw_arr.mean()), 4),
                "opinion_std":  round(float(raw_arr.std()),  4),
                "contrib":    round(float(arr.mean()) / total_mean, 3),
                "agree_rate": round(float((raw_arr > 0.5).mean()), 3),
            }

        # ── 합의 확률 (다수 페르소나 의견 > 0.5 비율) ──
        round_agree = []
        for ops in opinions_log:
            votes_yes = sum(1 for v in ops.values() if v > 0.5)
            round_agree.append(votes_yes / len(ops))
        agree_prob = float(np.mean([v >= 0.5 for v in round_agree]))

        # ── 추천 액션 ──
        final_score = float(consensus_arr.mean())
        action = _recommend(final_score, agree_prob, situation)

        result = {
            "situation":      {"risk": situation.risk, "speed": situation.speed,
                               "uncertainty": situation.uncertainty,
                               "complexity": situation.complexity,
                               "state_score": round(situation.state_score, 3)},
            "combo":          combo_keys,
            "synergy_bonus":  round(synergy, 4),
            "risk_penalty":   round(risk_penalty, 4),
            "n_rounds":       n_rounds,
            "final_score":    round(final_score, 4),
            "score_std":      round(float(consensus_arr.std()), 4),
            "agree_prob":     round(agree_prob, 3),
            "confidence":     _confidence(agree_prob, consensus_arr.std()),
            "action":         action,
            "personas":       persona_stats,
        }

        if verbose:
            _print_result(result)

        return result


def _recommend(score: float, agree_prob: float, sit: Situation) -> str:
    if agree_prob >= 0.8 and score >= 0.8:
        return "즉시 실행 (HIGH CONFIDENCE)"
    elif agree_prob >= 0.6:
        if sit.risk >= 7:
            return "단계적 실행 — 위험 모니터링 병행"
        return "실행 권장 (MODERATE)"
    elif agree_prob >= 0.4:
        return "조건부 실행 — 추가 정보 수집 후"
    else:
        return "보류 또는 전략 재검토"


def _confidence(agree_prob: float, std: float) -> str:
    if agree_prob >= 0.75 and std < 0.12:
        return "HIGH"
    elif agree_prob >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def _print_result(r: dict):
    sit = r["situation"]
    print("=" * 56)
    print("SIGOL 확률적 합의 결과")
    print("=" * 56)
    print(f"상황: Risk={sit['risk']} Speed={sit['speed']} "
          f"Uncertainty={sit['uncertainty']} Complexity={sit['complexity']}")
    print(f"상태점수: {sit['state_score']}")
    print()
    print(f"활성 페르소나: {' + '.join(r['combo'])}")
    print(f"시너지 보너스: ×{r['synergy_bonus']}")
    print(f"위험 패널티:   ×{r['risk_penalty']}")
    print()
    print(f"합의 점수:  {r['final_score']} ± {r['score_std']}")
    print(f"합의 확률:  {r['agree_prob']*100:.1f}%")
    print(f"신뢰도:     {r['confidence']}")
    print(f"추천 액션:  {r['action']}")
    print()
    print("── 페르소나별 의견 ──")
    for name, s in r["personas"].items():
        opinion_bar = "█" * int(s["opinion_mean"] * 20)
        agree_pct   = f"{s['agree_rate']*100:.0f}%"
        print(f"  {name:12s} [{s['role']:8s}]  "
              f"의견={s['opinion_mean']:.3f}±{s['opinion_std']:.3f}  "
              f"찬성={agree_pct:>4s}  {opinion_bar}")
    print("=" * 56)


# ── CLI 실행 ───────────────────────────────────────────────────
if __name__ == "__main__":
    engine = SigolPersonaEngine(seed=42)

    print("\n【 시나리오 A: 고위험 복잡 상황 】")
    engine.deliberate(Situation(risk=8, speed=7, uncertainty=6, complexity=9),
                      n_rounds=200, verbose=True)

    print("\n【 시나리오 B: 저위험 안정 상황 】")
    engine.deliberate(Situation(risk=3, speed=4, uncertainty=3, complexity=4),
                      n_rounds=200, verbose=True)

    print("\n【 시나리오 C: 고속 실행 요구 】")
    engine.deliberate(Situation(risk=5, speed=9, uncertainty=5, complexity=6),
                      n_rounds=200, verbose=True)
