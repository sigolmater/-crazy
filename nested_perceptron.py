"""
중첩 퍼셉트론 AI (Nested Perceptron AI)

재귀적 퍼셉트론 시스템: 각 깊이 단계마다 가우시안 노이즈로 가중치 섭동(perturbation)
- 수학: 선형 분류기 w·x > 0
- 물리: δE ~ σ perturbation (에너지 섭동)
- 진화: 돌연변이 기반 weight mutation
- 재현성: seed 고정으로 100% 재현 가능
"""

import numpy as np  # 벡터화 효율 (O(d) dot, d=3)


def perceptron_predict(inputs, weights):
    """
    단일 퍼셉트론 예측 함수

    수학: w·x > 0 (linear classifier, 물리: δE ~ σ perturbation)

    Args:
        inputs: 입력 벡터 (numpy array)
        weights: 가중치 벡터 (numpy array)

    Returns:
        bool: 예측 결과 (True/False)
    """
    return np.dot(inputs, weights) > 0


def nested_ai(depth, parent_weights, inputs, perturbation_std=0.1):
    """
    중첩 AI 재귀 함수

    재귀: tail opt for O(n) stack
    전략: mutation 진화 - 각 단계마다 N(0,σ) 노이즈 추가

    Args:
        depth: 재귀 깊이 (0이 되면 종료)
        parent_weights: 부모 가중치 벡터
        inputs: 입력 벡터
        perturbation_std: 섭동 표준편차 (default: 0.1)

    Returns:
        bool: 최종 예측 결과
    """
    if depth == 0:
        return perceptron_predict(inputs, parent_weights)

    # N(0,σ) 재현 seed=42
    child_weights = parent_weights + np.random.normal(0, perturbation_std, len(parent_weights))

    # 압축: single func, 무한 중첩 가능
    return nested_ai(depth - 1, child_weights, inputs, perturbation_std)


def run_reproducible_example():
    """
    재현 실행: seed 고정, 입력/부모 고정 (과학 검증: 100% True 재현)
    """
    np.random.seed(42)
    inputs = np.array([1.0, 0.0, 1.0])  # x = [x1,x2,x3]
    parent_weights = np.array([0.5, -0.5, 1.0])  # w0 = [w1,w2,w3]
    result = nested_ai(5, parent_weights, inputs)  # 5중 실행, 효율: O(15) ops

    print(f"5중첩 예측: {result}")  # 출력: True (재현 가능, 깊은 추론: perturbation 안정성 확인)
    return result


def analyze_depth_stability(max_depth=10, num_trials=100, perturbation_std=0.1):
    """
    깊이에 따른 예측 안정성 분석

    Args:
        max_depth: 최대 재귀 깊이
        num_trials: 각 깊이별 시행 횟수
        perturbation_std: 섭동 표준편차

    Returns:
        dict: 깊이별 True 비율
    """
    inputs = np.array([1.0, 0.0, 1.0])
    parent_weights = np.array([0.5, -0.5, 1.0])

    results = {}

    for depth in range(max_depth + 1):
        true_count = 0
        for trial in range(num_trials):
            np.random.seed(trial)  # 각 시행마다 다른 seed
            prediction = nested_ai(depth, parent_weights, inputs, perturbation_std)
            if prediction:
                true_count += 1

        results[depth] = true_count / num_trials

    return results


if __name__ == "__main__":
    # 재현 가능 실행
    print("=" * 50)
    print("재현 가능 실행 (seed=42)")
    print("=" * 50)
    run_reproducible_example()

    # 안정성 분석
    print("\n" + "=" * 50)
    print("깊이별 안정성 분석 (100회 시행)")
    print("=" * 50)
    stability = analyze_depth_stability()
    for depth, true_ratio in stability.items():
        print(f"깊이 {depth:2d}: True 비율 = {true_ratio:.2%}")
