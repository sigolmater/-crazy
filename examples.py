"""
중첩 퍼셉트론 AI 사용 예제

다양한 시나리오와 활용 방법을 보여줍니다.
"""

import numpy as np
from nested_perceptron import perceptron_predict, nested_ai, analyze_depth_stability


def example_1_basic_usage():
    """예제 1: 기본 사용법"""
    print("=" * 60)
    print("예제 1: 기본 사용법")
    print("=" * 60)

    np.random.seed(42)
    inputs = np.array([1.0, 0.0, 1.0])
    weights = np.array([0.5, -0.5, 1.0])

    print(f"입력 벡터: {inputs}")
    print(f"초기 가중치: {weights}")
    print(f"내적: {np.dot(inputs, weights)}")
    print()

    for depth in [0, 1, 3, 5, 10]:
        np.random.seed(42)
        result = nested_ai(depth, weights, inputs)
        print(f"깊이 {depth:2d}: {result}")

    print()


def example_2_reproducibility():
    """예제 2: 재현성 검증"""
    print("=" * 60)
    print("예제 2: 재현성 검증 (동일 seed → 동일 결과)")
    print("=" * 60)

    inputs = np.array([1.0, 0.0, 1.0])
    weights = np.array([0.5, -0.5, 1.0])

    print("10회 실행 (모두 seed=42)")
    results = []
    for i in range(10):
        np.random.seed(42)
        result = nested_ai(5, weights, inputs)
        results.append(result)
        print(f"실행 {i+1:2d}: {result}")

    all_same = all(r == results[0] for r in results)
    print(f"\n모든 결과 동일: {all_same} ✓")
    print()


def example_3_perturbation_effect():
    """예제 3: 섭동 강도에 따른 영향"""
    print("=" * 60)
    print("예제 3: 섭동 강도 (σ)에 따른 영향")
    print("=" * 60)

    inputs = np.array([1.0, 0.0, 1.0])
    weights = np.array([0.5, -0.5, 1.0])

    perturbation_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    print(f"깊이: 5, 시행: 100회\n")
    print(f"{'σ':>6s} | {'True 비율':>10s}")
    print("-" * 20)

    for std in perturbation_values:
        true_count = 0
        for trial in range(100):
            np.random.seed(trial)
            result = nested_ai(5, weights, inputs, perturbation_std=std)
            if result:
                true_count += 1

        ratio = true_count / 100
        print(f"{std:6.2f} | {ratio:9.1%}")

    print()


def example_4_stability_visualization():
    """예제 4: 깊이별 안정성 분석"""
    print("=" * 60)
    print("예제 4: 깊이별 안정성 분석")
    print("=" * 60)

    stability = analyze_depth_stability(max_depth=15, num_trials=100)

    print(f"{'깊이':>4s} | {'True 비율':>10s} | {'시각화':s}")
    print("-" * 50)

    for depth, ratio in stability.items():
        bar_length = int(ratio * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{depth:4d} | {ratio:9.1%} | {bar}")

    print()


def example_5_different_inputs():
    """예제 5: 다양한 입력 벡터"""
    print("=" * 60)
    print("예제 5: 다양한 입력 벡터 테스트")
    print("=" * 60)

    weights = np.array([0.5, -0.5, 1.0])

    test_inputs = [
        ("양수 벡터", np.array([1.0, 1.0, 1.0])),
        ("음수 벡터", np.array([-1.0, -1.0, -1.0])),
        ("혼합 벡터", np.array([1.0, -1.0, 1.0])),
        ("영 벡터", np.array([0.0, 0.0, 0.0])),
        ("단위 벡터", np.array([1.0, 0.0, 0.0])),
    ]

    print(f"가중치: {weights}\n")

    for name, inputs in test_inputs:
        np.random.seed(42)
        result = nested_ai(5, weights, inputs)
        dot_product = np.dot(inputs, weights)
        print(f"{name:12s}: 내적={dot_product:6.2f}, 예측={result}")

    print()


def example_6_weight_evolution():
    """예제 6: 가중치 진화 추적"""
    print("=" * 60)
    print("예제 6: 가중치 진화 추적 (5단계)")
    print("=" * 60)

    np.random.seed(42)
    inputs = np.array([1.0, 0.0, 1.0])
    parent_weights = np.array([0.5, -0.5, 1.0])

    print(f"초기 가중치: {parent_weights}\n")

    current_weights = parent_weights.copy()
    for depth in range(5, 0, -1):
        current_weights = current_weights + np.random.normal(0, 0.1, len(current_weights))
        prediction = perceptron_predict(inputs, current_weights)
        dot_product = np.dot(inputs, current_weights)

        print(f"단계 {6-depth}: w={current_weights}, w·x={dot_product:.3f}, 예측={prediction}")

    print()


def main():
    """모든 예제 실행"""
    print("\n" + "🔷" * 30)
    print("중첩 퍼셉트론 AI - 사용 예제")
    print("🔷" * 30 + "\n")

    example_1_basic_usage()
    example_2_reproducibility()
    example_3_perturbation_effect()
    example_4_stability_visualization()
    example_5_different_inputs()
    example_6_weight_evolution()

    print("=" * 60)
    print("모든 예제 완료! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
