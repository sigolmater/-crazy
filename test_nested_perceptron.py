"""
중첩 퍼셉트론 AI 테스트 스위트

재현성, 안정성, 수학적 정확성 검증
"""

import numpy as np
import unittest
from nested_perceptron import perceptron_predict, nested_ai, analyze_depth_stability


class TestPerceptronPredict(unittest.TestCase):
    """단일 퍼셉트론 예측 테스트"""

    def test_positive_prediction(self):
        """양성 예측 (w·x > 0)"""
        inputs = np.array([1.0, 1.0])
        weights = np.array([1.0, 1.0])
        result = perceptron_predict(inputs, weights)
        self.assertTrue(result)

    def test_negative_prediction(self):
        """음성 예측 (w·x < 0)"""
        inputs = np.array([1.0, 1.0])
        weights = np.array([-1.0, -1.0])
        result = perceptron_predict(inputs, weights)
        self.assertFalse(result)

    def test_zero_boundary(self):
        """경계선 케이스 (w·x = 0)"""
        inputs = np.array([1.0, -1.0])
        weights = np.array([1.0, 1.0])
        result = perceptron_predict(inputs, weights)
        self.assertFalse(result)  # 0은 False

    def test_3d_vector(self):
        """3차원 벡터 테스트"""
        inputs = np.array([1.0, 0.0, 1.0])
        weights = np.array([0.5, -0.5, 1.0])
        result = perceptron_predict(inputs, weights)
        # w·x = 1.0*0.5 + 0.0*(-0.5) + 1.0*1.0 = 1.5 > 0
        self.assertTrue(result)


class TestNestedAI(unittest.TestCase):
    """중첩 AI 재귀 테스트"""

    def test_depth_zero(self):
        """깊이 0: 단순 퍼셉트론과 동일"""
        np.random.seed(42)
        inputs = np.array([1.0, 0.0, 1.0])
        weights = np.array([0.5, -0.5, 1.0])

        result_nested = nested_ai(0, weights, inputs)
        result_direct = perceptron_predict(inputs, weights)

        self.assertEqual(result_nested, result_direct)

    def test_reproducibility(self):
        """재현성: 동일 seed → 동일 결과"""
        inputs = np.array([1.0, 0.0, 1.0])
        weights = np.array([0.5, -0.5, 1.0])

        # 첫 번째 실행
        np.random.seed(42)
        result1 = nested_ai(5, weights, inputs)

        # 두 번째 실행 (동일 seed)
        np.random.seed(42)
        result2 = nested_ai(5, weights, inputs)

        self.assertEqual(result1, result2)

    def test_expected_result(self):
        """과학 검증: seed=42, depth=5 → True"""
        np.random.seed(42)
        inputs = np.array([1.0, 0.0, 1.0])
        weights = np.array([0.5, -0.5, 1.0])
        result = nested_ai(5, weights, inputs)

        self.assertTrue(result)  # 100% True 재현

    def test_different_seeds(self):
        """다른 seed → 다른 결과 가능"""
        inputs = np.array([1.0, 0.0, 1.0])
        weights = np.array([0.5, -0.5, 1.0])

        results = []
        for seed in range(10):
            np.random.seed(seed)
            result = nested_ai(5, weights, inputs)
            results.append(result)

        # 모든 결과가 동일하지 않을 수 있음 (섭동 효과)
        # 단, 이 테스트는 단순히 실행되는지 확인
        self.assertEqual(len(results), 10)

    def test_custom_perturbation(self):
        """커스텀 섭동 표준편차"""
        np.random.seed(42)
        inputs = np.array([1.0, 0.0, 1.0])
        weights = np.array([0.5, -0.5, 1.0])

        # 작은 섭동
        result_small = nested_ai(5, weights, inputs, perturbation_std=0.01)

        # 큰 섭동
        np.random.seed(42)
        result_large = nested_ai(5, weights, inputs, perturbation_std=1.0)

        # 둘 다 bool이어야 함
        self.assertIsInstance(result_small, (bool, np.bool_))
        self.assertIsInstance(result_large, (bool, np.bool_))


class TestStabilityAnalysis(unittest.TestCase):
    """안정성 분석 테스트"""

    def test_stability_analysis_runs(self):
        """안정성 분석 함수 실행 확인"""
        results = analyze_depth_stability(max_depth=3, num_trials=10)

        # 모든 깊이에 대한 결과가 있어야 함
        self.assertEqual(len(results), 4)  # 0, 1, 2, 3

        # 모든 비율이 0~1 사이
        for depth, ratio in results.items():
            self.assertGreaterEqual(ratio, 0.0)
            self.assertLessEqual(ratio, 1.0)

    def test_depth_zero_deterministic(self):
        """깊이 0은 완전히 결정론적"""
        results = analyze_depth_stability(max_depth=0, num_trials=100)

        # 깊이 0은 섭동이 없으므로 항상 동일한 결과
        self.assertIn(results[0], [0.0, 1.0])


class TestMathematicalProperties(unittest.TestCase):
    """수학적 특성 검증"""

    def test_linearity(self):
        """선형성: 스케일링 불변성"""
        inputs = np.array([1.0, 0.0, 1.0])
        weights = np.array([0.5, -0.5, 1.0])

        result1 = perceptron_predict(inputs, weights)
        result2 = perceptron_predict(2 * inputs, weights)
        result3 = perceptron_predict(inputs, 2 * weights)

        # 모두 동일한 부호
        self.assertEqual(result1, result2)
        self.assertEqual(result1, result3)

    def test_weight_perturbation_bounds(self):
        """가중치 섭동이 합리적 범위 내"""
        np.random.seed(42)
        parent_weights = np.array([0.5, -0.5, 1.0])
        perturbation_std = 0.1

        # 여러 번 섭동 생성
        perturbed_weights_list = []
        for _ in range(100):
            child_weights = parent_weights + np.random.normal(0, perturbation_std, len(parent_weights))
            perturbed_weights_list.append(child_weights)

        # 평균은 parent_weights에 가까워야 함
        mean_weights = np.mean(perturbed_weights_list, axis=0)
        np.testing.assert_array_almost_equal(mean_weights, parent_weights, decimal=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
