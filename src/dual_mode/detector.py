"""
SecureRide DualModeDetector implementing Algorithm 1 from paper Section 5.
"""

from typing import List, Optional, Tuple, Any, Callable
import numpy as np


def make_fast_predictor(model: Any, L: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap model with a compiled tf.function for low-latency streaming evaluation (single window step).
    Falls back gracefully to __call__ or predict for mock/custom models.
    """
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model) or hasattr(model, "call"):
            @tf.function(input_signature=[tf.TensorSpec(shape=[1, L, 3], dtype=tf.float32)])
            def _compiled_call(x_tensor):
                return model(x_tensor, training=False)

            # Warm up graph
            _dummy = np.zeros((1, L, 3), dtype=np.float32)
            _ = _compiled_call(_dummy)
            return lambda x: _compiled_call(x).numpy()[0]
    except Exception:
        pass

    if hasattr(model, "__call__"):
        def _call_wrapper(x):
            res = model(x, training=False)
            return res.numpy()[0] if hasattr(res, "numpy") else res[0]
        return _call_wrapper

    return lambda x: model.predict(x, verbose=0)[0]


class DualModeDetector:
    """
    Streaming Dual-Mode Detector for SecureRide speeding detection.
    
    Implements paper Algorithm 1 (lines 2-14):
    - Starts in mode = "quick" with short window L_short.
    - If Quick detects illegal speeding (y=True), switches to "accurate" mode.
    - Accurate mode uses longer window L_long (combining previous buffer samples).
    - If Accurate confirms normal behavior (y=False), switches back to "quick".
    """

    def __init__(
        self,
        quick: Any,
        accurate: Any,
        mu: np.ndarray,
        std: np.ndarray,
        L_short: int = 5,
        L_long: int = 20,
    ):
        self.quick = quick
        self.accurate = accurate
        self.mu = np.array(mu, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.L_short = int(L_short)
        self.L_long = int(L_long)

        # Build fast compiled execution graphs for each model window size
        self.quick_predict_fn = make_fast_predictor(self.quick, self.L_short)
        self.accurate_predict_fn = make_fast_predictor(self.accurate, self.L_long)

        self.mode: str = "quick"
        self.buffer: List[np.ndarray] = []
        self.reset()

    def reset(self) -> None:
        """Reset state machine to initial quick mode and empty buffer (line 2)."""
        self.mode = "quick"
        self.buffer = []

    def _predict(self, window: np.ndarray, mode: str) -> bool:
        """Standardize window and predict illegal speeding binary label (0 or 1)."""
        x = (window - self.mu) / self.std  # standardization identical to training
        inp = x[np.newaxis, :, :].astype(np.float32)
        predict_fn = self.quick_predict_fn if mode == "quick" else self.accurate_predict_fn
        p = predict_fn(inp)
        return bool(np.argmax(p) == 1)

    def step(self, x_t: np.ndarray) -> Tuple[Optional[bool], str]:
        """
        Ingest a single timestep 3-feature reading x_t at 1 Hz.
        Returns:
            (y, mode): y is True/False or None during warmup; mode is current mode string.
        """
        self.buffer.append(np.array(x_t, dtype=np.float32))
        if len(self.buffer) > self.L_long:
            self.buffer.pop(0)  # keep rolling buffer up to L_long samples

        L = self.L_short if self.mode == "quick" else self.L_long
        if len(self.buffer) < L:
            return None, self.mode  # warmup: not enough history

        window = np.array(self.buffer[-L:])
        y = self._predict(window, self.mode)  # Algorithm 1 line 6

        # Lines 7-10: Mode switching
        if y and self.mode == "quick":
            self.mode = "accurate"
        elif not y and self.mode == "accurate":
            self.mode = "quick"

        # Return prediction y and the new mode state (line 11)
        return y, self.mode


def run_fixed(
    model: Any,
    features: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    L: int,
) -> List[Optional[bool]]:
    """
    Run fixed-mode baseline (Quick-only or Accurate-only) with no mode switching.
    """
    mu_arr = np.array(mu, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    predict_fn = make_fast_predictor(model, L)
    ys: List[Optional[bool]] = []
    
    for t in range(len(features)):
        if t + 1 < L:
            ys.append(None)
            continue
        window = features[t - L + 1 : t + 1]
        x = (window - mu_arr) / std_arr
        inp = x[np.newaxis, :, :].astype(np.float32)
        p = predict_fn(inp)
        ys.append(bool(np.argmax(p) == 1))
    return ys


def test_dual_mode_detector():
    """Unit tests for DualModeDetector."""
    class DummyModel:
        def __init__(self, trigger_val=0.15):
            self.trigger_val = trigger_val
            
        def __call__(self, x, training=False):
            val = float(np.mean(x))
            if val > self.trigger_val:
                res = np.array([[0.1, 0.9]], dtype=np.float32)
            else:
                res = np.array([[0.9, 0.1]], dtype=np.float32)
            class Out:
                def numpy(self_inner):
                    return res
            return Out()

    mu = np.zeros(3, dtype=np.float32)
    std = np.ones(3, dtype=np.float32)
    q_mod = DummyModel(0.15)
    a_mod = DummyModel(0.15)
    det = DualModeDetector(q_mod, a_mod, mu, std, L_short=5, L_long=20)

    # 1. Warm-up test
    for t in range(4):
        y, m = det.step(np.zeros(3))
        assert y is None, f"Expected None during warm-up at t={t}, got {y}"
        assert m == "quick"

    # 2. t=4 is 5th sample, should predict
    y, m = det.step(np.zeros(3))
    assert y is False
    assert m == "quick"

    # Fill up to 25 samples with 0 (normal)
    for _ in range(25):
        y, m = det.step(np.zeros(3))
        assert y is False
        assert m == "quick"
    assert len(det.buffer) == 20, f"Buffer must maintain max L_long=20, got {len(det.buffer)}"

    # Now simulate speeding onset: feed 1.0 (mean > 0.15)
    y, m = det.step(np.ones(3))
    assert y is True, "Quick model should trigger positive"
    assert m == "accurate", "Mode should switch to accurate immediately"
    assert len(det.buffer) == 20, f"Buffer must already have 20 samples for accurate model! Got {len(det.buffer)}"

    # Determinism test
    det.reset()
    seq = [np.random.randn(3) for _ in range(50)]
    out1 = [det.step(s) for s in seq]
    det.reset()
    out2 = [det.step(s) for s in seq]
    assert out1 == out2, "DualModeDetector must be completely deterministic upon reset()"
    print("All DualModeDetector unit tests PASSED successfully!")


if __name__ == "__main__":
    test_dual_mode_detector()
