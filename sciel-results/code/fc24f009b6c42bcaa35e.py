"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _autoconv_max_abs(x):
    """Compute maximum absolute value of linear autoconvolution."""
    # Linear convolution of x with itself (full mode)
    conv = np.convolve(x, x, mode='full')
    return np.max(np.abs(conv))


def _integral_squared(x):
    """Compute the square of the integral (sum for discrete)."""
    s = np.sum(x)
    return s * s


def _objective(x):
    """Objective: max_abs_autoconv / integral^2. If integral is 0, return large value."""
    x = np.asarray(x)
    denom = _integral_squared(x)
    if abs(denom) < 1e-10:
        return 1e10
    return _autoconv_max_abs(x) / denom


def _sign_step_constraint(x):
    """Return x as is; sign constraints are handled by bounding in optimization."""
    return x


def construct():
    """Construct a signed step function minimizing the score."""
    start_time = time.time()
    deadline = start_time + 120.0  # Must finish before evaluator kills at 150s

    # Initial best: alternating signs to reduce autoconvolution peak
    x_best = np.zeros(N)
    x_best[::2] = 1.0
    x_best[1::2] = -1.0
    best_score = _objective(x_best)

    # Try a few different initial patterns and pick the best
    patterns = []
    # Pattern 1: alternating
    p1 = np.zeros(N)
    p1[::2] = 1.0
    p1[1::2] = -1.0
    patterns.append(p1)

    # Pattern 2: blocks of +1 and -1
    p2 = np.zeros(N)
    for i in range(N):
        p2[i] = 1.0 if (i // 10) % 2 == 0 else -1.0
    patterns.append(p2)

    # Pattern 3: random
    np.random.seed(42)
    p3 = np.random.choice([-1.0, 1.0], size=N)
    patterns.append(p3)

    # Pattern 4: low-frequency sign changes
    p4 = np.zeros(N)
    for i in range(N):
        p4[i] = np.sign(np.sin(2 * np.pi * i / 100))
    p4[p4 == 0] = 1.0
    patterns.append(p4)

    for p in patterns:
        s = _objective(p)
        if s < best_score:
            best_score = s
            x_best = p.copy()

    # Use differential evolution to search over continuous amplitudes,
    # then round to signs for the final answer
    def _continuous_objective(params):
        """Objective for continuous amplitudes, with penalty for large values."""
        x = np.tanh(params)  # Map to (-1, 1)
        s = _objective(x)
        # Add a penalty to encourage binary-like values
        penalty = np.sum((1 - np.abs(x)) ** 2) * 0.01
        return s + penalty

    # Run differential evolution on a subset of dimensions for speed
    # We'll optimize all N dimensions but with limited evaluations
    bounds = [(-2, 2)] * N

    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:  # Leave 30s buffer
        try:
            result = differential_evolution(
                _continuous_objective,
                bounds,
                maxiter=50,
                tol=0.01,
                seed=42,
                popsize=15,
                polish=False,
                workers=1
            )
            x_cont = np.tanh(result.x)
            # Convert to signed step heights: use sign
            x_sign = np.sign(x_cont)
            x_sign[x_sign == 0] = 1.0
            s_cont = _objective(x_sign)
            if s_cont < best_score:
                best_score = s_cont
                x_best = x_sign.copy()
        except Exception:
            pass

    # Local polish: try flipping individual signs to improve
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        improved = True
        while improved and (time.time() - start_time) < deadline - 15:
            improved = False
            for i in range(N):
                if time.time() - start_time > deadline - 15:
                    break
                # Try flipping sign at position i
                x_try = x_best.copy()
                x_try[i] *= -1
                s_try = _objective(x_try)
                if s_try < best_score:
                    best_score = s_try
                    x_best = x_try
                    improved = True

    return x_best
# EVOLVE-BLOCK-END
