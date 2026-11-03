"""Rewritten: signed step heights minimizing max|autoconv| / (sum s)^2.

Strategy:
- Phase 1: gradient-based optimization on a smooth surrogate (log-sum-exp of |ac|^2)
  with multiple diverse restarts and temperature annealing.
- Phase 2: fast discrete local search using O(N) incremental autoconvolution updates.
  Evaluate all N single-flips per pass; take the best improving flip.
- Phase 3: simulated annealing to escape local minima.
- Phase 4: final continuous polish with tighter temperature.
- Explicit deadline guard with 10 s safety margin.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve


def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return float(max_ac) / sq_int


def _soft_max_ac(s, K=256):
    """Differentiable surrogate for max|autoconv| via top-K log-sum-exp on ac^2."""
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    lse = max_val + np.log(np.sum(np.exp(top - max_val)))
    return np.sqrt(lse)


def _soft_objective(x, K=256):
    """Soft objective: log-sum-exp surrogate for max|ac| divided by (sum s)^2."""
    s = np.tanh(x)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    return _soft_max_ac(s, K) / sq_int


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 s safety margin before the 150 s kill

    def _now():
        return time.time()

    best_score = np.inf
    best_s = None
    np.random.seed(42)

    # ---------------- Phase 1: gradient-based optimization ----------------
    n_restarts = 16
    for it in range(n_restarts):
        if _now() > deadline - 40:
            break

        if it < 4:
            x0 = np.random.randn(N) * 0.1
        elif it < 8:
            t = np.arange(N)
            freq = np.random.uniform(0.02, 0.5)
            x0 = 0.5 * np.sin(2 * np.pi * t * freq) + np.random.randn(N) * 0.05
        elif it < 12:
            x0 = np.random.randn(N) * 0.25
        else:
            x0 = np.random.randn(N) * 0.15

        temp = 2.0 * (0.55 ** (it / 10.0))
        try:
            result = minimize(
                _soft_objective,
                x0,
                method='L-BFGS-B',
                args=(256,),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            s_cont = np.tanh(result.x)
            s_bin = np.sign(s_cont)
            s_bin[s_bin == 0] = 1.0

            sc_b = _score_sequence(s_bin)
            if sc_b < best_score:
                best_score = sc_b
                best_s = s_bin.copy()
            sc_c = _score_sequence(s_cont)
            if sc_c < best_score:
                best_score = sc_c
                best_s = s_cont.copy()
        except Exception:
            continue

    # ---------------- Phase 2: discrete local search ----------------
    if best_s is not None:
        cur = best_s.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        cur_score = best_score

        def _flip_eval(idx):
            """Return (new_ac, new_sum, new_score) after flipping cur[idx]."""
            val = cur[idx]
            new_ac = cur_ac.copy()
            start = idx
            end = idx + N
            new_ac[start:end] += -4.0 * val * cur
            new_ac[idx] += 4.0 * val * val
            new_sum = cur_sum - 2.0 * val
            nsq = new_sum ** 2
            if nsq < 1e-10:
                return new_ac, new_sum, np.inf
            return new_ac, new_sum, float(np.max(np.abs(new_ac)) / nsq)

        # Greedy local search: repeatedly take the best single flip.
        max_passes = 400
        for _ in range(max_passes):
            if _now() > deadline - 25:
                break

            best_j = -1
            best_js = cur_score
            best_new_ac = None
            best_new_sum = cur_sum

            for j in range(N):
                new_ac, new_sum, new_score = _flip_eval(j)
                if new_score < best_js:
                    best_js = new_score
                    best_j = j
                    best_new_ac = new_ac
                    best_new_sum = new_sum

            if best_j == -1 or best_js >= cur_score - 1e-12:
                break

            cur[best_j] = -cur[best_j]
            cur_ac = best_new_ac
            cur_sum = best_new_sum
            cur_score = best_js

            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

        # Simulated annealing to escape local minima.
        T_start = 0.03
        sa_steps = 3000
        for it in range(sa_steps):
            if _now() > deadline - 25:
                break
            idx = int(np.random.randint(N))
            new_ac, new_sum, new_score = _flip_eval(idx)
            if new_score == np.inf:
                continue
            d = new_score - cur_score
            T = max(T_start * (0.995 ** it), 1e-7)
            if d < 0 or np.random.rand() < np.exp(-d / T):
                cur[idx] = -cur[idx]
                cur_ac, cur_sum, cur_score = new_ac, new_sum, new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_s = cur.copy()

    # ---------------- Phase 3: final high-precision polish ----------------
    if best_s is not None and _now() < deadline - 10:
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        for temp, iters in ((0.3, 150), (0.1, 150)):
            if _now() > deadline - 10:
                break
            try:
                result = minimize(
                    _soft_objective,
                    x_start,
                    method='L-BFGS-B',
                    args=(256,),
                    options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-9},
                )
                s_final = np.tanh(result.x)
                x_start = result.x
                sf = _score_sequence(s_final)
                if sf < best_score:
                    best_score = sf
                    best_s = s_final.copy()
                sf_b = _score_sequence(np.sign(s_final) * 1.0)
                if sf_b < best_score:
                    best_score = sf_b
                    best_s = np.sign(s_final).copy()
                    best_s[best_s == 0] = 1.0
            except Exception:
                pass

    # ---------------- Phase 4: one more local search round on best ----------------
    if best_s is not None and _now() < deadline - 8:
        cur = best_s.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        cur_score = best_score

        def _flip_eval2(idx):
            val = cur[idx]
            new_ac = cur_ac.copy()
            start = idx
            end = idx + N
            new_ac[start:end] += -4.0 * val * cur
            new_ac[idx] += 4.0 * val * val
            new_sum = cur_sum - 2.0 * val
            nsq = new_sum ** 2
            if nsq < 1e-10:
                return new_ac, new_sum, np.inf
            return new_ac, new_sum, float(np.max(np.abs(new_ac)) / nsq)

        max_passes2 = 200
        for _ in range(max_passes2):
            if _now() > deadline - 8:
                break
            best_j = -1
            best_js = cur_score
            best_new_ac = None
            best_new_sum = cur_sum
            for j in range(N):
                new_ac, new_sum, new_score = _flip_eval2(j)
                if new_score < best_js:
                    best_js = new_score
                    best_j = j
                    best_new_ac = new_ac
                    best_new_sum = new_sum
            if best_j == -1 or best_js >= cur_score - 1e-12:
                break
            cur[best_j] = -cur[best_j]
            cur_ac = best_new_ac
            cur_sum = best_new_sum
            cur_score = best_js
            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

    if best_s is None:
        return np.ones(N)
    return best_s
# EVOLVE-BLOCK-END
