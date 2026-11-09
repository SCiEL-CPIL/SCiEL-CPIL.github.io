import numpy as np
import time
from scipy.optimize import minimize


def _compute_P_fast(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] using O(n^2 log n) searchsorted.
    P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        # s0+s0 < 2*s0 - s0 => 2s0 < s0 => s0 < 0, but s>=0 so P=0
        return 0.0

    # Pairwise sums S_ij = s_i + s_j, shape (n, n)
    S = s[:, None] + s[None, :]
    # Weighted pairwise matrix W_ij = w_i * w_j
    W = w[:, None] * w[None, :]

    # For each (k, l), we need sum_{i,j: S_ij < R_kl} W_ij
    # where R_kl = 2*s_l - s_k
    # We can flatten S and W, sort by S, and use searchsorted for each R_kl

    S_flat = S.flatten()
    W_flat = W.flatten()

    # Sort S_flat and compute cumulative sum of W_flat in that order
    order = np.argsort(S_flat)
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    W_cumsum = np.cumsum(W_sorted)

    # R_kl = 2*s_l - s_k, shape (n, n)
    R = 2 * s[None, :] - s[:, None]

    # For each R value, find how many S values are strictly less than R
    # searchsorted with side='left' gives index of first element >= R
    # So the number of elements < R is that index
    indices = np.searchsorted(S_sorted, R.flatten(), side='left')

    # W_cumsum[idx-1] gives sum of W for elements with index < idx
    # Handle idx=0 case
    W_sum_for_R = np.zeros_like(indices, dtype=np.float64)
    mask_nonzero = indices > 0
    W_sum_for_R[mask_nonzero] = W_cumsum[indices[mask_nonzero] - 1]

    W_sum_for_R = W_sum_for_R.reshape(n, n)

    # P = sum_{k,l} w_k * w_l * W_sum_for_R[k,l]
    W_kl = w[:, None] * w[None, :]
    P = float(np.sum(W_kl * W_sum_for_R))
    return P


def _neg_P(params, n):
    """Negative P for minimization. params = [s_0,...,s_{n-1}, log_w_0,...,log_w_{n-1}]"""
    s = np.exp(params[:n])
    lw = params[n:]
    w = np.exp(lw)
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        return 1e10
    w = w / w_sum
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
        return 1e10
    return -_compute_P_fast(s, w)


def _try_optimize_n(n, start_time, deadline, maxiter=3000, n_restarts=5):
    """Optimize over continuous support and weights for given n using Nelder-Mead."""
    if time.time() > deadline - 30:
        return None, -np.inf

    best_P_val = -np.inf
    best_params = None

    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break

        # Different initializations
        if restart == 0:
            # Evenly spaced support
            s0 = np.linspace(0.1, 2.0, n)
            w0 = np.ones(n) / n
        elif restart == 1:
            # Geometric spacing
            s0 = np.logspace(-1, np.log10(5.0), n)
            w0 = np.ones(n) / n
        elif restart == 2:
            # Concentrated weights
            s0 = np.linspace(0.05, 3.0, n)
            w0 = np.ones(n) / n
            w0[0] = 0.5
            w0[n-1] = 0.3
            w0 = w0 / w0.sum()
        elif restart == 3:
            # Random
            s0 = np.sort(np.random.uniform(0.01, 10.0, n))
            w0 = np.random.dirichlet(np.ones(n))
        else:
            # Wide spread
            s0 = np.linspace(0.01, 20.0, n)
            w0 = np.ones(n) / n

        lw0 = np.log(np.maximum(w0, 1e-15))
        params_init = np.concatenate([np.log(np.maximum(s0, 0.01)), lw0])

        try:
            result = minimize(
                _neg_P, params_init, args=(n,),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            P_val = -result.fun
            if P_val > best_P_val:
                best_P_val = P_val
                best_params = result.x.copy()
        except Exception:
            pass

    if best_params is None or best_P_val <= -np.inf:
        return None, -np.inf

    s_opt = np.exp(best_params[:n])
    lw_opt = best_params[n:]
    w_opt = np.exp(lw_opt)
    w_opt = w_opt / w_opt.sum()

    return (s_opt, w_opt), best_P_val


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    # 1. Seed solution
    n_seed = 32
    support_seed = np.arange(n_seed, dtype=np.float64)
    weights_seed = np.ones(n_seed, dtype=np.float64) / n_seed
    best_P = _compute_P_fast(support_seed, weights_seed)
    best_support = support_seed.copy()
    best_weights = weights_seed.copy()

    # 2. Optimized Binary
    p_opt = (1 + np.sqrt(33)) / 16
    support_bin = np.array([0.0, 1.0])
    weights_bin = np.array([1 - p_opt, p_opt])
    P_bin = _compute_P_fast(support_bin, weights_bin)
    if P_bin > best_P:
        best_P = P_bin
        best_support = support_bin.copy()
        best_weights = weights_bin.copy()

    # 3. Optimize continuous support and weights for various n
    for n in [2, 3, 4, 5, 6, 8, 10, 12]:
        if time.time() > deadline - 60:
            break

        res, P_val = _try_optimize_n(n, start_time, deadline)
        if res is not None and P_val > best_P:
            best_P = P_val
            best_support = res[0].copy()
            best_weights = res[1].copy()

    # 4. Local refinement of best solution if it's small-n
    if len(best_support) <= 12 and time.time() < deadline - 20:
        n = len(best_support)
        s_cur = best_support.copy()
        w_cur = best_weights.copy()
        s_cur = np.maximum(s_cur, 0.01)
        w_cur = np.maximum(w_cur, 1e-15)
        w_cur = w_cur / w_cur.sum()

        params_cur = np.concatenate([np.log(s_cur), np.log(w_cur)])

        # Multiple Nelder-Mead restarts with small perturbations
        for trial in range(30):
            if time.time() > deadline - 10:
                break
            if trial == 0:
                p_init = params_cur.copy()
            else:
                p_init = params_cur + np.random.randn(len(params_cur)) * 0.05

            try:
                result = minimize(
                    _neg_P, p_init, args=(n,),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17}
                )
                P_trial = -result.fun
                if P_trial > best_P:
                    s_trial = np.exp(result.x[:n])
                    lw_trial = result.x[n:]
                    w_trial = np.exp(lw_trial)
                    w_trial = w_trial / w_trial.sum()
                    best_P = P_trial
                    best_support = s_trial
                    best_weights = w_trial
                    params_cur = result.x.copy()
            except Exception:
                pass

    # Final sanity check
    if (np.any(best_weights < -1e-9) or np.any(best_support < -1e-9) or
            np.sum(best_weights) <= 1e-9 or
            np.any(np.isnan(best_weights)) or np.any(np.isnan(best_support)) or
            np.any(np.isinf(best_weights)) or np.any(np.isinf(best_support))):
        return support_seed, weights_seed

    return best_support, best_weights
