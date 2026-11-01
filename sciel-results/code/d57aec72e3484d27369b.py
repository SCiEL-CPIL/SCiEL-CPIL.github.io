"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def compute_autocorr(h):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        R = np.zeros(N)
        R[0] = acf[0]
        R[1:N] = acf[1:N]
        return R

    def get_max_autocorr(R):
        return np.max(np.abs(R[1:N]))

    def hill_climb(h, R, current_val, max_passes=300, time_limit=None):
        """Greedy local search with incremental O(N) updates."""
        if time_limit is None:
            time_limit = deadline - 5.0
        h = h.copy()
        R = R.copy()
        k = np.arange(1, N)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > time_limit:
                break
            improved = False
            passes += 1
            order = np.random.permutation(N)
            for i in order:
                if time.time() > time_limit:
                    break
                # Delta for flipping h[i]
                idx1 = i + k
                valid1 = idx1 < N
                term1 = np.zeros(N-1)
                term1[valid1] = h[idx1[valid1]]
                idx2 = i - k
                valid2 = idx2 >= 0
                term2 = np.zeros(N-1)
                term2[valid2] = h[idx2[valid2]]
                delta = -2.0 * h[i] * (term1 + term2)
                R[1:N] += delta
                new_val = get_max_autocorr(R)
                if new_val < current_val:
                    h[i] *= -1.0
                    current_val = new_val
                    improved = True
                else:
                    R[1:N] -= delta
        return h, R, current_val

    def soft_max_sidelobe(ac, beta=50.0):
        """
        Smooth approximation of max |ac[1:N]| using log-sum-exp.
        """
        sidelobes = np.abs(ac[1:N])
        max_val = np.max(sidelobes)
        if max_val < 1e-10:
            return max_val
        shifted = beta * (sidelobes - max_val)
        log_sum = np.log(np.sum(np.exp(shifted)))
        return max_val + log_sum / beta

    def objective_and_gradient(x, beta=50.0):
        """
        Objective: Smooth approximation of max sidelobe.
        Gradient: Analytic.
        """
        h = x
        ac = compute_autocorr(h)
        sidelobes_abs = np.abs(ac[1:N])
        
        max_val = np.max(sidelobes_abs)
        if max_val < 1e-10:
            return max_val, np.zeros(N)
            
        shifted = beta * (sidelobes_abs - max_val)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        obj = max_val + np.log(sum_exp) / beta
        
        weights = beta * exp_shifted / sum_exp
        
        grad = np.zeros(N)
        
        for k in range(1, N):
            w_k = weights[k-1]
            if w_k < 1e-15:
                continue
            
            sign_k = np.sign(ac[k])
            c_k = sign_k * w_k
            
            if c_k == 0:
                continue
                
            grad[0:N-k] += c_k * h[k:N]
            grad[k:N] += c_k * h[0:N-k]
            
        return obj, grad

    def continuous_optimize(x0, beta, time_limit):
        """
        Run L-BFGS-B on the continuous relaxation, then round.
        """
        def obj_grad(x):
            return objective_and_gradient(x, beta)
        
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20})
        except Exception:
            return x0, None
        
        # Round to binary
        h_bin = np.sign(res.x)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
        
        return h_bin, res.fun

    def multi_block_anneal(h, R, current_val, time_limit):
        """
        Structural change: divide sequence into 4 quarters.
        For each quarter, run segment-restricted annealing, then hill-climb.
        """
        q_len = N // 4
        h_new = h.copy()
        R_new = R.copy()
        val_new = current_val
        for q in range(4):
            if time.time() > time_limit:
                break
            seg_start = q * q_len
            seg_len = q_len if q < 3 else N - seg_start
            # Simple segment annealing
            T = 3.0
            iters = 1500
            cooling = (0.05 / 3.0) ** (1.0 / max(iters, 1))
            k = np.arange(1, N)
            indices = np.arange(seg_start, seg_start + seg_len)
            for it in range(iters):
                if time.time() > time_limit:
                    break
                i = indices[np.random.randint(seg_len)]
                idx1 = i + k
                valid1 = idx1 < N
                term1 = np.zeros(N-1)
                term1[valid1] = h_new[idx1[valid1]]
                idx2 = i - k
                valid2 = idx2 >= 0
                term2 = np.zeros(N-1)
                term2[valid2] = h_new[idx2[valid2]]
                delta = -2.0 * h_new[i] * (term1 + term2)
                R_new[1:N] += delta
                new_val = get_max_autocorr(R_new)
                delta_val = new_val - val_new
                if delta_val < 0:
                    h_new[i] *= -1.0
                    val_new = new_val
                else:
                    if np.random.rand() < np.exp(-delta_val / T):
                        h_new[i] *= -1.0
                        val_new = new_val
                    else:
                        R_new[1:N] -= delta
                T *= cooling
                if T < 0.05:
                    T = 0.05
        # Final polish
        h_new, R_new, val_new = hill_climb(h_new, R_new, val_new, max_passes=200, time_limit=time_limit)
        return h_new, R_new, val_new

    rng = np.random.RandomState(123)
    best_h = None
    best_R = None
    best_val = float('inf')

    # Phase 1: Random restarts with hill climbing
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        h = rng.choice([1, -1], N).astype(np.float64)
        R = compute_autocorr(h)
        val = get_max_autocorr(R)
        h, R, val = hill_climb(h, R, val, max_passes=100, time_limit=deadline - 20.0)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            best_R = R.copy()

    if best_h is None:
        best_h = rng.choice([1, -1], N).astype(np.float64)
        best_R = compute_autocorr(best_h)
        best_val = get_max_autocorr(best_R)

    # Phase 2: Continuous relaxation with L-BFGS-B
    if time.time() < deadline - 30.0:
        betas = [20.0, 50.0, 100.0]
        for beta in betas:
            if time.time() > deadline - 25.0:
                break
            x0 = best_h.copy()
            h_cand, _ = continuous_optimize(x0, beta, deadline - 20.0)
            R_cand = compute_autocorr(h_cand)
            val_cand = get_max_autocorr(R_cand)
            # Polish with hill climbing
            h_cand, R_cand, val_cand = hill_climb(h_cand, R_cand, val_cand, max_passes=100, time_limit=deadline - 15.0)
            if val_cand < best_val:
                best_val = val_cand
                best_h = h_cand.copy()
                best_R = R_cand.copy()

    # Phase 3: Multi-block annealing
    if time.time() < deadline - 20.0:
        h3, R3, val3 = multi_block_anneal(best_h, best_R, best_val, deadline - 10.0)
        if val3 < best_val:
            best_val = val3
            best_h = h3.copy()
            best_R = R3.copy()

    # Phase 4: Final continuous optimization from best
    if time.time() < deadline - 15.0:
        h_cand, _ = continuous_optimize(best_h.copy(), 100.0, deadline - 10.0)
        R_cand = compute_autocorr(h_cand)
        val_cand = get_max_autocorr(R_cand)
        h_cand, R_cand, val_cand = hill_climb(h_cand, R_cand, val_cand, max_passes=200, time_limit=deadline - 5.0)
        if val_cand < best_val:
            best_val = val_cand
            best_h = h_cand.copy()
            best_R = R_cand.copy()

    # Final polish
    if time.time() < deadline - 5.0:
        h5, R5, val5 = hill_climb(best_h, best_R, best_val, max_passes=300, time_limit=deadline - 2.0)
        if val5 < best_val:
            best_val = val5
            best_h = h5.copy()

    return best_h
# EVOLVE-BLOCK-END
