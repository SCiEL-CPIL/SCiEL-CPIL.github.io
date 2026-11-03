import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
import sys

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _soft_max(x, beta=50.0):
    """Smooth approximation of max via log-sum-exp."""
    x_max = np.max(x)
    if x_max < 1e-12:
        return x_max
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _score_raw(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||autoconv(h)||_inf) with proper scaling."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    ac = np.convolve(h, h, mode='full')
    ac_max = DELTA * np.max(ac)
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    def elapsed():
        return time.time() - start_time
    
    # Initial guess: uniform
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    np.random.seed(42)
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse structured
    for trial in range(20):
        k = np.random.randint(3, 12)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Concentrated Gaussians
    for c in np.linspace(2, N-3, 10):
        for w in np.linspace(0.5, N/3, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 4. Triangular
    for c in np.linspace(2, N-3, 8):
        for w in np.linspace(2, N, 5):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(15):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
            
    # 6. Structured sparse: symmetric pairs
    for k in range(3, 8):
        for offset in range(1, 6):
            h_pair = np.zeros(N)
            center = N // 2
            for i in range(k):
                pos1 = center - offset - i
                pos2 = center + offset + i
                if 0 <= pos1 < N:
                    h_pair[pos1] = 1.0
                if 0 <= pos2 < N:
                    h_pair[pos2] = 1.0
            if np.sum(h_pair) > 0:
                h_pair /= np.sum(h_pair)
                inits.append(h_pair)
    
    # 7. Sidon set initializations
    # A Sidon set has all pairwise sums distinct, minimizing autoconvolution collisions
    sidon_sets = []
    # Greedy construction of Sidon sets
    def _is_sidon(S):
        S = sorted(S)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def _build_sidon(target_size, n_max):
        """Build a Sidon set of given size from [0, n_max)."""
        S = [0]
        for candidate in range(1, n_max):
            if len(S) >= target_size:
                break
            test_S = S + [candidate]
            if _is_sidon(test_S):
                S.append(candidate)
        return S if len(S) >= 3 else None
    
    # Try various Sidon set sizes
    for size in [4, 5, 6, 7, 8]:
        S = _build_sidon(size, N)
        if S is not None:
            h_sidon = np.zeros(N)
            for idx in S:
                h_sidon[idx] = 1.0
            h_sidon /= np.sum(h_sidon)
            inits.append(h_sidon)
            # Also try with slight perturbations
            for perturb in [0.1, 0.3, 0.5]:
                h_pert = h_sidon.copy()
                for idx in S:
                    h_pert[idx] *= (1 + perturb * np.random.randn())
                h_pert = np.maximum(h_pert, 0)
                if np.sum(h_pert) > 1e-10:
                    h_pert /= np.sum(h_pert)
                    inits.append(h_pert)

    # --- Phase 2: Epigraph Formulation Optimization ---
    # Variables: x = [h_0, ..., h_{N-1}, M]
    # Objective: Minimize M (x[N])
    # Constraints:
    # 1. sum(h) - 1 == 0
    # 2. h_i >= 0 (via bounds)
    # 3. M - conv_k >= 0 for all k
    
    num_conv = 2 * N - 1
    
    def objective(x):
        return x[N]
        
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
        
    def constraint_eq_fun(x):
        return np.sum(x[:N]) - 1.0
        
    def constraint_eq_jac(x):
        jac = np.zeros(N + 1)
        jac[:N] = 1.0
        return jac
        
    def constraint_ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        # Use soft-max for smoother optimization
        return M - _soft_max(conv, beta=30.0)
        
    def constraint_ineq_jac(x):
        h = x[:N]
        jac = np.zeros((num_conv, N + 1))
        jac[:, N] = 1.0
        # d(M - conv_k)/dh_j = -2 * h_{k-j} where valid
        for j in range(N):
            k_start = j
            k_end = min(j + N, num_conv)
            if k_start < num_conv and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] = -2.0 * h[indices_h]
        return jac
        
    constraints = [
        {
            'type': 'eq',
            'fun': constraint_eq_fun,
            'jac': constraint_eq_jac
        },
        {
            'type': 'ineq',
            'fun': constraint_ineq_fun,
            'jac': constraint_ineq_jac
        }
    ]
    
    bounds = [(0.0, 1.0)] * N + [(0.0, None)]
    
    def run_epigraph(h_init, maxiter=200, ftol=1e-10):
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        conv_init = np.convolve(h_init, h_init, mode='full')
        M_init = np.max(conv_init) + 1e-6
        x0 = np.concatenate((h_init, [M_init]))
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    return h_candidate
        except Exception:
            pass
        return None
    
    # Run epigraph from multiple inits
    for h_init in inits:
        if elapsed() > TIME_LIMIT - 50:
            break
        h_candidate = run_epigraph(h_init, maxiter=200, ftol=1e-10)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # --- Phase 2.5: Direct ratio maximization with L-BFGS-B ---
    def _neg_ratio(h):
        """Negative of the ratio for minimization."""
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        l2_sq = np.sum(h**2)
        ac = np.convolve(h, h, mode='full')
        ac_max = DELTA * np.max(ac)
        if ac_max < 1e-12:
            return 0.0
        return -(l2_sq / (l1 * ac_max))
    
    def _neg_ratio_grad(h):
        """Gradient of negative ratio (numerical)."""
        eps = 1e-7
        grad = np.zeros_like(h)
        f0 = _neg_ratio(h)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            grad[i] = (_neg_ratio(h_plus) - f0) / eps
        return grad
    
    # Try L-BFGS-B from best_h and top few inits
    if elapsed() < TIME_LIMIT - 30:
        top_inits = [best_h] + inits[:5]
        for h_init in top_inits:
            if elapsed() > TIME_LIMIT - 30:
                break
            h_init = np.maximum(h_init, 0)
            if np.sum(h_init) < 1e-15:
                continue
            h_init = h_init / np.sum(h_init)
            try:
                res = minimize(
                    _neg_ratio,
                    h_init,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * N,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_candidate = np.maximum(res.x, 0)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
            except Exception:
                pass
    
    # --- Phase 3: Block-Coordinate Descent Refinement ---
    def block_coordinate_refine(h_start, max_iter=30):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if elapsed() > TIME_LIMIT - 15:
                break
            improved = False
            iter_count += 1
            
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            M_val = conv[k_star]
            
            # For each position i, try to optimize h[i] and h[k_star - i] (if valid)
            for i in range(N):
                if elapsed() > TIME_LIMIT - 15:
                    break
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                    
                hi = h[i]
                hj = h[j]
                
                t_min = -hi
                t_max = hj
                
                if t_max - t_min < 1e-15:
                    continue
                    
                best_t = 0.0
                best_local_score = _score_raw(h)
                
                for t in np.linspace(t_min * 0.8, t_max * 0.8, 8):
                    h_test = h.copy()
                    h_test[i] += t
                    h_test[j] -= t
                    if np.any(h_test < -1e-15):
                        continue
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) > 1e-15:
                        h_test /= np.sum(h_test)
                        s_test = _score_raw(h_test)
                        if s_test > best_local_score:
                            best_local_score = s_test
                            best_t = t
                
                if abs(best_t) > 1e-12:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    if np.sum(h) > 1e-15:
                        h /= np.sum(h)
                    improved = True
            
            # Also try single-coordinate updates
            for i in range(N):
                if elapsed() > TIME_LIMIT - 15:
                    break
                hi = h[i]
                if hi < 1e-15:
                    continue
                    
                # Try moving mass from i to other positions
                for j in range(N):
                    if j == i:
                        continue
                    hj = h[j]
                    t_min = -hi
                    t_max = 0.0  # Can't increase h[j] beyond its current value in this simple scheme
                    # Actually, let's just try reducing h[i] and distributing to others
                    # Simpler: try h[i] -> h[i] - delta, and add delta to random other position
                    
                # Try reducing h[i] by small amount
                for delta in [hi * 0.01, hi * 0.05, hi * 0.1, hi * 0.2]:
                    h_test = h.copy()
                    h_test[i] -= delta
                    # Distribute delta to the position with highest gradient
                    # For simplicity, add to the position that maximizes score
                    best_j = -1
                    best_s = best_local_score
                    for j in range(N):
                        h_test[j] = h[j] + delta
                        if np.any(h_test < -1e-15):
                            h_test[j] = h[j]
                            continue
                        s_test = _score_raw(h_test)
                        if s_test > best_s:
                            best_s = s_test
                            best_j = j
                    if best_j >= 0:
                        h[i] -= delta
                        h[best_j] += delta
                        h = np.clip(h, 0, None)
                        if np.sum(h) > 1e-15:
                            h /= np.sum(h)
                        improved = True
                        break
                        
        return h
    
    # Apply block-coordinate refinement to best_h
    if elapsed() < TIME_LIMIT - 20:
        h_refined = block_coordinate_refine(best_h, max_iter=30)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # --- Phase 4: Additional random restarts with epigraph ---
    for i in range(30):
        if elapsed() > TIME_LIMIT - 10:
            break
        h_init = np.random.dirichlet(np.ones(N) * 0.3)  # more peaked
        h_candidate = run_epigraph(h_init, maxiter=150, ftol=1e-9)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # Final refinement
    if elapsed() < TIME_LIMIT - 10:
        h_refined = block_coordinate_refine(best_h, max_iter=20)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Final high-precision polish: tighten support and re-optimize
    if elapsed() < TIME_LIMIT - 5:
        # Zero out very small entries
        h_polish = best_h.copy()
        h_polish[h_polish < 1e-6] = 0.0
        if np.sum(h_polish) > 1e-15:
            h_polish /= np.sum(h_polish)
            # Run one more epigraph optimization from this tightened start
            h_candidate = run_epigraph(h_polish, maxiter=300, ftol=1e-12)
            if h_candidate is not None:
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
            # Also try L-BFGS-B polish
            try:
                res = minimize(
                    _neg_ratio,
                    h_polish,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * N,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                h_candidate = np.maximum(res.x, 0)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
            except Exception:
                pass
    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print("Score:", _score_raw(h))
    print("Heights:", h)
