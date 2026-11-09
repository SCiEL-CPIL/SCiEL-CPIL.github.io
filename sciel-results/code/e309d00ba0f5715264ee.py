import time
import random
import math
import numpy as np
from scipy.optimize import minimize

def count_mono_4ap(colors):
    """
    Count monochromatic 4-APs using vectorized operations.
    colors: numpy array of shape (256,) with values 0 or 1.
    """
    n = 256
    c = colors
    if 'AP_INDICES' in globals():
        aps = AP_INDICES
        # c[aps] gives shape (N, 4)
        mono = (c[aps][:, 0] == c[aps][:, 1]) & \
               (c[aps][:, 1] == c[aps][:, 2]) & \
               (c[aps][:, 2] == c[aps][:, 3])
        return int(np.sum(mono))
    else:
        # Fallback
        count = 0
        for d in range(1, 86):
            max_a = n - 3*d
            for a in range(1, max_a+1):
                i = a-1
                if c[i] == c[i+d] == c[i+2*d] == c[i+3*d]:
                    count += 1
        return count

def thue_morse(n):
    seq = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def build_affected_and_others(n=256):
    """
    For each position p, store the indices of the other 3 elements in each 4-AP containing p.
    Returns a list of numpy arrays `others[p]` of shape (num_aps_containing_p, 3).
    Also returns the incidence matrix M (num_aps x n) where M[i, j] = 1 if position j is in AP i.
    """
    others = [None] * n
    ap_list = []
    for d in range(1, 86):
        max_a = n - 3*d
        for a in range(1, max_a+1):
            i0 = a - 1
            ap_list.append((i0, i0+d, i0+2*d, i0+3*d))
    
    num_aps = len(ap_list)
    temp_others = [[] for _ in range(n)]
    M = np.zeros((num_aps, n), dtype=np.int8)
    
    for ap_idx, ap in enumerate(ap_list):
        i0, i1, i2, i3 = ap
        temp_others[i0].append([i1, i2, i3])
        temp_others[i1].append([i0, i2, i3])
        temp_others[i2].append([i0, i1, i3])
        temp_others[i3].append([i0, i1, i2])
        
        M[ap_idx, i0] = 1
        M[ap_idx, i1] = 1
        M[ap_idx, i2] = 1
        M[ap_idx, i3] = 1

    for p in range(n):
        if temp_others[p]:
            others[p] = np.array(temp_others[p], dtype=np.int32)
        else:
            others[p] = np.empty((0, 3), dtype=np.int32)
            
    return others, np.array(ap_list, dtype=np.int32), M

def delta_mono_fast(c, p, others):
    """
    Calculate delta in mono count if we flip position p.
    """
    old_val = c[p]
    o = others[p]
    if len(o) == 0:
        return 0
    
    vals = c[o]
    sums = np.sum(vals, axis=1)
    
    if old_val == 0:
        delta = np.where(sums == 0, -1, 0) + np.where(sums == 3, 1, 0)
    else:
        delta = np.where(sums == 3, -1, 0) + np.where(sums == 0, 1, 0)
        
    return int(np.sum(delta))

def local_search_fast(c, deadline, others):
    """
    Greedy local search.
    """
    c = c.copy()
    cur_count = count_mono_4ap(c)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in range(256):
            if time.time() > deadline:
                break
            d = delta_mono_fast(c, p, others)
            if d < 0:
                c[p] = 1 - c[p]
                cur_count += d
                improved = True
    return c, cur_count

def simulated_annealing_fast(c_init, deadline, others, start_temp=15.0, end_temp=0.1):
    """
    Simulated Annealing with fast delta and restarts.
    """
    c = c_init.copy()
    cur_count = count_mono_4ap(c)
    best_c = c.copy()
    best_count = cur_count
    
    # Estimate step time
    t0 = time.time()
    trial_steps = 50
    for _ in range(trial_steps):
        p = random.randint(0, 255)
        _ = delta_mono_fast(c, p, others)
    t1 = time.time()
    time_per_step = (t1 - t0) / trial_steps
    
    budget = deadline - time.time()
    max_steps = int(budget / max(time_per_step, 1e-6))
    
    if max_steps <= 0:
        return best_c, best_count

    # We will run SA in phases, restarting if stuck
    phase_budget = max_steps // 5
    for phase in range(5):
        if time.time() > deadline - 2:
            break
            
        alpha = (end_temp / start_temp) ** (1.0 / phase_budget)
        temp = start_temp
        
        no_improve_steps = 0
        for i in range(phase_budget):
            if time.time() > deadline - 2:
                break
                
            p = random.randint(0, 255)
            d = delta_mono_fast(c, p, others)
            
            if d < 0:
                accept = True
            else:
                if temp < 1e-6:
                    accept = False
                else:
                    # Use log to avoid underflow if possible, but exp is fine for small d
                    ratio = math.exp(-d / temp)
                    accept = random.random() < ratio
            
            if accept:
                c[p] = 1 - c[p]
                cur_count += d
                if cur_count < best_count:
                    best_count = cur_count
                    best_c = c.copy()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
                
            temp *= alpha
            
            # Early restart if stuck for too long at low temp
            if no_improve_steps > phase_budget // 4 and temp < start_temp * 0.1:
                break
                
        # After each phase, polish with local search
        if time.time() < deadline - 5:
            c, cur_count = local_search_fast(c, deadline - 5, others)
            if cur_count < best_count:
                best_count = cur_count
                best_c = c.copy()
                
    return best_c, best_count

def qubo_energy_and_grad(x, ap_indices, lambda_bal=1.0):
    """
    Compute QUBO energy and gradient for continuous relaxation.
    x: continuous vector in [0,1]^n (or R^n), n=256
    Energy = sum over 4-APs of product of (2*c_i - 1) for the 4 positions
             + lambda_bal * (mean(c) - 0.5)^2
    We use x = 2*c - 1 (in [-1,1]) for the QUBO formulation.
    """
    # x is in [-1, 1] range representing 2*c - 1
    # Monochromatic when all 4 values are equal (all 1 or all -1)
    # Product of 4 values = 1 when monochromatic, = 1 when all same sign
    # Actually: for values in {-1,1}, product = 1 iff all same, = -1 if mixed
    # But in continuous relaxation, we use the product directly as a proxy
    
    aps = ap_indices  # shape (num_aps, 4)
    vals = x[aps]  # shape (num_aps, 4)
    
    # Product of the 4 values in each AP
    prod = np.prod(vals, axis=1)  # shape (num_aps,)
    
    # Energy: we want to MINIMIZE monochromatic, so minimize sum of prod
    # (prod = 1 means mono, prod < 1 means non-mono)
    # Actually we want to maximize non-mono, so minimize -sum(prod) or minimize sum(prod)
    # Let's minimize sum(prod) which is high when mono
    
    energy = np.sum(prod)
    
    # Balance term: penalize deviation from 50/50
    mean_x = np.mean(x)
    energy += lambda_bal * mean_x ** 2
    
    # Gradient
    # d/dx_i of sum over APs containing i of prod
    # For each AP j containing position i: d/dx_i (prod_j) = prod_j / x_i (if x_i != 0)
    # More carefully: if AP j has indices [i0,i1,i2,i3] and i = i_k, then
    # d/dx_{i_k} prod = prod / x_{i_k}
    
    grad = np.zeros_like(x)
    # For each AP, add contribution to each of its 4 positions
    for k in range(4):
        idx = aps[:, k]
        # prod / x[idx] with safe division
        safe_x = np.where(np.abs(x[idx]) < 1e-8, np.sign(x[idx]) * 1e-8 + 1e-8 * np.where(x[idx] >= 0, 1, -1), x[idx])
        contrib = prod / safe_x
        np.add.at(grad, idx, contrib)
    
    # Balance gradient
    grad += 2 * lambda_bal * mean_x / len(x)
    
    return energy, grad

def qubo_anneal(x_init, ap_indices, deadline, lambda_bal=1.0, n_restarts=3):
    """
    Penalty-annealing heuristic for the QUBO formulation.
    We minimize E(x) = sum_aps prod(x[ap]) + lambda * mean(x)^2
    over continuous x, then threshold to binary.
    Uses L-BFGS-B with time-based stopping.
    """
    best_c = None
    best_count = float('inf')
    n = len(x_init)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        if restart == 0:
            x0 = x_init.copy()
        else:
            # Perturb
            x0 = x_init + 0.3 * np.random.randn(n)
        
        # Use scipy minimize with L-BFGS-B
        try:
            result = minimize(
                lambda x: qubo_energy_and_grad(x, ap_indices, lambda_bal)[0],
                x0,
                jac=lambda x: qubo_energy_and_grad(x, ap_indices, lambda_bal)[1],
                method='L-BFGS-B',
                bounds=[(-1.5, 1.5)] * n,
                options={'maxiter': 200, 'maxfun': 5000}
            )
            x_opt = result.x
            
            # Threshold to binary
            c_candidate = (x_opt > 0).astype(np.int8)
            cnt = count_mono_4ap(c_candidate)
            
            if cnt < best_count:
                best_count = cnt
                best_c = c_candidate.copy()
                
        except Exception:
            continue
        
        # Also try flipping the threshold
        if time.time() < deadline - 5:
            c_flip = (x_opt <= 0).astype(np.int8)
            cnt_flip = count_mono_4ap(c_flip)
            if cnt_flip < best_count:
                best_count = cnt_flip
                best_c = c_flip.copy()
    
    return best_c, best_count

def spectral_refine(c, M, k=15):
    """
    Project the centered coloring vector onto the subspace orthogonal to the top k right singular vectors of M.
    Re-threshold to binary.
    """
    n = len(c)
    c_float = c.astype(np.float64)
    mean_c = np.mean(c_float)
    x = c_float - mean_c
    
    try:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        top_k_vt = Vt[:k, :]
        V_top = top_k_vt.T
        x_proj = x - V_top @ (V_top.T @ x)
        c_new = (x_proj > 0).astype(np.int8)
        return c_new
    except Exception:
        return c.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s margin
    
    n = 256
    
    # Precompute structures
    others, ap_indices, M = build_affected_and_others(n)
    global AP_INDICES
    AP_INDICES = ap_indices
    
    # Generate candidates
    candidates = []
    
    # 1. Thue-Morse
    tm = thue_morse(n)
    candidates.append(tm)
    candidates.append(1 - tm)
    
    # 2. Random
    for _ in range(10):
        candidates.append(np.random.randint(0, 2, n, dtype=np.int8))
        
    # 3. Periodic
    p2 = np.array([i % 2 for i in range(n)], dtype=np.int8)
    candidates.append(p2)
    p3 = np.array([i % 3 % 2 for i in range(n)], dtype=np.int8)
    candidates.append(p3)
    p4 = np.array([i % 4 % 2 for i in range(n)], dtype=np.int8)
    candidates.append(p4)
    
    # 4. Spectral Initializations
    # Use SVD of M to generate diverse starting points
    try:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Try projecting random vectors onto the subspace spanned by the bottom singular vectors
        # or orthogonal to the top ones.
        k_init = 20
        V_top_init = Vt[:k_init, :].T # (n, k_init)
        
        for _ in range(5):
            r = np.random.randn(n)
            r_proj = r - V_top_init @ (V_top_init.T @ r)
            c_spec = (r_proj > 0).astype(np.int8)
            candidates.append(c_spec)
            
        # Also try the eigenvectors themselves as starting points (thresholded)
        # The smallest singular vectors might be good
        for i in range(5, 15):
            if i < len(Vt):
                v = Vt[i, :]
                c_v = (v > np.median(v)).astype(np.int8)
                candidates.append(c_v)
    except Exception:
        pass

    # Evaluate initial candidates
    best_c = None
    best_count = float('inf')
    evaluated = []
    
    for cand in candidates:
        if time.time() > deadline - 20:
            break
        cnt = count_mono_4ap(cand)
        evaluated.append((cnt, cand))
        
    # Sort by count
    evaluated.sort(key=lambda x: x[0])
    
    # Take top 3 for detailed optimization
    top_candidates = [cand for cnt, cand in evaluated[:3]]
    
    if not top_candidates:
        top_candidates = [candidates[0]]
        
    # Optimize each top candidate
    for i, cand in enumerate(top_candidates):
        if time.time() > deadline - 10:
            break
            
        # 1. Local Search
        c_ls, cnt_ls = local_search_fast(cand, deadline - 20, others)
        if cnt_ls < best_count:
            best_count = cnt_ls
            best_c = c_ls.copy()
            
        # 2. QUBO Penalty Annealing (NEW)
        if time.time() < deadline - 15:
            # Convert to x = 2*c - 1
            x_init = (2 * c_ls.astype(np.float64) - 1.0)
            qubo_c, qubo_count = qubo_anneal(x_init, ap_indices, deadline - 15, lambda_bal=0.5, n_restarts=2)
            if qubo_c is not None and qubo_count < best_count:
                best_count = qubo_count
                best_c = qubo_c.copy()
                # Polish QUBO result
                c_polish, cnt_polish = local_search_fast(qubo_c, deadline - 10, others)
                if cnt_polish < best_count:
                    best_count = cnt_polish
                    best_c = c_polish.copy()
            
        # 3. Spectral Refinement
        if time.time() < deadline - 10:
            c_spec = spectral_refine(c_ls, M, k=10)
            cnt_spec = count_mono_4ap(c_spec)
            if cnt_spec < best_count:
                best_count = cnt_spec
                best_c = c_spec.copy()
                # Polish spectral result
                c_polish, cnt_polish = local_search_fast(c_spec, deadline - 8, others)
                if cnt_polish < best_count:
                    best_count = cnt_polish
                    best_c = c_polish.copy()
                    
        # 4. Simulated Annealing
        if time.time() < deadline - 5:
            sa_best_c, sa_best_count = simulated_annealing_fast(c_ls, deadline - 2, others)
            if sa_best_count < best_count:
                best_count = sa_best_count
                best_c = sa_best_c.copy()
                
    # Final Polish
    if time.time() < deadline - 2:
        best_c, best_count = local_search_fast(best_c, deadline - 2, others)
        
    return best_c.tolist()

if __name__ == "__main__":
    res = construct()
    assert len(res) == 256
    assert all(x in [0, 1] for x in res)
    print("Count:", count_mono_4ap(np.array(res, dtype=np.int8)))
