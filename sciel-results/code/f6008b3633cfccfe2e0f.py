"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(100):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final precise adjustment
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-14:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _dp_discretized(M_levels, target_sum, N_steps):
    """
    Dynamic programming over discretized height levels.
    
    Nodes: (step_index, height_level, accumulated_sum_bucket)
    Edge cost: local overlap penalty h_i * (1 - h_{i+1})
    
    We discretize heights into M_levels bins in [0,1].
    The accumulated sum is tracked in integer units of 1/M_levels.
    Target sum = N/2 = N*M_levels/2 in these units.
    
    Returns: best path (sequence of height levels) and its score.
    """
    # Height levels: M_levels discrete values in [0, 1]
    levels = np.linspace(0.0, 1.0, M_levels)
    # Accumulated sum in units of 1/M_levels
    # Max possible sum: N * 1.0 = N*M_levels units
    max_sum_units = N * M_levels
    target_units = int(round(N / 2.0 * M_levels))
    
    # DP: dp[step][height_idx][sum_units] = min cost to reach this state
    # Cost is the sum of overlap penalties so far
    # We want to minimize the MAX overlap, but for DP we can use sum as a proxy
    # or track max separately. Let's use sum of overlaps as cost for DP,
    # then refine.
    
    # To keep memory manageable, we limit sum_units range
    # At step i, sum can range from 0 to i*M_levels
    # But we need to reach target_units at step N
    
    # Use a sparse approach: only track reachable (height_idx, sum_units) pairs
    # dp[i] = dict mapping (h_idx, s_units) -> (min_cost, prev_h_idx)
    
    INF = float('inf')
    
    # Initialize: at step 0, h_0 can be any level, sum = level
    dp = [None] * (N + 1)
    dp[0] = {}
    for hi in range(M_levels):
        h_val = levels[hi]
        s_units = int(round(h_val * M_levels))
        # Cost: no overlap yet (or we can define overlap[0] = h_0 * (1-0) = h_0)
        # Actually overlap[i] = h[i] * (1 - h[i+1]) for i < N-1, and h[N-1] for i=N-1
        # So at step 0, we haven't computed any overlap yet
        cost = 0.0
        dp[0][(hi, s_units)] = (cost, -1)
    
    # Transition
    for step in range(N - 1):
        dp[step + 1] = {}
        for (hi, s_units), (cost, prev_hi) in dp[step].items():
            h_curr = levels[hi]
            for hi_next in range(M_levels):
                h_next = levels[hi_next]
                # Overlap at this step: h_curr * (1 - h_next)
                overlap = h_curr * (1.0 - h_next)
                new_cost = cost + overlap
                new_s = s_units + int(round(h_next * M_levels))
                
                # Pruning: only keep states that can still reach target
                remaining_steps = N - 1 - step
                min_possible_remaining = 0  # min sum from remaining steps
                max_possible_remaining = remaining_steps * M_levels  # max sum
                if new_s + min_possible_remaining > target_units:
                    continue
                if new_s + max_possible_remaining < target_units:
                    continue
                
                key = (hi_next, new_s)
                if key not in dp[step + 1] or new_cost < dp[step + 1][key][0]:
                    dp[step + 1][key] = (new_cost, hi)
        
        # Memory management: if too many states, prune
        if len(dp[step + 1]) > 500000:
            # Keep only best 500000 states by cost
            items = sorted(dp[step + 1].items(), key=lambda x: x[1][0])
            dp[step + 1] = dict(items[:500000])
    
    # Final step: add the last overlap h[N-1] * (1 - 0) = h[N-1]
    best_cost = INF
    best_path = None
    for (hi, s_units), (cost, prev_hi) in dp[N - 1].items():
        if s_units != target_units:
            continue
        h_last = levels[hi]
        final_cost = cost + h_last  # last overlap is h[N-1]
        if final_cost < best_cost:
            best_cost = final_cost
            # Reconstruct path
            path = [hi]
            curr_key = (hi, s_units)
            for step in range(N - 1, 0, -1):
                prev_h = dp[step][curr_key][1]
                # Find the sum at previous step
                h_curr_val = levels[curr_key[0]]
                s_curr = curr_key[1]
                s_prev = s_curr - int(round(h_curr_val * M_levels))
                curr_key = (prev_h, s_prev)
                path.append(prev_h)
            path.reverse()
            best_path = path
    
    if best_path is None:
        return None, INF
    
    # Convert to continuous heights
    h_discrete = np.array([levels[hi] for hi in best_path])
    return h_discrete, best_cost

def _refine_continuous(h_init, max_iters=500, tol=1e-12):
    """
    Refine a discrete solution to continuous using local search.
    Uses projected gradient descent with the exact max-overlap objective.
    """
    h = h_init.copy().astype(float)
    h = _enforce_sum(h)
    
    current_score = _score(h)
    
    # Use subgradient method for min-max problem
    # We want to minimize max_i overlap[i] subject to sum(h) = N/2, 0 <= h <= 1
    # Use a soft-max approach with increasing sharpness
    
    alpha = 50.0
    for it in range(max_iters):
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        
        # Soft max gradient
        x_max = np.max(overlap)
        exps = np.exp(alpha * (overlap - x_max))
        weights = exps / np.sum(exps)
        
        # Gradient of soft max w.r.t. h
        grad = np.zeros(N)
        for i in range(N - 1):
            # overlap[i] = h[i] * (1 - h[i+1])
            # d/dh[i] = (1 - h[i+1])
            # d/dh[i+1] = -h[i]
            grad[i] += weights[i] * (1.0 - h[i+1])
            grad[i+1] += weights[i] * (-h[i])
        # overlap[N-1] = h[N-1] * (1 - 0) = h[N-1]
        grad[N-1] += weights[N-1] * 1.0
        
        # Projected gradient step with sum constraint
        # First, project gradient onto the hyperplane sum(h) = N/2
        grad_proj = grad - np.mean(grad)
        
        # Adaptive step size
        step_size = 0.01
        for _ in range(10):
            h_new = h - step_size * grad_proj
            h_new = np.clip(h_new, 0.0, 1.0)
            h_new = _enforce_sum(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
                break
            step_size *= 0.5
        
        # Increase sharpness
        alpha *= 1.2
        if alpha > 5000:
            alpha = 5000
        
        if it % 50 == 0:
            # Check for convergence
            if np.max(np.abs(grad_proj)) < tol:
                break
    
    return h, current_score

def _two_block_analytic():
    """
    Construct a two-block analytic initial solution.
    Based on the idea that optimal solutions may have a block structure.
    """
    # Try various block patterns
    best_h = None
    best_score = float('inf')
    
    # Pattern: alternating blocks of different heights
    for n_blocks in range(2, 20):
        block_len = N // n_blocks
        remainder = N % n_blocks
        
        # Generate block heights
        for _ in range(50):
            # Random block heights that average to 0.5
            block_heights = np.random.uniform(0.1, 0.9, n_blocks)
            # Scale to have mean 0.5
            block_heights = 0.5 + (block_heights - np.mean(block_heights)) * 0.5
            block_heights = np.clip(block_heights, 0.0, 1.0)
            
            # Build full sequence
            h = np.zeros(N)
            idx = 0
            for b in range(n_blocks):
                length = block_len + (1 if b < remainder else 0)
                h[idx:idx+length] = block_heights[b]
                idx += length
            
            h = _enforce_sum(h)
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Start with constant solution
    h_const = np.full(N, 0.5)
    current_score = _score(h_const)
    h_final = h_const.copy()
    
    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _enforce_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - N/2.0) < 1e-10:
                sc = _score(h_c)
                if sc < current_score - 1e-15:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False
    
    # Phase 1: DP with discretized heights
    try:
        if time.time() < deadline - 60.0:
            # Try different levels of discretization
            for M_levels in [20, 30, 40, 50]:
                if time.time() > deadline - 55.0:
                    break
                t0 = time.time()
                h_dp, cost_dp = _dp_discretized(M_levels, N/2.0, N)
                t1 = time.time()
                if h_dp is not None:
                    # Refine the discrete solution
                    h_refined, score_refined = _refine_continuous(h_dp, max_iters=200)
                    _try_update(h_refined)
                    if t1 - t0 > 30:
                        break  # Too slow, stop trying larger M
    except Exception:
        pass
    
    # Phase 2: Two-block analytic initialization + refinement
    try:
        if time.time() < deadline - 40.0:
            h_2block, score_2block = _two_block_analytic()
            if h_2block is not None:
                _try_update(h_2block)
                # Refine from this point
                h_ref, score_ref = _refine_continuous(h_2block, max_iters=300)
                _try_update(h_ref)
    except Exception:
        pass
    
    # Phase 3: SLSQP epigraph optimization from current best
    try:
        if time.time() < deadline - 20.0:
            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                x_max = np.max(overlap)
                smax = x_max + (1.0/500.0) * np.log(np.sum(np.exp(500.0 * (overlap - x_max))))
                return t + 10000.0 * np.maximum(0.0, smax - t)**2
            
            x0 = np.concatenate([h_final, [current_score + 0.01]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0
            
            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)
            
            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]
            
            result = minimize(
                objective_smooth,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
            )
            
            if result.fun < 100.0:
                _try_update(result.x[:N])
    except Exception:
        pass
    
    # Phase 4: Multi-start with diverse initializations
    try:
        if time.time() < deadline - 10.0:
            np.random.seed(42)
            n_starts = 6
            for trial in range(n_starts):
                if time.time() > deadline - 5.0:
                    break
                
                if trial == 0:
                    freq = 1.0
                    h_rand = 0.5 + 0.3 * np.sin(2 * np.pi * freq * np.arange(N) / N)
                elif trial == 1:
                    freq = 2.0
                    h_rand = 0.5 + 0.25 * np.sin(2 * np.pi * freq * np.arange(N) / N)
                elif trial == 2:
                    freq = 3.0
                    h_rand = 0.5 + 0.2 * np.sin(2 * np.pi * freq * np.arange(N) / N)
                elif trial == 3:
                    h_rand = 0.5 + 0.15 * np.sin(2 * np.pi * np.arange(N) / N) + 0.1 * np.sin(4 * np.pi * np.arange(N) / N)
                elif trial == 4:
                    # Sawtooth-like
                    h_rand = 0.5 + 0.3 * np.sign(np.sin(2 * np.pi * 2 * np.arange(N) / N)) * np.abs(np.sin(2 * np.pi * np.arange(N) / N))
                else:
                    # Random smooth
                    h_rand = np.random.rand(N)
                    kernel = np.ones(9) / 9
                    h_rand = np.convolve(h_rand, kernel, mode='same')
                    h_rand = 0.2 + 0.6 * (h_rand - h_rand.min()) / (h_rand.max() - h_rand.min() + 1e-10)
                
                h_rand = np.clip(h_rand, 0.0, 1.0)
                h_rand = _enforce_sum(h_rand)
                
                # Quick refinement
                h_ref, _ = _refine_continuous(h_rand, max_iters=100)
                _try_update(h_ref)
    except Exception:
        pass
    
    # Phase 5: Final polish with SLSQP
    try:
        if time.time() < deadline - 2.0:
            def obj_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                x_max = np.max(overlap)
                smax = x_max + (1.0/1000.0) * np.log(np.sum(np.exp(1000.0 * (overlap - x_max))))
                return t + 50000.0 * np.maximum(0.0, smax - t)**2
            
            x0_p = np.concatenate([h_final, [current_score + 0.001]])
            bounds_p = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def cons_sum_p(params):
                return np.sum(params[:N]) - N/2.0
            
            def cons_epi_p(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)
            
            cons_p = [
                {'type': 'eq', 'fun': cons_sum_p},
                {'type': 'ineq', 'fun': cons_epi_p}
            ]
            
            res_p = minimize(
                obj_polish,
                x0_p,
                method='SLSQP',
                bounds=bounds_p,
                constraints=cons_p,
                options={'maxiter': 3000, 'ftol': 1e-16, 'disp': False}
            )
            
            if res_p.fun < 100.0:
                _try_update(res_p.x[:N])
    except Exception:
        pass
    
    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
