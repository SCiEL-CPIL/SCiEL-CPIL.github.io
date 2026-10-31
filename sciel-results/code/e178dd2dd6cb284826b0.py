"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h (adjacent shift)."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _overlap_max(h):
    """Compute max of h[i]*(1-h[i+1]) with zero-padding at the end."""
    # h[i] * (1 - h[i+1]) for i=0..N-2, and h[N-1] * (1 - 0) = h[N-1]
    overlaps = h[:-1] * (1.0 - h[1:])
    # Last element: h[N-1] * (1 - 0) = h[N-1]
    overlaps = np.append(overlaps, h[-1])
    return overlaps.max()

def _smooth_max(values, alpha):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(values)
    if m > 1e10:
        return m
    shifted = alpha * (values - m)
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted))) / alpha

def _objective(h, alpha):
    """Smooth surrogate objective: smooth max of adjacent overlaps."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    return _smooth_max(overlaps, alpha)

def _objective_grad(h, alpha):
    """Gradient of the smooth max objective."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    m = np.max(overlaps)
    shifted = alpha * (overlaps - m)
    shifted = np.clip(shifted, -50, 50)
    exp_vals = np.exp(shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of overlaps with respect to h
    # overlap[i] = h[i]*(1-h[i+1]) for i < N-1
    # overlap[N-1] = h[N-1]
    grad = np.zeros(N)
    for i in range(N - 1):
        # d(overlap[i])/dh[i] = (1 - h[i+1])
        # d(overlap[i])/dh[i+1] = -h[i]
        grad[i] += weights[i] * (1.0 - h[i+1])
        grad[i+1] += weights[i] * (-h[i])
    # Last overlap: d(h[N-1])/dh[N-1] = 1
    grad[N-1] += weights[N-1] * 1.0
    
    return grad

def _enforce_constraints(h):
    """Project h onto feasible set: [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    target = N / 2.0
    if abs(s - target) > 1e-12:
        diff = target - s
        # Distribute residual to elements not at boundary
        can_add = h < 1.0
        can_sub = h > 0.0
        if diff > 0:
            mask = can_add
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
            else:
                h += diff / N
        else:
            mask = can_sub
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
            else:
                h += diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final correction
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _optimize_from_start(h0, alpha, max_iter=200):
    """Optimize from a starting point using SLSQP."""
    n = len(h0)
    target = n / 2.0
    
    bounds = [(0.0, 1.0)] * n
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - target, 
                    'jac': lambda x: np.ones(n)}]
    
    def obj(x):
        return _objective(x, alpha)
    
    def obj_grad(x):
        return _objective_grad(x, alpha)
    
    try:
        res = minimize(obj, h0, jac=obj_grad, method='SLSQP', bounds=bounds, 
                      constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
        if res.success or (hasattr(res, 'fun') and res.fun < _objective(h0, alpha)):
            return res.x
    except Exception:
        pass
    return h0

def _try_ramp_pattern(deadline):
    """Try a ramp/plateau pattern: values rise from 0 to 1 and fall back."""
    best_h = None
    best_score = float('inf')
    
    # Try various ramp widths and positions
    for width in range(5, N+1, 2):
        for offset in range(0, N - width + 1, 2):
            h = np.zeros(N)
            # Ramp up
            ramp_len = min(width // 2, N - offset)
            if ramp_len > 0:
                h[offset:offset+ramp_len] = np.linspace(0, 1, ramp_len)
            # Plateau
            plateau_start = offset + ramp_len
            plateau_end = offset + width
            if plateau_start < plateau_end and plateau_end <= N:
                h[plateau_start:plateau_end] = 1.0
            # Ramp down
            if plateau_end < N:
                ramp_down_len = min(width // 2, N - plateau_end)
                if ramp_down_len > 0:
                    h[plateau_end:plateau_end+ramp_down_len] = np.linspace(1, 0, ramp_down_len)
            
            h = _enforce_constraints(h)
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        if time.time() > deadline - 20:
            break
    
    return best_h, best_score

def _try_block_pattern(deadline):
    """Try block patterns with smooth transitions."""
    best_h = None
    best_score = float('inf')
    
    # Try different block configurations
    for n_blocks in [2, 3, 4, 5]:
        for block_width in range(5, N // n_blocks + 1, 3):
            h = np.zeros(N)
            for i in range(n_blocks):
                start = i * (N // n_blocks)
                end = start + block_width
                if end <= N:
                    # Smooth transition
                    trans = min(3, block_width // 2)
                    if trans > 0:
                        h[start:start+trans] = np.linspace(0, 1, trans)
                        h[end-trans:end] = np.linspace(1, 0, trans)
                        if block_width > 2 * trans:
                            h[start+trans:end-trans] = 1.0
                    else:
                        h[start:end] = 1.0
            
            h = _enforce_constraints(h)
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        if time.time() > deadline - 20:
            break
    
    return best_h, best_score

def _dp_check(M, K=200):
    """
    Check if there exists a sequence h with max overlap <= M and sum = N/2.
    Returns the sequence h if found, else None.
    """
    heights = np.linspace(0, 1, K)
    
    # L[i][j] = min sum ending at height j at step i
    # U[i][j] = max sum ending at height j at step i
    # We only need the previous step to compute the next.
    # But we need to backtrack, so we store the parent or the decision.
    # To save memory, we can store the argmin/argmax index for backtracking.
    # Or just store L and U for all steps if K and N are small enough.
    # N=100, K=200. 100*200*8 bytes = 160KB. Very small.
    
    L = np.zeros((N, K))
    U = np.zeros((N, K))
    parent = np.zeros((N, K), dtype=int) # parent[j] at step i is the index j_prev
    
    # Initialize step 0
    # h_0 can be any height. Sum = h_0.
    L[0, :] = heights
    U[0, :] = heights
    parent[0, :] = -1 # No parent
    
    # Precompute constraints for speed?
    # For a given M and h_next (j'), find max h_curr (j) such that h_curr * (1 - h_next) <= M.
    # h_curr <= M / (1 - h_next)
    
    for i in range(N - 1):
        # Compute L[i+1] and U[i+1]
        # For each j_next in 0..K-1:
        #   find max_j such that heights[max_j] * (1 - heights[j_next]) <= M
        #   L[i+1, j_next] = min(L[i, :max_j+1]) + heights[j_next]
        #   U[i+1, j_next] = max(U[i, :max_j+1]) + heights[j_next]
        #   parent[i+1, j_next] = argmin(L[i, :max_j+1]) or argmax
        
        # Vectorize the constraint check
        # h_curr * (1 - h_next) <= M
        # (1 - h_next) is a vector of size K (for h_next)
        # h_curr is a vector of size K (for h_curr)
        
        # We want for each j_next, the largest index j_curr such that condition holds.
        # Since heights are sorted, the condition holds for j_curr <= some bound.
        
        one_minus_h_next = 1.0 - heights
        # Avoid division by zero
        # If h_next = 1, 1-h_next = 0, condition is 0 <= M (always true if M>=0).
        # So bound is K-1.
        
        # Calculate bound for each j_next
        # bound_val = M / (1 - h_next)
        # If 1-h_next is 0, bound_val is inf.
        
        with np.errstate(divide='ignore', invalid='ignore'):
            bound_vals = M / one_minus_h_next
            bound_vals[one_minus_h_next == 0] = np.inf
        
        # Find indices
        # np.searchsorted(heights, bound_vals, side='right') - 1
        # This gives the index of the last height <= bound_val.
        max_indices = np.searchsorted(heights, bound_vals, side='right') - 1
        # Clamp to [0, K-1]
        max_indices = np.clip(max_indices, 0, K - 1)
        
        # Now, for each j_next, we need min(L[i, :max_indices[j_next]+1])
        # This is a prefix min operation, but the prefix length varies.
        # We can compute prefix min of L[i] and prefix max of U[i].
        
        # Prefix min: P_min[k] = min(L[i, 0..k])
        # We can use np.accumulate with minimum, but we need to handle the variable length.
        # Actually, if we compute the full prefix min array PMin where PMin[k] = min(L[i, 0..k]),
        # then min(L[i, 0..max_idx]) is just PMin[max_idx].
        
        PMin = np.minimum.accumulate(L[i, :])
        PMax = np.maximum.accumulate(U[i, :])
        
        # Get the values
        min_sums = PMin[max_indices]
        max_sums = PMax[max_indices]
        
        # Find the argmin/argmax for backtracking
        # We need the index j_curr that achieved the min/max in the prefix.
        # This is harder to vectorize directly with accumulate.
        # However, for backtracking, we can just store the max_index and assume the "best" path
        # or we can refine the backtracking.
        # Actually, to reconstruct the path, we need the specific j_curr.
        # Let's store the j_curr that gives the min sum.
        # We can find this by iterating? No, too slow.
        # Alternative: Just store the max_index. When backtracking, we can check which j_curr in the prefix
        # actually leads to the target sum?
        # Or simpler: The DP guarantees existence. We can just pick any valid path.
        # Let's store the index of the minimum in the prefix.
        # We can compute this by:
        # For each j_next, we want argmin(L[i, 0..max_idx]).
        # We can precompute the argmin for all prefixes?
        # ArgMinPrefix[k] = argmin(L[i, 0..k]).
        
        # Let's compute ArgMinPrefix and ArgMaxPrefix.
        # This can be done in a loop or using a trick.
        # Since K=200, a loop over K for each i is 200*100 = 20000 iterations. Fine.
        
        arg_min_prefix = np.zeros(K, dtype=int)
        arg_max_prefix = np.zeros(K, dtype=int)
        
        current_min_val = np.inf
        current_max_val = -np.inf
        current_min_idx = 0
        current_max_idx = 0
        
        for k in range(K):
            if L[i, k] < current_min_val:
                current_min_val = L[i, k]
                current_min_idx = k
            if U[i, k] > current_max_val:
                current_max_val = U[i, k]
                current_max_idx = k
            arg_min_prefix[k] = current_min_idx
            arg_max_prefix[k] = current_max_idx
            
        # Now, for each j_next, the best parent is arg_min_prefix[max_indices[j_next]]
        # But wait, we need to ensure the sum falls in the range.
        # The DP state (L, U) represents the *range* of possible sums.
        # If we pick the parent that minimizes the sum, the new sum is L[i+1].
        # If we pick the parent that maximizes the sum, the new sum is U[i+1].
        # For backtracking, we need to be able to reach N/2.
        # We can store the parent for the "mid" point or just store the range and backtrack greedily?
        # Let's store the parent index for the minimum sum path.
        # If N/2 is in [L[N-1], U[N-1]], we can backtrack by checking which parent allows N/2 to be in the range.
        
        # Let's just store the parent that achieves the minimum sum for now.
        # And also the parent that achieves the maximum sum?
        # Actually, we can store the parent for the "middle" of the range?
        # Let's store the parent index for the minimum sum.
        # parent[i+1, j_next] = arg_min_prefix[max_indices[j_next]]
        
        L[i+1, :] = min_sums + heights
        U[i+1, :] = max_sums + heights
        parent[i+1, :] = arg_min_prefix[max_indices]
        
    # Check final step
    # We need h_{N-1} <= M (constraint with h_N=0)
    # And sum in [L[N-1], U[N-1]] must contain N/2.
    
    target_sum = N / 2.0
    valid_mask = (heights <= M) & (L[N-1, :] <= target_sum) & (U[N-1, :] >= target_sum)
    
    if not np.any(valid_mask):
        return None
        
    # Backtrack
    h = np.zeros(N)
    j_curr = np.where(valid_mask)[0][0] # Pick first valid
    
    # We need to backtrack from N-1 to 0.
    # At step i, we have j_curr. The sum accumulated so far is S_i.
    # We want to choose j_prev such that S_{i-1} + h_{j_curr} = S_i?
    # No, S_i is the sum up to i.
    # We know S_{N-1} = target_sum.
    # S_{N-2} must be such that S_{N-2} + h_{N-1} = target_sum?
    # No, the sum is the sum of all h's.
    # S_i = sum_{k=0}^i h_k.
    # We need to find a path such that the final sum is target_sum.
    # Our DP stored ranges.
    # To backtrack, we start with j_{N-1}.
    # We need to find j_{N-2} such that:
    # 1. Transition j_{N-2} -> j_{N-1} is valid (h_{N-2}(1-h_{N-1}) <= M)
    # 2. The remaining sum S_{N-2} is in [L[N-2, j_{N-2}], U[N-2, j_{N-2}]]
    #    AND S_{N-2} = target_sum - h_{N-1}
    
    # So, for i = N-2 down to 0:
    #   target_prev_sum = target_sum - sum_{k=i+1}^{N-1} h_k
    #   We need to find j_prev such that:
    #   - parent[i+1, j_curr] == j_prev? No, parent was just one choice.
    #   - We need ANY j_prev such that:
    #     - h_{j_prev}(1-h_{j_curr}) <= M
    #     - L[i, j_prev] <= target_prev_sum <= U[i, j_prev]
    
    # Let's implement this backtracking.
    
    # First, determine the heights for the path.
    # We start with j_{N-1}.
    # We need to determine h_{N-2}, ..., h_0.
    
    # We'll maintain a running sum from the end.
    sum_so_far = h[j_curr] # This is h_{N-1}
    h[N-1] = heights[j_curr]
    
    for i in range(N - 2, -1, -1):
        target_prev_sum = target_sum - sum_so_far
        
        # Find valid j_prev
        # Condition 1: h_{j_prev}(1 - h_{j_curr}) <= M
        # Condition 2: L[i, j_prev] <= target_prev_sum <= U[i, j_prev]
        
        one_minus_h_curr = 1.0 - heights[j_curr]
        if one_minus_h_curr == 0:
            max_h_prev = np.inf
        else:
            max_h_prev = M / one_minus_h_curr
            
        # Find indices where heights <= max_h_prev
        max_idx = np.searchsorted(heights, max_h_prev, side='right') - 1
        if max_idx < 0:
            return None # No valid parent
            
        # Check range constraint
        # We need L[i, j] <= target_prev_sum <= U[i, j] for some j <= max_idx
        candidates = np.arange(max_idx + 1)
        valid_candidates = candidates[(L[i, candidates] <= target_prev_sum) & (U[i, candidates] >= target_prev_sum)]
        
        if len(valid_candidates) == 0:
            return None # Dead end
            
        # Pick one (e.g., the first one)
        j_prev = valid_candidates[0]
        h[i] = heights[j_prev]
        sum_so_far += heights[j_prev]
        j_curr = j_prev
        
    return h

def construct():
    deadline = time.time() + 140.0  # Safety margin
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Binary search for M
    low = 0.0
    high = 1.0
    
    # Initial guess from simple patterns
    h_init, s_init = _try_ramp_pattern(deadline)
    if h_init is not None:
        best_h = h_init
        best_score = s_init
        high = best_score
        
    # Refine with DP
    # We want to find the smallest M such that _dp_check(M) is not None.
    # Let's do binary search.
    
    # Ensure high is feasible
    while _dp_check(high) is None:
        high = min(1.0, high * 1.5)
        if high >= 1.0:
            break
            
    # Binary search
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        mid = (low + high) / 2.0
        h_mid = _dp_check(mid)
        if h_mid is not None:
            # Check score
            score = _overlap_max(h_mid)
            if score < best_score:
                best_score = score
                best_h = h_mid
            high = mid
        else:
            low = mid
            
    # Final polish with optimization?
    # The DP solution is discrete. We can try to optimize it continuously.
    if time.time() < deadline - 5:
        # Use the DP solution as a start for SLSQP
        h_opt = _optimize_from_start(best_h, 1000.0, max_iter=200)
        h_opt = _enforce_constraints(h_opt)
        score_opt = _overlap_max(h_opt)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
            
    # Ensure exact sum
    best_h = _enforce_constraints(best_h)
    
    return best_h
