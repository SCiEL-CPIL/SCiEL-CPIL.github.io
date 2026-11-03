"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlap_max(h, shifts):
    """Compute the maximum overlap of h with shifted (1-h) over given shifts.
    Overlap for shift k: sum_i h[i] * (1-h[(i+k) % N])
    We want to minimize the maximum over k in shifts."""
    N = len(h)
    one_minus_h = 1.0 - h
    max_val = 0.0
    for k in shifts:
        shifted = np.roll(one_minus_h, k)
        ov = np.dot(h, shifted)
        if ov > max_val:
            max_val = ov
    return max_val

def _smooth_max_overlap(h, shifts, beta=20.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    N = len(h)
    one_minus_h = 1.0 - h
    vals = []
    for k in shifts:
        shifted = np.roll(one_minus_h, k)
        vals.append(np.dot(h, shifted))
    vals = np.array(vals)
    m = np.max(vals)
    if beta == 0:
        return m
    return m + np.log(np.mean(np.exp(beta * (vals - m)))) / beta

def _grad_smooth_max_overlap(h, shifts, beta=20.0):
    """Gradient of smooth max overlap."""
    N = len(h)
    one_minus_h = 1.0 - h
    vals = np.zeros(len(shifts))
    for i, k in enumerate(shifts):
        shifted = np.roll(one_minus_h, k)
        vals[i] = np.dot(h, shifted)
    
    m = np.max(vals)
    weights = np.exp(beta * (vals - m))
    weights /= np.sum(weights)
    
    # Gradient: d/dh sum_k w_k * dot(h, roll(1-h, k))
    # = sum_k w_k * (roll(1-h, k) - roll(h, -k))  ... let's derive carefully
    # f_k(h) = sum_i h[i] * (1 - h[(i+k)%N])
    # df_k/dh[j] = (1 - h[(j+k)%N]) - h[(j-k)%N]
    grad = np.zeros(N)
    for i, k in enumerate(shifts):
        # term for each j: (1 - h[(j+k)%N]) - h[(j-k)%N]
        term = (1.0 - np.roll(h, k)) - np.roll(h, -k)
        grad += weights[i] * term
    return grad

def _project_to_simplex(v, target_sum):
    """Project v onto the simplex {x >= 0, sum(x) = target_sum}."""
    n = len(v)
    # We need sum = target_sum, x >= 0, x <= 1
    # First project to sum = target_sum with x >= 0
    v = np.sort(v)[::-1]
    cssv = np.cumsum(v)
    rho = np.nonzero(v * np.arange(1, n + 1) > (cssv - target_sum))[0][-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    result = np.maximum(v - theta, 0)
    # Now clip to [0, 1] and renormalize if needed
    result = np.clip(result, 0, 1)
    s = np.sum(result)
    if s > 0:
        result = result * (target_sum / s)
        result = np.clip(result, 0, 1)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # We want to minimize max over shifts k of sum_i h[i]*(1-h[(i+k)%N])
    # The shift k=0 gives sum_i h[i]*(1-h[i]) = sum h[i] - sum h[i]^2 = N/2 - sum h[i]^2
    # To minimize this, we want to maximize sum h[i]^2, i.e., make h as binary as possible.
    # But we also care about other shifts.
    
    # Key insight: if h is binary (0 or 1) with exactly N/2 ones, then:
    # Overlap at shift k = number of i where h[i]=1 and h[(i+k)%N]=0
    # = N/2 - (number of i where h[i]=1 and h[(i+k)%N]=1)
    # = N/2 - autocorrelation at shift k
    # So minimizing max overlap = N/2 - min_k autocorrelation(k) for k != 0
    # Wait, we want to minimize the max, so we want to minimize (N/2 - min_autocorr)
    # i.e., maximize the minimum autocorrelation over non-zero shifts.
    
    # Actually let me re-examine: overlap_k = sum_i h[i]*(1-h[(i+k)%N])
    # If h is binary with N/2 ones: overlap_k = N/2 - C(k) where C(k) = sum_i h[i]*h[(i+k)%N]
    # So max_k overlap_k = N/2 - min_k C(k)
    # To minimize max_k overlap_k, we want to maximize min_k C(k).
    
    # For random binary sequences with N/2 ones, C(k) for k!=0 is approximately N/4 with variance ~N/16.
    # The minimum over N-1 shifts will be around N/4 - c*sqrt(N) for some c.
    # So max overlap ~ N/2 - N/4 + c*sqrt(N) = N/4 + c*sqrt(N).
    
    # For N=100, N/4 = 25. The baseline uniform h=0.5 gives overlap_k = sum 0.5*0.5 = N*0.25 = 25 for all k.
    # So binary sequences might do better if min autocorrelation is > N/4.
    
    # Let's try to find a binary sequence with high minimum autocorrelation.
    # Use simulated annealing / local search on binary sequences.
    
    best_h = None
    best_score = np.inf
    
    def _score_binary(h):
        """Compute max overlap for a binary h."""
        N = len(h)
        max_ov = 0.0
        for k in range(1, N):
            shifted = np.roll(h, k)
            ov = np.sum(h * (1 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    def _score_general(h):
        """Compute max overlap for general h in [0,1]."""
        N = len(h)
        max_ov = 0.0
        for k in range(1, N):
            shifted = np.roll(1.0 - h, k)
            ov = np.dot(h, shifted)
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    # Start with binary local search
    np.random.seed(42)
    
    # Generate diverse initial binary sequences
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random binary sequence with exactly N/2 ones
        h = np.zeros(N)
        ones = np.random.choice(N, N//2, replace=False)
        h[ones] = 1.0
        
        current_score = _score_binary(h)
        
        # Local search: flip pairs of bits to maintain sum constraint
        # Or flip a 1 to 0 and a 0 to 1
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            # Try all possible swaps
            ones_idx = np.where(h == 1)[0]
            zeros_idx = np.where(h == 0)[0]
            if len(ones_idx) == 0 or len(zeros_idx) == 0:
                break
            
            # Randomly sample swaps for speed
            num_trials = min(200, len(ones_idx) * len(zeros_idx))
            for _ in range(num_trials):
                i = ones_idx[np.random.randint(len(ones_idx))]
                j = zeros_idx[np.random.randint(len(zeros_idx))]
                if i == j:
                    continue
                h[i] = 0
                h[j] = 1
                new_score = _score_binary(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
                else:
                    h[i] = 1
                    h[j] = 0
                if time.time() > deadline - 10:
                    break
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
    
    if best_h is None:
        return np.full(N, 0.5)
    
    # Now polish with continuous optimization
    # Use projected gradient descent on the smooth surrogate
    h = best_h.copy()
    
    # Use all shifts 1..N-1
    shifts = list(range(1, N))
    
    # Beta schedule for log-sum-exp
    betas = [10, 50, 100, 200, 500]
    lr_schedule = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for beta, lr in zip(betas, lr_schedule):
        if time.time() > deadline - 15:
            break
        for it in range(50):
            if time.time() > deadline - 15:
                break
            grad = _grad_smooth_max_overlap(h, shifts, beta)
            h = h - lr * grad
            h = _project_to_simplex(h, N / 2.0)
    
    # Final polish with exact max
    for it in range(100):
        if time.time() > deadline - 5:
            break
        # Subgradient of max overlap
        max_ov = 0.0
        best_k = 0
        for k in range(1, N):
            shifted = np.roll(1.0 - h, k)
            ov = np.dot(h, shifted)
            if ov > max_ov:
                max_ov = ov
                best_k = k
        
        shifted = np.roll(1.0 - h, best_k)
        # Gradient of dot(h, shifted) w.r.t. h: shifted - roll(h, -best_k)
        grad = shifted - np.roll(h, -best_k)
        h = h - 0.01 * grad
        h = _project_to_simplex(h, N / 2.0)
    
    final_score = _score_general(h)
    if final_score < best_score:
        best_score = final_score
        best_h = h
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N / 2.0 / s)
        best_h = np.clip(best_h, 0, 1)
    
    # Final correction for exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-10:
        # Distribute the difference
        n_nonzero = np.sum(best_h > 0)
        if n_nonzero > 0:
            best_h[best_h > 0] += diff / n_nonzero
            best_h = np.clip(best_h, 0, 1)
            # Re-correct
            diff2 = N / 2.0 - np.sum(best_h)
            if abs(diff2) > 1e-10:
                idx = np.argsort(best_h)[::-1]
                for i in idx:
                    if best_h[i] + diff2 > 0 and best_h[i] + diff2 < 1:
                        best_h[i] += diff2
                        break
    
    return best_h
# EVOLVE-BLOCK-END
