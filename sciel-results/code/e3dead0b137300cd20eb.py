"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Use a fixed N that is manageable for FISTA
    # N=128 is a good balance
    N = 128
    
    # Precompute C and d for this N
    s_candidates = set()
    for k in range(N + 1):
        s_candidates.add(k / N)
        s_candidates.add(1.0 + k / N)
    s_candidates.add(0.0)
    s_candidates.add(1.0)
    s_candidates.add(2.0)
    
    s_vals = sorted([s for s in s_candidates if 0.0 <= s <= 2.0])
    num_shifts = len(s_vals)
    
    C = np.zeros((num_shifts, N))
    d = np.zeros(num_shifts)
    
    for j, s in enumerate(s_vals):
        if s <= 1.0:
            for i in range(N):
                step_start = 2.0 * i / N
                step_end = 2.0 * (i + 1) / N
                lo1 = max(0.0, step_start)
                hi1 = min(1.0 - s, step_end)
                m1 = max(0.0, hi1 - lo1)
                lo2 = max(1.0 + s, step_start)
                hi2 = min(2.0, step_end)
                m2 = max(0.0, hi2 - lo2)
                C[j, i] = m1 - m2
                d[j] += m2
        else:
            for i in range(N):
                step_start = 2.0 * i / N
                step_end = 2.0 * (i + 1) / N
                lo1 = max(0.0, step_start)
                hi1 = min(s - 1.0, step_end)
                m1 = max(0.0, hi1 - lo1)
                lo2 = max(2.0 - s, step_start)
                hi2 = min(2.0, step_end)
                m2 = max(0.0, hi2 - lo2)
                C[j, i] = -m1 + m2
                d[j] += m1
    
    # Objective: Minimize max_j (C_j h + d_j)
    # We use a smooth approximation: logsumexp(beta * (C h + d)) / beta
    # And FISTA to minimize it subject to constraints.
    
    beta = 50.0
    
    def compute_grad(h):
        O = C @ h + d
        max_O = np.max(O)
        exp_O = np.exp(beta * (O - max_O))
        sum_exp = np.sum(exp_O)
        probs = exp_O / sum_exp
        grad = probs @ C
        return grad, O
    
    def project(h):
        # Project h onto {x | 0 <= x_i <= 1, sum(x) = N/2}
        # 1. Clip to [0,1]
        h = np.clip(h, 0, 1)
        # 2. Adjust to meet sum constraint
        # We need to add a constant c to h, then clip again, iterating until sum is correct.
        # This is water-filling.
        target_sum = N / 2.0
        for _ in range(20):
            current_sum = np.sum(h)
            diff = target_sum - current_sum
            if abs(diff) < 1e-9:
                break
            h = h + diff / N
            h = np.clip(h, 0, 1)
        return h
    
    # Initialize
    h_curr = np.full(N, 0.5)
    h_prev = h_curr.copy()
    t_k = 1.0
    best_h = h_curr.copy()
    best_val = np.max(C @ h_curr + d)
    
    # FISTA Loop
    # Lipschitz constant L for the smooth part.
    # grad = softmax @ C. Norm of C columns is bounded.
    # Let's estimate L. ||C||_F^2 is a safe upper bound for the Lipschitz constant of the gradient of logsumexp?
    # Actually, the Hessian of logsumexp is positive semi-definite with eigenvalues <= 1 (for standard logsumexp).
    # Here we have scaled by beta. So Hessian eigenvalues <= beta * 1? No.
    # f(h) = 1/beta log sum exp(beta (Ch+d)).
    # grad = C^T softmax(beta(Ch+d)).
    # Hessian = C^T diag(softmax) C.
    # Eigenvalues of diag(softmax) are <= 1.
    # So eigenvalues of Hessian <= lambda_max(C^T C).
    # Let's compute L = ||C||_2^2.
    L = np.linalg.norm(C, 2)**2
    if L == 0: L = 1.0
    step_size = 1.0 / L
    
    max_iter = 2000
    
    for k in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        grad, O = compute_grad(h_curr)
        
        # Gradient step
        y = h_curr - step_size * grad
        
        # Projection
        h_next = project(y)
        
        # FISTA momentum
        t_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        h_curr = h_next + ((t_k - 1) / t_next) * (h_next - h_prev)
        h_curr = np.clip(h_curr, 0, 1) # Ensure we stay in bounds for next iteration's gradient
        
        # Evaluate
        val = np.max(C @ h_next + d)
        if val < best_val:
            best_val = val
            best_h = h_next.copy()
            
        h_prev = h_curr
        t_k = t_next
        
    # Final polish: ensure exact sum
    diff = N/2.0 - np.sum(best_h)
    if abs(diff) > 1e-10:
        best_h = project(best_h)
        
    return best_h

# EVOLVE-BLOCK-END
