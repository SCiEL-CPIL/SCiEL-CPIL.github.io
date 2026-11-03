"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_overlap(h, shift):
    """
    Compute the overlap of h with its shifted version by 'shift' positions.
    The overlap is defined as sum_i h[i] * h[(i + shift) % N].
    We want to minimize the maximum overlap over all non-zero shifts.
    """
    N = len(h)
    max_overlap = 0.0
    for s in range(1, N):
        shifted = np.roll(h, -s)
        overlap = np.dot(h, shifted)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _compute_all_overlaps(h):
    """Compute all non-zero shift overlaps."""
    N = len(h)
    overlaps = []
    for s in range(1, N):
        shifted = np.roll(h, -s)
        overlaps.append(np.dot(h, shifted))
    return np.array(overlaps)

def _objective(x, N):
    """
    Objective function: minimize the maximum overlap of h with shifted 1-h.
    h is the step heights, sum(h) = N/2.
    Overlap of h with shifted (1-h) = sum_i h[i] * (1 - h[(i+s)%N])
    = sum_i h[i] - sum_i h[i]*h[(i+s)%N]
    = N/2 - autocorrelation_shift_s
    
    So we want to maximize the minimum autocorrelation over non-zero shifts.
    Equivalently, minimize the maximum of (N/2 - autocorrelation_s).
    
    Wait, let me re-read the task. "minimize the maximum overlap of h with shifted 1-h"
    
    overlap_s = sum_i h[i] * (1 - h[(i+s)%N]) = N/2 - sum_i h[i]h[(i+s)%N]
    
    We want to minimize max_s overlap_s.
    Since N/2 is constant, this is equivalent to maximizing min_s (sum_i h[i]h[(i+s)%N]).
    
    Actually, let's just directly compute it.
    """
    h = x
    N = len(h)
    max_overlap = 0.0
    for s in range(1, N):
        shifted = np.roll(h, -s)
        one_minus_shifted = 1.0 - shifted
        overlap = np.dot(h, one_minus_shifted)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _objective_smooth(x, N, temperature=0.01):
    """
    Smooth approximation using log-sum-exp for the max.
    """
    h = x
    N = len(h)
    overlaps = np.zeros(N-1)
    for s in range(1, N):
        shifted = np.roll(h, -s)
        one_minus_shifted = 1.0 - shifted
        overlaps[s-1] = np.dot(h, one_minus_shifted)
    
    # log-sum-exp trick for smooth max
    m = np.max(overlaps)
    if temperature <= 0:
        return m
    log_sum = np.log(np.sum(np.exp((overlaps - m) / temperature)))
    return m + temperature * log_sum

def _gradient(x, N, temperature=0.01):
    """
    Gradient of the smooth objective.
    """
    h = x
    N = len(h)
    overlaps = np.zeros(N-1)
    for s in range(1, N):
        shifted = np.roll(h, -s)
        one_minus_shifted = 1.0 - shifted
        overlaps[s-1] = np.dot(h, one_minus_shifted)
    
    m = np.max(overlaps)
    if temperature <= 0:
        # Subgradient: pick the argmax
        idx = np.argmax(overlaps)
        s = idx + 1
        shifted = np.roll(h, -s)
        one_minus_shifted = 1.0 - shifted
        # d/dh_i [sum_j h[j] * (1 - h[(j+s)%N])]
        # = sum_j delta_{j,i} (1 - h[(j+s)%N]) + h[j] * (-delta_{(j+s)%N, i})
        # = (1 - h[(i+s)%N]) - sum_j h[j] delta_{(j+s)%N, i}
        # = (1 - h[(i+s)%N]) - h[(i-s)%N]
        grad = np.zeros(N)
        for i in range(N):
            grad[i] = (1 - h[(i+s) % N]) - h[(i-s) % N]
        return grad
    
    weights = np.exp((overlaps - m) / temperature) / np.sum(np.exp((overlaps - m) / temperature))
    
    grad = np.zeros(N)
    for s in range(1, N):
        w = weights[s-1]
        for i in range(N):
            grad[i] += w * ((1 - h[(i+s) % N]) - h[(i-s) % N])
    
    return grad / temperature  # Wait, this might not be right. Let me think.

def _objective_and_grad(x, N, temperature=0.01):
    """
    Compute objective and gradient together for efficiency.
    """
    h = x
    N = len(h)
    overlaps = np.zeros(N-1)
    for s in range(1, N):
        shifted = np.roll(h, -s)
        one_minus_shifted = 1.0 - shifted
        overlaps[s-1] = np.dot(h, one_minus_shifted)
    
    m = np.max(overlaps)
    if temperature <= 0:
        idx = np.argmax(overlaps)
        s = idx + 1
        grad = np.zeros(N)
        for i in range(N):
            grad[i] = (1 - h[(i+s) % N]) - h[(i-s) % N]
        return m, grad
    
    exp_vals = np.exp((overlaps - m) / temperature)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    obj = m + temperature * np.log(sum_exp)
    
    grad = np.zeros(N)
    for s in range(1, N):
        w = weights[s-1]
        if w > 1e-10:
            for i in range(N):
                grad[i] += w * ((1 - h[(i+s) % N]) - h[(i-s) % N])
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial guess: alternating 0 and 1 pattern or 0.5 everywhere
    # Let's try a few initializations
    best_h = None
    best_score = np.inf
    
    # Try several random initializations
    np.random.seed(42)
    
    n_trials = 20
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if trial == 0:
            # Uniform
            h0 = np.full(N, 0.5)
        elif trial == 1:
            # Alternating
            h0 = np.zeros(N)
            h0[::2] = 1.0
        elif trial == 2:
            # Random binary with correct sum
            h0 = np.zeros(N)
            idx = np.random.choice(N, N//2, replace=False)
            h0[idx] = 1.0
        elif trial == 3:
            # Random continuous
            h0 = np.random.uniform(0, 1, N)
            # Adjust to have correct sum
            target_sum = N / 2.0
            current_sum = np.sum(h0)
            if current_sum > 0:
                h0 = h0 * (target_sum / current_sum)
                h0 = np.clip(h0, 0, 1)
        else:
            # Perturbed random
            h0 = np.random.uniform(0, 1, N)
            target_sum = N / 2.0
            current_sum = np.sum(h0)
            if current_sum > 0:
                h0 = h0 * (target_sum / current_sum)
                h0 = np.clip(h0, 0, 1)
        
        # Use SLSQP with bounds
        bounds = [(0, 1)] * N
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0, 'jac': lambda x: np.ones(N)}
        ]
        
        temperature = 0.05
        result = minimize(
            _objective_smooth,
            h0,
            args=(N, temperature),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        if result.success and np.sum(result.x) >= N/2.0 - 1e-6:
            # Project back to feasible region
            h_opt = np.clip(result.x, 0, 1)
            # Ensure exact sum
            diff = N/2.0 - np.sum(h_opt)
            if abs(diff) > 1e-10:
                # Distribute the difference
                mask = (h_opt < 1.0) & (h_opt > 0.0)
                if np.sum(mask) > 0:
                    h_opt[mask] += diff / np.sum(mask)
                    h_opt = np.clip(h_opt, 0, 1)
            
            score = _compute_overlap(h_opt, 0)  # max overlap
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        
        # Also try with smaller temperature for refinement
        if time.time() < deadline - 5:
            temperature = 0.01
            result2 = minimize(
                _objective_smooth,
                h0,
                args=(N, temperature),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-9}
            )
            
            if result2.success:
                h_opt = np.clip(result2.x, 0, 1)
                diff = N/2.0 - np.sum(h_opt)
                if abs(diff) > 1e-10:
                    mask = (h_opt < 1.0) & (h_opt > 0.0)
                    if np.sum(mask) > 0:
                        h_opt[mask] += diff / np.sum(mask)
                        h_opt = np.clip(h_opt, 0, 1)
                
                score = _compute_overlap(h_opt, 0)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
    
    # Final polish: local search on the best solution
    if best_h is not None and time.time() < deadline - 10:
        # Try coordinate descent
        h = best_h.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_score = _compute_overlap(h, 0)
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                for delta in [0.01, 0.05, 0.1, -0.01, -0.05, -0.1]:
                    h_new = h.copy()
                    h_new[i] += delta
                    # Compensate by adjusting another coordinate
                    # Find a coordinate j where we can adjust
                    for j in range(N):
                        if j == i:
                            continue
                        h_new[j] -= delta
                        if 0 <= h_new[i] <= 1 and 0 <= h_new[j] <= 1:
                            new_score = _compute_overlap(h_new, 0)
                            if new_score < current_score - 1e-10:
                                h = h_new.copy()
                                current_score = new_score
                                improved = True
                                break
    
    # Ensure valid output
    if best_h is None:
        return np.full(N, 0.5)
    
    # Final validation and correction
    best_h = np.clip(best_h, 0, 1)
    diff = N/2.0 - np.sum(best_h)
    if abs(diff) > 1e-8:
        mask = (best_h < 1.0) & (best_h > 0.0)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
    
    return best_h

# EVOLVE-BLOCK-END
