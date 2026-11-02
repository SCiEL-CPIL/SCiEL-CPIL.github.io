"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def _compute_overlap(h, shift):
    """
    Compute the overlap integral of h with its shifted version (1-h).
    
    For step functions on [0, 2] with N steps of width 2/N:
    h_i is the height on [2i/N, 2(i+1)/N) for i=0..N-1
    (1-h) is the complement function.
    
    The shifted version (1-h)(x - shift) means we look at (1-h) evaluated at x-shift.
    So overlap = integral_0^2 h(x) * (1-h)(x - shift) dx
    
    With step functions, this becomes a sum over intervals.
    """
    step_width = 2.0 / N
    h_shifted = np.zeros(N)
    
    # For each step i of h, we need (1-h) at position shifted by 'shift'
    # The shift is in units of the domain [0,2], so in step units it's shift / step_width
    shift_steps = shift / step_width
    
    for i in range(N):
        # The i-th step of h covers [i*step_width, (i+1)*step_width)
        # We need (1-h) at positions x - shift for x in this interval
        # So we need (1-h) at [i*step_width - shift, (i+1)*step_width - shift)
        start_pos = i * step_width - shift
        end_pos = (i + 1) * step_width - shift
        
        # Integrate (1-h) over this interval
        integral = 0.0
        for j in range(N):
            # j-th step of (1-h) covers [j*step_width, (j+1)*step_width)
            # Overlap of [start_pos, end_pos) with [j*step_width, (j+1)*step_width)
            lo = max(start_pos, j * step_width)
            hi = min(end_pos, (j + 1) * step_width)
            if hi > lo:
                integral += (1.0 - h[j]) * (hi - lo)
        
        h_shifted[i] = integral / step_width  # average value of (1-h) over the shifted interval
    
    # The overlap is sum of h_i * (shifted 1-h)_i * step_width
    return np.sum(h * h_shifted) * step_width


def _max_overlap(h):
    """Compute the maximum overlap over all shifts in [0, 2]."""
    # Sample shifts
    num_shifts = 200
    shifts = np.linspace(0, 2, num_shifts)
    max_ov = 0.0
    for s in shifts:
        ov = _compute_overlap(h, s)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _evaluate(h):
    """Evaluate the score (max overlap) for a given h."""
    # Check constraints
    if not np.all(h >= -1e-9) or not np.all(h <= 1 + 1e-9):
        return 1e10
    if abs(np.sum(h) - N / 2) > 1e-6:
        return 1e10
    return _max_overlap(h)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initial solution: constant 0.5
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _evaluate(h)
    
    # Use subgradient method to minimize max overlap
    # The max overlap is the max over shifts of bilinear forms
    
    num_shifts = 100
    shifts = np.linspace(0.01, 1.99, num_shifts)
    
    # Precompute the overlap matrices for each shift
    # overlap(s) = h^T M_s h where M_s is a matrix
    # Actually, overlap(s) = sum_i h_i * (sum_j (1-h_j) * overlap_width(i, j, s))
    # = sum_i h_i * (step_width - sum_j h_j * overlap_width(i, j, s))
    # = sum_i h_i * step_width - sum_{i,j} h_i h_j * overlap_width(i, j, s)
    
    # So overlap(s) = (2/N) * sum_i h_i - sum_{i,j} h_i h_j * W_{ij}(s)
    # where W_{ij}(s) is the width of overlap between step i of h and step j of (1-h) shifted by s
    
    # Let's precompute W matrices
    W_matrices = np.zeros((num_shifts, N, N))
    step_width = 2.0 / N
    
    for si, s in enumerate(shifts):
        for i in range(N):
            for j in range(N):
                # Step i of h: [i*sw, (i+1)*sw)
                # Step j of (1-h) shifted by s: [j*sw + s, (j+1)*sw + s)
                lo = max(i * step_width, j * step_width + s)
                hi = min((i + 1) * step_width, (j + 1) * step_width + s)
                if hi > lo:
                    W_matrices[si, i, j] = hi - lo
    
    # Now overlap(s) = 2/N * sum(h) - sum_{i,j} h_i h_j W_{ij}(s)
    # = 1.0 - sum_{i,j} h_i h_j W_{ij}(s)  (since sum(h) = N/2, so 2/N * N/2 = 1)
    
    def overlap_vector(h_vec):
        """Compute overlaps for all shifts."""
        overlaps = np.ones(num_shifts)
        for si in range(num_shifts):
            W = W_matrices[si]
            # sum_{i,j} h_i h_j W_{ij}
            val = 0.0
            for i in range(N):
                if h_vec[i] != 0:
                    val += h_vec[i] * np.dot(h_vec, W[i])
            overlaps[si] = 1.0 - val
        return overlaps
    
    # Subgradient descent on max(overlaps)
    # The subgradient of max(f_1, ..., f_m) at point x is in the convex hull of gradients of active functions
    # For our case, we want to minimize max_s overlap(s)
    # gradient of overlap(s) w.r.t. h_i = -2 * sum_j h_j W_{ij}(s) (for i != j terms) 
    # Actually, d/dh_i [sum_{k,l} h_k h_l W_{kl}] = 2 * sum_l h_l W_{il} (symmetric)
    # So d overlap(s)/dh_i = -2 * sum_l h_l W_{il}(s)
    
    lr = 0.05
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        ov = overlap_vector(h)
        max_ov = np.max(ov)
        active = (ov >= max_ov - 1e-9)
        
        if max_ov < best_score:
            best_score = max_ov
            best_h = h.copy()
        
        # Compute subgradient: average gradient over active shifts
        grad = np.zeros(N)
        for si in range(num_shifts):
            if active[si]:
                W = W_matrices[si]
                # d overlap(s)/dh_i = -2 * sum_l h_l W_{il}(s)
                grad -= 2.0 * W @ h
        
        # Normalize
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        # Update
        h = h - lr * grad
        
        # Project onto feasible set: 0 <= h_i <= 1, sum(h) = N/2
        # Simple projection: clip and then adjust for sum
        h = np.clip(h, 0, 1)
        # Adjust for sum constraint
        diff = N / 2 - np.sum(h)
        if diff > 0:
            # Need to increase sum; add to elements with room
            mask = h < 1.0
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
        elif diff < 0:
            # Need to decrease sum
            mask = h > 0.0
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
        
        h = np.clip(h, 0, 1)
        
        # Decay learning rate
        lr *= 0.995
    
    # Final evaluation
    final_score = _evaluate(best_h)
    if final_score < best_score:
        best_score = final_score
    
    return best_h

# EVOLVE-BLOCK-END
