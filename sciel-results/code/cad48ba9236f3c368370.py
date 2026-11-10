# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_ratio(P):
    """
    Compute the D/M ratio for a given 10x10 distribution P.
    P[y, x] is the probability.
    Returns 0 if invalid.
    """
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-9:
        return 0.0
    P = P / s
    
    # Grid indices
    # P is (10, 10). y is row, x is col.
    y_idx, x_idx = np.indices((10, 10))
    
    # Flatten
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w_flat = P.flatten()
    
    # Values for entropy calculations
    # D = H(x - y)
    v_d = x_flat - y_flat
    
    # M components
    v_m0 = x_flat           # H(x)
    v_m1 = x_flat + y_flat  # H(x + y)
    v_m2 = x_flat + 2 * y_flat # H(x + 2y)
    
    def entropy_from_values(values, weights):
        # Use bincount for speed
        # Values are integers.
        # Determine range
        min_v = np.min(values)
        max_v = np.max(values)
        # Shift to non-negative
        shifted = values - min_v
        # Sum weights for each value
        # np.bincount requires non-negative integers
        # Max value range is small (<= 18 or 27)
        counts = np.bincount(shifted, weights=weights, minlength=max_v - min_v + 1)
        # Filter zero counts
        probs = counts[counts > 1e-12]
        if len(probs) == 0:
            return 0.0
        # Normalize just in case (should sum to 1)
        probs = probs / probs.sum()
        # Shannon entropy in bits
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_from_values(v_d, w_flat)
    M0 = entropy_from_values(v_m0, w_flat)
    M1 = entropy_from_values(v_m1, w_flat)
    M2 = entropy_from_values(v_m2, w_flat)
    
    M = max(M0, M1, M2)
    
    if M < 1e-9:
        return 0.0
        
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation).
    """
    n = 10
    start_time = time.time()
    deadline = 600.0 # seconds
    
    # Initial guess: Uniform
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Precompute indices for speed
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    
    v_d = x_flat - y_flat
    v_m0 = x_flat
    v_m1 = x_flat + y_flat
    v_m2 = x_flat + 2 * y_flat
    
    # Precompute min/max for bincount to avoid recalculating?
    # Values are fixed integers.
    d_min, d_max = np.min(v_d), np.max(v_d)
    m0_min, m0_max = np.min(v_m0), np.max(v_m0)
    m1_min, m1_max = np.min(v_m1), np.max(v_m1)
    m2_min, m2_max = np.min(v_m2), np.max(v_m2)
    
    def fast_entropy(values, weights, v_min, v_max):
        shifted = values - v_min
        counts = np.bincount(shifted, weights=weights, minlength=v_max - v_min + 1)
        probs = counts[counts > 1e-12]
        if len(probs) == 0:
            return 0.0
        probs = probs / probs.sum()
        return -np.sum(probs * np.log2(probs))
    
    def eval_ratio(P_flat):
        P_flat = np.clip(P_flat, 0, None)
        s = P_flat.sum()
        if s < 1e-9:
            return 0.0
        P_norm = P_flat / s
        
        D = fast_entropy(v_d, P_norm, d_min, d_max)
        M0 = fast_entropy(v_m0, P_norm, m0_min, m0_max)
        M1 = fast_entropy(v_m1, P_norm, m1_min, m1_max)
        M2 = fast_entropy(v_m2, P_norm, m2_min, m2_max)
        
        M = max(M0, M1, M2)
        if M < 1e-9:
            return 0.0
        return D / M

    best_P = P.copy()
    best_score = eval_ratio(best_P.flatten())
    
    # Simulated Annealing
    # We will perturb P by moving mass between cells.
    # To maintain sum=1, we pick two cells i, j and move delta from i to j.
    # Or pick a cell and redistribute.
    
    T = 0.1
    T_min = 0.001
    cooling = 0.999
    steps = 0
    max_steps = 500000 # Adjust based on time
    
    # Initialize with some structure? Uniform is safe.
    # Let's try starting with a "cross" or diagonal to jump out of uniform local optimum?
    # Uniform is a saddle point or local max?
    # Score 1.0 for uniform.
    # Let's initialize with a distribution concentrated on the main diagonal?
    # P[i, i] = 1/10.
    # This gives D=0. Score 0.
    # Let's initialize with a distribution on the anti-diagonal?
    # P[i, 9-i] = 1/10.
    # X+Y=9. H(U)=0.
    # X varies 9..0. H(X)=log2(10).
    # X-Y varies.
    # This gave score 1.0.
    
    # Let's start with uniform and let SA explore.
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < deadline - 5.0:
        steps += 1
        if steps % 10000 == 0:
            # Check time
            if time.time() - start_time > deadline - 5.0:
                break
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Generate a move
        # Pick two distinct cells
        i = rng.integers(0, 100)
        j = rng.integers(0, 100)
        if i == j:
            continue
            
        # Amount to move
        # Scale with T?
        # Max possible move is P[i]
        max_move = best_P.flatten()[i]
        if max_move < 1e-9:
            continue
            
        # Random delta
        # delta is uniform in [-max_move, max_move]
        delta = rng.uniform(-max_move, max_move)
        
        new_P_flat = best_P.flatten().copy()
        new_P_flat[i] -= delta
        new_P_flat[j] += delta
        
        # Clip negative
        if new_P_flat[i] < 0 or new_P_flat[j] < 0:
            continue
            
        # Normalize?
        # The move preserves sum, so no renormalization needed if we don't clip.
        # But we clipped? No, we checked bounds.
        # However, floating point errors might accumulate.
        # Let's renormalize to be safe.
        s = new_P_flat.sum()
        if s < 1e-9:
            continue
        new_P_flat = new_P_flat / s
        
        new_score = eval_ratio(new_P_flat)
        
        if new_score > best_score:
            best_P_flat = new_P_flat
            best_score = new_score
            best_P = new_P_flat.reshape((10, 10))
        else:
            # Accept with probability exp((new_score - best_score) / T)
            # Note: scores can be 0. If best is 1.0 and new is 0.9, delta is -0.1.
            # exp(-0.1/0.1) = exp(-1) ~ 0.37.
            delta_score = new_score - best_score
            if delta_score < 0:
                prob = np.exp(delta_score / T)
                if rng.random() < prob:
                    best_P_flat = new_P_flat
                    best_score = new_score
                    best_P = new_P_flat.reshape((10, 10))
            # If new_score > best_score, it was already handled above.
            
    # Final check
    final_P = np.clip(best_P, 0, None)
    s = final_P.sum()
    if s > 1e-9:
        final_P = final_P / s
    else:
        final_P = np.ones((10, 10)) / 100.0
        
    return final_P

# EVOLVE-BLOCK-END
