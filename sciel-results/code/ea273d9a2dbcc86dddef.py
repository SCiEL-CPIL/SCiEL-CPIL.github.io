import time
import numpy as np
import random

def construct():
    n = 256
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    # 1. Precompute all 4-term APs
    # 0-indexed: a, a+d, a+2d, a+3d < n
    ap_a = []
    ap_b = []
    ap_c = []
    ap_d = []
    
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_a.append(a)
            ap_b.append(a + d)
            ap_c.append(a + 2 * d)
            ap_d.append(a + 3 * d)
            
    ap_a = np.array(ap_a, dtype=np.int32)
    ap_b = np.array(ap_b, dtype=np.int32)
    ap_c = np.array(ap_c, dtype=np.int32)
    ap_d = np.array(ap_d, dtype=np.int32)
    
    num_aps = len(ap_a)
    
    # 2. Initialization
    # Thue-Morse is a good start
    def thue_morse(i):
        return bin(i).count('1') % 2
        
    # x in {-1, 1}
    # color 0 -> 1, color 1 -> -1
    # x_i = 1 if color 0, -1 if color 1
    x = np.array([1 if thue_morse(i) == 0 else -1 for i in range(n)], dtype=np.float64)
    
    best_x = x.copy()
    
    def get_count(x_vec):
        # x_vec is in {-1, 1}
        # Mono AP if x_a * x_b * x_c * x_d == 1
        # Count = sum( (1 + prod)/2 )
        # We want to minimize sum(prod)
        # Let's just count directly for best tracking
        prods = x_vec[ap_a] * x_vec[ap_b] * x_vec[ap_c] * x_vec[ap_d]
        # prods are 1 or -1
        # Mono if prod == 1
        return np.sum(prods == 1)
        
    best_count = get_count(x)
    current_count = best_count
    
    rng = np.random.default_rng(42)
    
    # Precompute indices for efficient P calculation
    # P[i] = sum of S[k] for all k where i is in ap k
    # We can use np.add.at
    
    while time.time() < deadline:
        # Calculate S: product of x for each AP
        S = x[ap_a] * x[ap_b] * x[ap_c] * x[ap_d]
        
        # Calculate P: marginal effect sum
        P = np.zeros(n, dtype=np.float64)
        np.add.at(P, ap_a, S)
        np.add.at(P, ap_b, S)
        np.add.at(P, ap_c, S)
        np.add.at(P, ap_d, S)
        
        # Delta E if we flip bit i:
        # E_new = E - 2 * sum_{k containing i} S_k
        # So Delta_i = -2 * P_i
        # We want Delta < 0, i.e., P_i > 0
        # If P_i > 0, flipping x_i reduces the sum of products (and thus reduces mono count? 
        # Wait.
        # Mono count = (N_aps + Sum(S)) / 2.
        # Minimizing Mono count is equivalent to Minimizing Sum(S).
        # If P_i > 0, flipping i changes Sum(S) by -2 P_i, which is negative.
        # So Sum(S) decreases. Mono count decreases.
        
        flip_mask = P > 0
        
        if np.any(flip_mask):
            # Flip all beneficial bits
            x[flip_mask] *= -1
            current_count = get_count(x)
            
            if current_count < best_count:
                best_count = current_count
                best_x = x.copy()
        else:
            # Stuck in local minimum. Perturb.
            # Flip a random subset of bits to escape
            # Number of flips increases slightly over time or is random
            num_flips = rng.integers(1, 10)
            flip_indices = rng.choice(n, size=num_flips, replace=False)
            x[flip_indices] *= -1
            current_count = get_count(x)
            
            if current_count < best_count:
                best_count = current_count
                best_x = x.copy()
            
            # Occasionally, if stuck for a long time, do a bigger restart
            # But the random flips should handle it.
            # To be safe, if we don't improve for a while, we could re-seed from best
            # But simple random walks usually work with this aggressive descent.
            
    # Convert best_x back to 0/1 colors
    # x=1 -> 0, x=-1 -> 1
    colors = [0 if xi == 1 else 1 for xi in best_x]
    
    return colors

# EVOLVE-BLOCK-END
