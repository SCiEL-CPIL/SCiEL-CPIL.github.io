"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x8 matrix.
    Uses a structured Strassen-style approach to reduce rank below schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    nm, mp, pn = n*m, m*p, p*n  # 4, 16, 16
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 32
    U_s = [[0] * rank_school for _ in range(nm)]
    V_s = [[0] * rank_school for _ in range(mp)]
    W_s = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    def verify_fast(U, V, W, rank):
        """Fast verification using numpy."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check valid triples
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    prod = U_arr[a] * V_arr[b] * W_arr[c]
                    if abs(prod.sum() - 1.0) > 1e-9:
                        return False
        
        # Check invalid triples
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    for c in range(pn):
                        prod = U_arr[a] * V_arr[b] * W_arr[c]
                        if abs(prod.sum()) > 1e-9:
                            return False
                else:
                    c_expected = k_b * n + i_a
                    for c in range(pn):
                        if c != c_expected:
                            prod = U_arr[a] * V_arr[b] * W_arr[c]
                            if abs(prod.sum()) > 1e-9:
                                return False
        return True
    
    # Try Strassen-style decomposition for rank 24
    # The idea: use 7 terms per 2x2 block multiplication, but adapt for 2x8
    # For M_{2,2,8}, we can potentially use rank 24 by combining Strassen with the 8 columns
    
    # Strategy: For each pair of k values, use Strassen's 7-term algorithm
    # on the 2x2 block, then combine across the 4 pairs of k values
    # This gives rank 7 * 4 = 28, but we can do better by sharing
    
    # Actually, let's try a direct approach:
    # The tensor M_{2,2,8} can be viewed as 8 copies of M_{2,2,1}
    # M_{2,2,1} has rank 7 (Strassen)
    # But we can share terms across the 8 copies
    
    # Let's try rank 24 with a structured pattern
    # We'll use the fact that the 8 output columns can be grouped
    
    # Approach: Use 6 terms per "block" of 2 k-values, giving 6*4 = 24
    # This requires careful construction
    
    # For now, try random search with structured sparsity for ranks 24-31
    random.seed(42)
    
    # Use a smarter initialization: start from schoolbook and try to merge terms
    for target_rank in range(24, min(32, best_rank)):
        if time.time() > deadline:
            break
        
        attempts = 0
        max_attempts = 2000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate structured random U, V, W
            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]
            W = [[0] * target_rank for _ in range(pn)]
            
            # Use a pattern based on grouping k values
            # Group k into 4 pairs: (0,1), (2,3), (4,5), (6,7)
            # For each pair, use 6 terms (instead of 8)
            
            # Initialize with a base pattern
            base_rank = 6
            for pair_idx in range(4):
                k0 = pair_idx * 2
                k1 = pair_idx * 2 + 1
                for term in range(base_rank):
                    r = pair_idx * base_rank + term
                    if r >= target_rank:
                        break
                    
                    # Random coefficients from a small set
                    coeffs = [1, -1, 0.5, -0.5]
                    
                    # U: 4 entries (i=0,1; j=0,1)
                    for i in range(n):
                        for j in range(m):
                            if random.random() < 0.7:
                                U[i*m + j][r] = random.choice(coeffs)
                    
                    # V: 16 entries (j=0,1; k=0..7)
                    for j in range(m):
                        for k in range(p):
                            # Only non-zero for k in current pair
                            if k == k0 or k == k1:
                                if random.random() < 0.7:
                                    V[j*p + k][r] = random.choice(coeffs)
                    
                    # W: 16 entries (k=0..7; i=0,1)
                    for k in range(p):
                        for i in range(n):
                            if k == k0 or k == k1:
                                if random.random() < 0.7:
                                    W[k*n + i][r] = random.choice(coeffs)
            
            # Verify
            if verify_fast(U, V, W, target_rank):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    result = {
        "U": best["U"],
        "V": best["V"],
        "W": best["W"]
    }
    
    return result
# EVOLVE-BLOCK-END
