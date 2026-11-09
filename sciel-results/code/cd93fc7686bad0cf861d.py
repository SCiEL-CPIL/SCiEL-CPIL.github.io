"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against the identity tensor: T[i,j,k] = 1 if (i,j,k) valid, 0 otherwise
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try to find a rank-10 decomposition for 2x3x2 block
    # If found, total rank = 50. Otherwise fall back to rank-11 (total 55).
    
    # First, try a systematic search for rank-10
    best_rank = 55
    best_decomp = None
    
    # Known rank-11 decomposition (Strassen 7 + outer 4)
    def get_rank11_decomp():
        strassen_terms = [
            {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
            {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
            {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
            {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
            {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
        ]
        outer_terms = [
            {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
            {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
            {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
            {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
        ]
        return strassen_terms + outer_terms
    
    # Try to find rank-10 by combining Strassen-like terms with the outer part
    # Key insight: can we merge one Strassen term with one outer term?
    # Try: use 6 Strassen terms + 4 outer terms = 10, but need to verify correctness
    
    # Alternative: try a different partition. Instead of splitting j into {0,1} and {2},
    # try other algebraic combinations.
    
    # Let's try: 7 Strassen + 3 merged outer terms
    # Can we combine two outer products into one term?
    # A02*B20 and A12*B21 -> need separate terms since they go to different C entries
    
    # Actually, let's try a completely different approach:
    # Use the fact that we have 3 columns in A and 3 rows in B.
    # Try rank-10 with a different structure.
    
    # For now, use the known rank-11. Total = 55.
    rank_per_block = 11
    total_rank = 5 * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    all_terms = get_rank11_decomp()
    
    col = 0
    for b in range(5):
        for term in all_terms:
            for i, j, val in term['U']:
                U[i * m + j][col] += val
            for j, k_local, val in term['V']:
                k_global = b * 2 + k_local
                V[j * p + k_global][col] += val
            for k_local, i, val in term['W']:
                k_global = b * 2 + k_local
                W[k_global * n + i][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
