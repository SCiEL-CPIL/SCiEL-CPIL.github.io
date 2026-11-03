"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Default: Schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    def is_valid(U, V, W):
        # Verify decomposition
        # T[a, b, o] = sum_c U[a, c] * V[b, c] * W[o, c]
        # Should match 1 if (a,b,o) correspond to (i,j,k) with a=im+j, b=jp+k, o=kn+i
        # And 0 otherwise.
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Flatten tensors
        # Check a few random points or all points
        # All points: 9 * 18 * 18 = 2916 checks. Rank 54 -> 157k mults. Fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    o = k * n + i
                    val = 0
                    for c in range(rank):
                        val += U[a][c] * V[b][c] * W[o][c]
                    if val != 1:
                        return False
        # Check some zero entries
        # a=0 (i=0,j=0), b=0 (j=0,k=0), o=1 (k=0,i=1) -> Mismatch k/i? 
        # T[0,0,1] should be 0 because i=0, k=0 in a/b but i=1 in o.
        # Let's check a few mismatches.
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            o = random.randint(0, p*n - 1)
            # Decode indices
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_o = o // n
            i_o = o % n
            
            if i_a == i_o and j_a == j_b and j_b == k_b: # Wait, j_a==j_b is required. k_b is k.
                # If indices match, value should be 1
                pass
            else:
                # If indices don't match, value should be 0
                # Specifically, if j_a != j_b, it's 0.
                # If i_a != i_o, it's 0.
                # If j_a == j_b and i_a == i_o, then k_b must be k_o? 
                # No, k is determined by b and o.
                # If j_a == j_b, then b corresponds to (j, k_b).
                # o corresponds to (k_o, i_o).
                # a corresponds to (i_a, j_a).
                # Match if i_a == i_o and j_a == j_b. k is free?
                # No, T[i,j,k] = 1.
                # a=im+j, b=jp+k, o=kn+i.
                # So if i_a==i_o and j_a==j_b, then k_b must be k_o?
                # Yes, because b fixes k, o fixes k.
                if i_a == i_o and j_a == j_b:
                    if k_b != k_o:
                        # Should be 0
                        val = 0
                        for c in range(rank):
                            val += U[a][c] * V[b][c] * W[o][c]
                        if val != 0:
                            return False
                else:
                    # Should be 0
                    val = 0
                    for c in range(rank):
                        val += U[a][c] * V[b][c] * W[o][c]
                    if val != 0:
                        return False
        return True

    # Try to find a better decomposition?
    # For now, return schoolbook if valid.
    if is_valid(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback if verification fails (should not happen)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
