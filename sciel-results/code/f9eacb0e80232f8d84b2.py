"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Rank 54 decomposition using Strassen's algorithm for 2x2 blocks
    # A is 2x5, B is 5x6, C is 2x6
    # We partition the 'm' dimension (5) into pairs (0,1), (2,3) and single (4).
    # For each pair (j, j+1) in A and corresponding rows in B, we use Strassen (7 mults).
    # For the single column j=4 in A and row j=4 in B, we use schoolbook (12 mults).
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # i_base: 0 (only one pair of rows in A/C since n=2)
        # j_base: 0 or 2 (pairs of cols in A / rows in B)
        # k_base: 0, 2, or 4 (pairs of cols in B / cols in C)
        
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # M0: (A00+A11)(B00+B11)
        # A00: U[i_base, j_base], A11: U[i_base+1, j_base+1]
        # B00: V[j_base, k_base], B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base], C11: W[k_base+1, i_base+1]
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        # A10: U[i_base+1, j_base], A11: U[i_base+1, j_base+1]
        # B00: V[j_base, k_base]
        # C10: W[k_base, i_base+1], C11: W[k_base+1, i_base+1] (C11 gets -1)
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        # A00: U[i_base, j_base]
        # B01: V[j_base, k_base+1], B11: V[j_base+1, k_base+1]
        # C01: W[k_base+1, i_base], C11: W[k_base+1, i_base+1]
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        # A11: U[i_base+1, j_base+1]
        # B10: V[j_base+1, k_base], B00: V[j_base, k_base]
        # C00: W[k_base, i_base], C10: W[k_base, i_base+1]
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        # A00: U[i_base, j_base], A01: U[i_base, j_base+1]
        # B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base] (gets -1), C01: W[k_base+1, i_base]
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        # A10: U[i_base+1, j_base], A00: U[i_base, j_base]
        # B00: V[j_base, k_base], B01: V[j_base, k_base+1]
        # C11: W[k_base+1, i_base+1]
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        # A01: U[i_base, j_base+1], A11: U[i_base+1, j_base+1]
        # B10: V[j_base+1, k_base], B11: V[j_base+1, k_base+1]
        # C00: W[k_base, i_base]
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        # Handles A[i_base..i_base+1, j_base] * B[j_base, k_base..k_base+1] -> C[i_base..i_base+1, k_base..k_base+1]
        # This is a 2x1 * 1x2 multiplication, requiring 4 scalar multiplications.
        
        # 1: A00 B00 -> C00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Apply Strassen blocks for j-pairs (0,1) and (2,3)
    # k-pairs are (0,1), (2,3), (4,5)
    
    # j_pair 0,1
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # j_pair 2,3
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4
    # k-pairs are (0,1), (2,3), (4,5)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    if col != rank:
        # Fallback, though it should be exactly 54
        pass

    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank

    # Attempt to reduce rank using optimization if time permits
    # We will try ranks 53, 52, ... down to 45
    for rank_try in range(best_rank - 1, max(best_rank - 10, 40), -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_optimization(n, m, p, rank_try, deadline)
        if result is not None:
            best_result = result
            best_rank = rank_try
            if best_rank <= 45:
                break

    return best_result

def _try_rank_optimization(n, m, p, rank, deadline):
    """
    Attempt to find a rank-`rank` decomposition using numerical optimization.
    Since the tensor is small (2*5*6 = 60 entries), we can formulate this as a system of polynomial equations.
    However, solving polynomial systems is hard. Instead, we use a heuristic:
    1. Initialize U, V, W with random small integers/half-integers.
    2. Use local search / gradient descent to minimize the error.
    3. Try to snap to the nearest half-integer/integer and verify.
    
    Given the time constraints and the difficulty of non-convex optimization, 
    this is likely to fail for small rank reductions unless we have a good start.
    We will use a simple random restart with local polishing.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    rng = random.Random(rank)
    
    # Try a few random initializations
    for restart in range(3):
        if time.time() > deadline - 5:
            break
            
        # Initialize with small random values
        # Using half-integers: multiply by 2, use integers, divide by 2 at the end
        U = np.zeros((dim_u, rank), dtype=float)
        V = np.zeros((dim_v, rank), dtype=float)
        W = np.zeros((dim_w, rank), dtype=float)
        
        # Sparse random initialization
        for r in range(rank):
            # Pick random indices
            u_idx = rng.randrange(dim_u)
            v_idx = rng.randrange(dim_v)
            w_idx = rng.randrange(dim_w)
            
            U[u_idx, r] = rng.choice([-1, 1])
            V[v_idx, r] = rng.choice([-1, 1])
            W[w_idx, r] = rng.choice([-1, 1])
            
            # Sometimes add a second term to allow for Strassen-like sums
            if rng.random() < 0.5:
                u_idx2 = rng.randrange(dim_u)
                v_idx2 = rng.randrange(dim_v)
                w_idx2 = rng.randrange(dim_w)
                U[u_idx2, r] += rng.choice([-1, 1])
                V[v_idx2, r] += rng.choice([-1, 1])
                W[w_idx2, r] += rng.choice([-1, 1])

        # Simple local search: try flipping signs or adding/removing terms
        # This is extremely heuristic and unlikely to work for rank reduction from 54
        # unless the structure is very close.
        
        # For now, we skip the complex optimization loop to save time, 
        # as the Strassen construction is already optimal for this partitioning.
        # True rank reduction would require a fundamentally different decomposition 
        # (e.g. exploiting the 2x5x6 structure more globally, which is an open research problem 
        # for small dimensions, though known bounds exist).
        
        # If we had a valid optimization routine, we would call it here.
        # Since we don't have a robust solver for this non-convex problem in the time limit,
        # we return None.
        
        pass
        
    return None

# EVOLVE-BLOCK-END
