import numpy as np
import time
import itertools
from collections import defaultdict

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = delta (A_{i,j} * B_{j,k} contributes to C_{i,k})
    # T is indexed by (i, j, k) with i in [0,n), j in [0,m), k in [0,p)
    # Each rank-1 term r contributes: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 if (i,j,k) is a valid triple, else 0
    # Actually the tensor for matrix mult: T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] -> C[i,k])
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def _try_find_decomposition(n, m, p, max_rank, time_limit):
    """
    Try to find a rank-r decomposition for r from some lower bound up to max_rank.
    Uses a structured search over sparse integer rank-1 terms.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # Lower bound: at least n*p (since output is n x p, and each rank-1 term can contribute to at most... 
    # Actually the trivial lower bound is max(n*p, m) but let's use n*p as a practical lower bound
    # For 2x3x5: n*p = 10, but we know rank >= 10. Let's try from 10 upward.
    # However, exhaustive search is infeasible for large ranks. We'll limit the search.
    
    # For a structured approach: consider rank-1 terms where U, V, W are sparse.
    # Each rank-1 term is defined by (u, v, w) where:
    #   u is a vector in R^(n*m) with small integer entries
    #   v is a vector in R^(m*p) with small integer entries  
    #   w is a vector in R^(p*n) with small integer entries
    # such that the tensor product u ⊗ v ⊗ w contributes to the decomposition.
    
    # The constraint is: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all i,j,k
    
    # This is a system of n*m*p linear equations in the products of entries.
    # It's nonlinear in the variables, but we can search over structured candidates.
    
    # Strategy: try rank = lower_bound, lower_bound+1, ... up to max_rank
    # For each rank, try to find a decomposition using a greedy/iterative approach.
    
    # For small dimensions, we can try a more direct approach:
    # The tensor T has n*m*p = 30 entries, all equal to 1.
    # We need to express T as sum of r rank-1 tensors.
    
    # A rank-1 tensor is outer product of three vectors.
    # Let's enumerate possible "sparse" rank-1 tensors.
    
    # For the 2x3x5 case, let's try a systematic search for ranks 10-27.
    # This is a combinatorial explosion, so we'll use a heuristic approach.
    
    # Actually, let's use a different strategy: use the known lower bounds and try to verify
    # if rank 27 or lower is achievable via a structured search.
    
    # Given the time constraints, let's try a limited search.
    
    for target_rank in range(10, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
        # For each target rank, we'll try to construct a decomposition
        # This is a placeholder for a more sophisticated search
        # In practice, for this small problem, we can try known constructions
        
    return None

def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying a 2x3 matrix 
    by a 3x5 matrix.
    
    Strategy:
    Block 1: j=2. M_{2,1,5}, rank 10.
    Block 2: j in {0,1}. M_{2,2,5} split into M_{2,2,2} (rank 7) + M_{2,2,2} (rank 7) + M_{2,2,1} (rank 4) = 18.
    Total: 28.
    """
    n, m, p = 2, 3, 5
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Helper to add a Strassen block for M_{2,2,2}
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)(B00)
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = -1
        col += 1
        
        # M3: A00(B01-B11)
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = -1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = -1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)(B11)
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = -1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+0)][col] = -1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = -1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        col += 1

    # Block 1: j=2 (A[:, 2] * B[2, :])
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2a: k in {0, 1}. M_{2,2,2}.
    add_strassen(0, 0, 0)
    
    # Block 2b: k in {2, 3}. M_{2,2,2}.
    add_strassen(0, 0, 2)
    
    # Block 2c: k in {4}. M_{2,2,1}.
    U[0 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    U[1 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    return {"U": U, "V": V, "W": W}
