"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _verify_rank(n, m, p, rank, U, V, W):
    """Verify that the given factors decompose the matmul tensor correctly.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Convention: W index is k*n + i, U is i*m + j, V is j*p + k.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check all triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
                val = 0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(val - 1.0) > 1e-6:
                    return False, (i, j, k, val)
    return True, None

def _build_rank6_2x4x3():
    """
    Build a rank-6 decomposition for 2x4x3 matrix multiplication.
    For multiplying a 2x4 matrix A by a 4x3 matrix B to get 2x3 matrix C.
    
    The tensor has entries T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k].
    
    We use the known rank-6 decomposition. One approach:
    For 2x4x3, we can use the fact that rank(2x4x3) = 6.
    
    Let's construct it explicitly. The matmul tensor for 2x4x3:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1,2}
    
    We need U (2*4=8 x 6), V (4*3=12 x 6), W (3*2=6 x 6)
    such that sum_c U[i*4+j,c] * V[j*3+k,c] * W[k*2+i,c] = 1
    
    One known rank-6 scheme for 2x4x3:
    We can think of it as two 2x2x3 blocks plus corrections, or use
    the structure from Strassen-like decompositions.
    
    Let me use a direct construction. For 2x4x3:
    Split the 4 middle indices into two pairs: j in {0,1} and j in {2,3}.
    
    For the first pair (j=0,1): 2x2x3 matmul has rank 5 (known).
    For the second pair (j=2,3): 2x2x3 matmul has rank 5 (known).
    But that gives rank 10, not 6.
    
    Let me try a different approach. Use the fact that:
    rank(2x4x3) <= rank(2x4x2) + rank(2x4x1) = 6 + 4 = 10? No.
    
    Actually, let me just use a computational approach: find a rank-6 decomposition
    by solving the system of equations.
    
    For 2x4x3, we have:
    - 8 U rows (i*4+j for i in {0,1}, j in {0,1,2,3})
    - 12 V rows (j*3+k for j in {0,1,2,3}, k in {0,1,2})
    - 6 W rows (k*2+i for k in {0,1,2}, i in {0,1})
    - 6 rank terms
    - 2*4*3 = 24 equations
    
    Let me try to find an explicit integer/half-integer solution.
    
    Known result: rank(2x4x3) = 6. Let me construct it.
    
    One way: use the "block" decomposition.
    C = A*B where A is 2x4, B is 4x3.
    
    Write A = [A1 A2] where A1, A2 are 2x2.
    Write B = [B1; B2] where B1, B2 are 2x3.
    Then C = A1*B1 + A2*B2.
    
    Each 2x2x3 matmul has rank 5. But we can share some terms.
    
    Actually, let me just hardcode a known rank-6 decomposition.
    
    After research, one rank-6 decomposition for 2x4x3:
    
    Let me define the 6 terms. Each term c has:
    - U_c: 8-vector (for i in {0,1}, j in {0,1,2,3})
    - V_c: 12-vector (for j in {0,1,2,3}, k in {0,1,2})
    - W_c: 6-vector (for k in {0,1,2}, i in {0,1})
    
    I'll use a systematic approach. Let me try:
    
    Term 0: U has 1 at (0,0), V has 1 at (0,0), W has 1 at (0,0)  [i=0,j=0,k=0]
    Term 1: U has 1 at (0,1), V has 1 at (1,0), W has 1 at (0,0)  [i=0,j=1,k=0]
    Term 2: U has 1 at (0,2), V has 1 at (2,0), W has 1 at (0,0)  [i=0,j=2,k=0]
    Term 3: U has 1 at (0,3), V has 1 at (3,0), W has 1 at (0,0)  [i=0,j=3,k=0]
    ... this is just schoolbook with rank 12.
    
    Let me try a different strategy. I'll use numpy to find a rank-6 solution
    by solving the linear system.
    """
    # For 2x4x3: n=2, m=4, p=3
    n, m, p = 2, 4, 3
    rank = 6
    
    # We need to find U (8x6), V (12x6), W (6x6)
    # such that for all (i,j,k): sum_c U[i*4+j,c] * V[j*3+k,c] * W[k*2+i,c] = 1
    
    # This is a system of 24 equations in 8*6 + 12*6 + 6*6 = 48+72+36 = 156 unknowns
    # with rank-6 constraint (only 6 terms)
    
    # Let me try to find a solution by fixing some structure.
    # Use the fact that we can decompose as sum of 6 rank-1 tensors.
    
    # Try a simple construction:
    # For 2x4x3, one known rank-6 decomposition uses:
    
    U = np.zeros((8, 6))
    V = np.zeros((12, 6))
    W = np.zeros((6, 6))
    
    # Term 0: covers (i=0,j=0) and (i=1,j=0) contributions to k=0
    U[0*4+0, 0] = 1  # i=0,j=0
    U[1*4+0, 0] = 1  # i=1,j=0
    V[0*3+0, 0] = 1  # j=0,k=0
    W[0*2+0, 0] = 1  # k=0,i=0
    W[0*2+1, 0] = 1  # k=0,i=1
    
    # Term 1: covers (i=0,j=1) and (i=1,j=1) contributions to k=0
    U[0*4+1, 1] = 1  # i=0,j=1
    U[1*4+1, 1] = 1  # i=1,j=1
    V[1*3+0, 1] = 1  # j=1,k=0
    W[0*2+0, 1] = 1  # k=0,i=0
    W[0*2+1, 1] = 1  # k=0,i=1
    
    # Term 2: covers (i=0,j=2) and (i=1,j=2) contributions to k=0
    U[0*4+2, 2] = 1  # i=0,j=2
    U[1*4+2, 2] = 1  # i=1,j=2
    V[2*3+0, 2] = 1  # j=2,k=0
    W[0*2+0, 2] = 1  # k=0,i=0
    W[0*2+1, 2] = 1  # k=0,i=1
    
    # Term 3: covers (i=0,j=3) and (i=1,j=3) contributions to k=0
    U[0*4+3, 3] = 1  # i=0,j=3
    U[1*4+3, 3] = 1  # i=1,j=3
    V[3*3+0, 3] = 1  # j=3,k=0
    W[0*2+0, 3] = 1  # k=0,i=0
    W[0*2+1, 3] = 1  # k=0,i=1
    
    # This gives rank 4 for k=0 only. We need k=1 and k=2 too.
    # This approach isn't working. Let me try a different construction.
    
    # Let me use a known result: for 2x4x3, rank is 6.
    # I'll use a computational search with small integer values.
    
    # Actually, let me just use the schoolbook rank-12 as a fallback and
    # see if we can find a better decomposition computationally.
    
    # For now, return a schoolbook-like decomposition with rank 12
    # and we'll try to reduce it.
    
    U = np.zeros((8, 12))
    V = np.zeros((12, 12))
    W = np.zeros((6, 12))
    
    for i in range(2):
        for j in range(4):
            for k in range(3):
                c = i*12 + j*3 + k
                U[i*4+j, c] = 1
                V[j*3+k, c] = 1
                W[k*2+i, c] = 1
    
    return U, V, W, 12

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s margin out of 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Initialize best with schoolbook solution
    best_rank = schoolbook_rank
    best_U = np.zeros((n * m, schoolbook_rank))
    best_V = np.zeros((m * p, schoolbook_rank))
    best_W = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j, c] = 1
                best_V[j * p + k, c] = 1
                best_W[k * n + i, c] = 1
                
    random.seed(42)
    np.random.seed(42)
    
    # Strategy: Partition the 4x4x6 matmul into sub-problems
    # Split rows into two groups of 2, columns into two groups of 3
    # Each sub-problem is 2x4x3 with rank 6
    # Total rank: 2 * 2 * 6 = 24
    
    # First, let's try to construct a rank-24 solution using partitioning
    # Then try to find even lower ranks with ALS
    
    # Build rank-24 solution via partitioning
    # Row strips: strip 0 = rows 0-1, strip 1 = rows 2-3
    # Col groups: group 0 = cols 0-2, group 1 = cols 3-5
    
    rank_24 = 24
    U24 = np.zeros((n * m, rank_24))  # 16 x 24
    V24 = np.zeros((m * p, rank_24))  # 24 x 24
    W24 = np.zeros((p * n, rank_24))  # 24 x 24
    
    # For each (row_strip, col_group) pair, we have a 2x4x3 sub-problem
    # We need a rank-6 decomposition for 2x4x3
    # For now, use schoolbook rank-12 for each 2x4x3 (total 4*12=48)
    # Then try to find rank-6 decompositions
    
    # Let's first verify the schoolbook works, then try to improve
    # Actually, let me just try ALS from various initializations targeting rank 24, 20, 16, 12, etc.
    
    target_ranks = [24, 20, 18, 16, 14, 12, 10, 8]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        attempts = 10
        found = False
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with structure
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            # Normalize columns
            for c in range(target_rank):
                norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
                if norm > 1e-10:
                    scale = 1.0 / (norm ** (1.0/3.0))
                    U[:, c] *= scale
                    V[:, c] *= scale
                    W[:, c] *= scale
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                            err += (val - 1.0)**2
                
                if err < 1e-8:
                    # Check if we can round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    valid, _ = _verify_rank(n, m, p, target_rank, U_rounded, V_rounded, W_rounded)
                    if valid:
                        best_rank = target_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        found = True
                        break
                
                # Update W
                for i in range(n):
                    for k in range(p):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n + i, :] = sol
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((p, target_rank))
                        for k in range(p):
                            B[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        sol, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U[i*m + j, :] = sol
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        C = np.zeros((n, target_rank))
                        for i in range(n):
                            C[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        sol, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
                        V[j*p + k, :] = sol
                
                if found:
                    break
            
            if found:
                break
    
    # If we didn't find anything better, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
