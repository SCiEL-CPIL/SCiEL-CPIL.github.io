"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, rank):
    """Verify that T[i*m+j, j*p+k, k*n+i] = sum_c U[..,c]*V[..,c]*W[..,c] = 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_numpy(U_np, V_np, W_np, n, m, p):
    """Fast numpy verification."""
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def _try_random(rank, n, m, p, deadline, max_iters=5000, sparse=False):
    """Try random search for rank-r decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        if sparse:
            nnz = max(1, rank // 5)
            U = [[0.0]*rank for _ in range(nm)]
            V = [[0.0]*rank for _ in range(mp)]
            W = [[0.0]*rank for _ in range(pn)]
            for r in range(nm):
                cols = random.sample(range(rank), min(nnz, rank))
                for c in cols:
                    U[r][c] = random.choice(vals)
            for r in range(mp):
                cols = random.sample(range(rank), min(nnz, rank))
                for c in cols:
                    V[r][c] = random.choice(vals)
            for r in range(pn):
                cols = random.sample(range(rank), min(nnz, rank))
                for c in cols:
                    W[r][c] = random.choice(vals)
        else:
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(pn)]
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        if _verify_numpy(U_np, V_np, W_np, n, m, p):
            return U, V, W
    return None

def _try_block_decomp(n, m, p):
    """
    Try block-based decomposition.
    For 2x4x5: split m=4 into two blocks of 2.
    This gives two independent 2x2x5 subproblems.
    Each 2x2x5 can be further analyzed.
    
    Actually, let's try a different block structure:
    Split the computation into overlapping blocks using Winograd-style transforms.
    
    For 2x2x2: Strassen gives rank 7
    For 2x2x3: Winograd gives rank 8
    For 2x2x5: we can split 5 into 2+2+1, using rank 7+7+4=18? Or better.
    
    Let me try: decompose 2x4x5 as:
    - Split m=4 into m1=2, m2=2
    - C = A1*B1 + A2*B2 where A1 is 2x2, B1 is 2x5, A2 is 2x2, B2 is 2x5
    - Each 2x2x5 subproblem: can use rank-? decomposition
    
    For 2x2x5: split p=5 into p1=2, p2=2, p3=1
    - 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x1 (rank 4) = 18 per block
    - Total: 2 * 18 = 36? That's not better than 40.
    
    Let me try a smarter approach: use the fact that we can share some terms.
    """
    # Try to construct a rank-30 or lower decomposition using block structure
    # For now, return None and rely on random search
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 40
    
    # Try to find lower rank decompositions
    # Known lower bound for 2x4x5 is at least 20 (trivial) but likely higher
    # Try ranks from 25 upward (optimistic) to 39
    
    for target_rank in range(25, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random search for this rank
        result = _try_random(target_rank, n, m, p, deadline - 10, max_iters=3000, sparse=True)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    # If we found something better, great. Otherwise use schoolbook.
    # Final verification
    if best_rank < n*m*p:
        if not _verify(best_U, best_V, best_W, n, m, p, best_rank):
            # Fallback to schoolbook
            best_U, best_V, best_W = U_sb, V_sb, W_sb
            best_rank = n*m*p
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
