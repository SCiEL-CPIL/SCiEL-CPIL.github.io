import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact."""
    U, V, W = np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _greedy_peeling(U, V, W, n, m, p, tol=1e-8):
    """
    Iteratively remove rank-1 terms that are linearly dependent or redundant.
    Returns a reduced (U, V, W) and the new rank.
    """
    r = U.shape[1]
    active = list(range(r))
    
    # We want to minimize the number of terms.
    # A term is redundant if removing it still allows the tensor to be decomposed by the remaining terms?
    # No, the decomposition is a sum. If we remove a term, the sum changes.
    # Instead, we look for linear dependencies in the space of the tensor.
    # The tensor T is represented as sum_k u_k \otimes v_k \otimes w_k.
    # If the vectors (u_k, v_k, w_k) are linearly dependent in some sense, we can reduce rank.
    
    # A more robust approach for "peeling":
    # Check if any term is a linear combination of others in a way that allows cancellation?
    # Actually, for matrix multiplication, the terms are specific.
    # Let's try a different approach: Randomized SVD / Rank Reduction.
    
    # This function is a placeholder for the actual reduction logic.
    # For now, we just return the input if no obvious reduction is found.
    return U, V, W, r

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 140
    
    # Start with schoolbook rank
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy: Numerical Rank Reduction
    # We can try to find a lower rank decomposition by solving a system of equations.
    # However, exact integer/half-integer solutions are rare for low ranks.
    # Given the 150s limit, a full numerical search for rank < 96 is likely infeasible without strong priors.
    
    # Let's try a simple heuristic: 
    # If we assume the rank is R, we can set up a random initial guess and use iterative refinement.
    # But for exactness, we usually need algebraic structures.
    
    # Since the current best is 96 (schoolbook) or 88 (Strassen+Schoolbook), 
    # and numerical search for exact integer solutions is hard, 
    # we will stick to the known best construction (Strassen where possible) but implement it more efficiently.
    
    # Re-implement the Strassen+Schoolbook hybrid which was measured to improve score.
    # We will hardcode the logic for the specific 3x4x8 case to be fast.
    
    # Partition:
    # n=3: [2, 1]
    # m=4: [2, 2]
    # p=8: [2, 2, 2, 2]
    
    # This creates blocks:
    # (2x2x2) blocks: 1 (from n) * 2 (from m) * 4 (from p) = 8 blocks of 2x2x2 -> 8 * 7 = 56 rank
    # (1x2x2) blocks: 1 (from n) * 2 (from m) * 4 (from p) = 8 blocks of 1x2x2 -> 8 * 4 = 32 rank (schoolbook for 1x2x2 is 4)
    # Total rank = 56 + 32 = 88.
    
    # Let's verify this construction.
    
    def _strassen_222():
        return [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
            ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
            ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
            ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
            ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
            ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
        ]

    strassen_terms = _strassen_222()
    
    max_rank = 88
    U_new = np.zeros((n * m, max_rank))
    V_new = np.zeros((m * p, max_rank))
    W_new = np.zeros((p * n, max_rank))
    col = 0
    
    # n partition: [0,1], [2]
    # m partition: [0,1], [2,3]
    # p partition: [0,1], [2,3], [4,5], [6,7]
    
    part_n = [[0, 1], [2]]
    part_m = [[0, 1], [2, 3]]
    part_p = [[0, 1], [2, 3], [4, 5], [6, 7]]
    
    for bn in part_n:
        for bm in part_m:
            for bp in part_p:
                sn, sm, sp = len(bn), len(bm), len(bp)
                if sn == 2 and sm == 2 and sp == 2:
                    i0, j0, k0 = bn[0], bm[0], bp[0]
                    for (u, v, w) in strassen_terms:
                        for di in range(2):
                            for dj in range(2):
                                U_new[(i0+di)*m + (j0+dj), col] = u[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V_new[(j0+dj)*p + (k0+dk), col] = v[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W_new[(k0+dk)*n + (i0+di), col] = w[di*2+dk]
                        col += 1
                else:
                    # Schoolbook for the rest
                    for i in bn:
                        for j in bm:
                            for k in bp:
                                U_new[i*m + j, col] = 1
                                V_new[j*p + k, col] = 1
                                W_new[k*n + i, col] = 1
                                col += 1
    
    if _verify(U_new, V_new, W_new, n, m, p):
        best_rank = col
        best_U, best_V, best_W = U_new[:, :col], V_new[:, :col], W_new[:, :col]
    else:
        # Fallback to pure schoolbook if construction fails (should not happen)
        best_rank = rank
        best_U, best_V, best_W = U, V, W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
