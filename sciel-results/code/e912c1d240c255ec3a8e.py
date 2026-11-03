"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 75
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for 120s limit
    
    def check_deadline():
        return time.time() > deadline
    
    # Build schoolbook decomposition as fallback
    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition: check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Strategy: Use the block decomposition approach from inspiration 3,
    # but with improved low-rank search for sub-problems.
    # 
    # The key insight from the current program is to split 5 = 2 + 3,
    # giving 4 sub-problems. We can try to find lower-rank decompositions
    # for each sub-problem using alternating least squares.
    
    def find_low_rank_decomp(n_sub, m_sub, p_sub, target_rank, max_iters=100, n_restarts=3):
        """Try to find a rank-target_rank decomposition for n_sub x m_sub x p_sub."""
        if target_rank >= n_sub * m_sub * p_sub:
            # Use schoolbook
            U = [[0] * (n_sub * m_sub * p_sub) for _ in range(n_sub * m_sub)]
            V = [[0] * (n_sub * m_sub * p_sub) for _ in range(m_sub * p_sub)]
            W = [[0] * (n_sub * m_sub * p_sub) for _ in range(p_sub * n_sub)]
            col = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U[i * m_sub + j][col] = 1
                        V[j * p_sub + k][col] = 1
                        W[k * n_sub + i][col] = 1
                        col += 1
            return U, V, W, n_sub * m_sub * p_sub
        
        best_rank = n_sub * m_sub * p_sub
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(n_restarts):
            if check_deadline():
                break
            
            np.random.seed(42 + attempt * 17)
            U = np.random.randn(n_sub * m_sub, target_rank) * 0.5
            V = np.random.randn(m_sub * p_sub, target_rank) * 0.5
            W = np.random.randn(p_sub * n_sub, target_rank) * 0.5
            
            # Iterative refinement: alternate between solving for each matrix
            converged = False
            for iter in range(max_iters):
                if check_deadline():
                    break
                
                # Update U: fix V, W
                # For each (i,j): sum_k sum_c U[i*m_sub+j,c] * V[j*p_sub+k,c] * W[k*n_sub+i,c] = 1
                # Let A[i*m_sub+j, c] = sum_k V[j*p_sub+k,c] * W[k*n_sub+i,c]
                # Then A[i*m_sub+j, :] * U[i*m_sub+j, :] = 1
                A = np.zeros((n_sub * m_sub, target_rank))
                for i in range(n_sub):
                    for j in range(m_sub):
                        for c in range(target_rank):
                            s = 0
                            for k in range(p_sub):
                                s += V[j * p_sub + k, c] * W[k * n_sub + i, c]
                            A[i * m_sub + j, c] = s
                
                # Solve row by row
                for r in range(n_sub * m_sub):
                    if np.linalg.norm(A[r, :]) > 1e-10:
                        U[r, :] = np.linalg.lstsq(A[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                
                # Update V: fix U, W
                # For each (j,k): sum_i sum_c U[i*m_sub+j,c] * V[j*p_sub+k,c] * W[k*n_sub+i,c] = 1
                # Let B[j*p_sub+k, c] = sum_i U[i*m_sub+j,c] * W[k*n_sub+i,c]
                B = np.zeros((m_sub * p_sub, target_rank))
                for j in range(m_sub):
                    for k in range(p_sub):
                        for c in range(target_rank):
                            s = 0
                            for i in range(n_sub):
                                s += U[i * m_sub + j, c] * W[k * n_sub + i, c]
                            B[j * p_sub + k, c] = s
                
                for r in range(m_sub * p_sub):
                    if np.linalg.norm(B[r, :]) > 1e-10:
                        V[r, :] = np.linalg.lstsq(B[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                
                # Update W: fix U, V
                # For each (k,i): sum_j sum_c U[i*m_sub+j,c] * V[j*p_sub+k,c] * W[k*n_sub+i,c] = 1
                # Let C[k*n_sub+i, c] = sum_j U[i*m_sub+j,c] * V[j*p_sub+k,c]
                C = np.zeros((p_sub * n_sub, target_rank))
                for k in range(p_sub):
                    for i in range(n_sub):
                        for c in range(target_rank):
                            s = 0
                            for j in range(m_sub):
                                s += U[i * m_sub + j, c] * V[j * p_sub + k, c]
                            C[k * n_sub + i, c] = s
                
                for r in range(p_sub * n_sub):
                    if np.linalg.norm(C[r, :]) > 1e-10:
                        W[r, :] = np.linalg.lstsq(C[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                
                # Check convergence
                err = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            val = np.sum(U[i * m_sub + j, :] * V[j * p_sub + k, :] * W[k * n_sub + i, :])
                            err += (val - 1) ** 2
                
                if err < 1e-8:
                    converged = True
                    break
            
            if converged:
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        else:
            # Fall back to schoolbook
            rank = n_sub * m_sub * p_sub
            U = [[0] * rank for _ in range(n_sub * m_sub)]
            V = [[0] * rank for _ in range(m_sub * p_sub)]
            W = [[0] * rank for _ in range(p_sub * n_sub)]
            col = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U[i * m_sub + j][col] = 1
                        V[j * p_sub + k][col] = 1
                        W[k * n_sub + i][col] = 1
                        col += 1
            return U, V, W, rank
    
    # Try to find lower-rank decompositions for the sub-problems
    # Sub-problem 1: 3x2x2 (rank 12)
    # Sub-problem 2: 3x3x2 (rank 18)
    # Sub-problem 3: 3x2x3 (rank 18)
    # Sub-problem 4: 3x3x3 (rank 27)
    
    best_total_rank = rank_school
    best_U, best_V, best_W = build_schoolbook()
    
    # Try a few target ranks for the sub-problems, starting from schoolbook
    # and going down
    sub_configs = [
        (3, 2, 2, 12),  # Sub-problem 1
        (3, 3, 2, 18),  # Sub-problem 2
        (3, 2, 3, 18),  # Sub-problem 3
        (3, 3, 3, 27),  # Sub-problem 4
    ]
    
    # Try to reduce ranks for each sub-problem
    # Start with the largest sub-problem (3x3x3) and try to reduce it
    # Then try the others
    
    # For 3x3x3, try ranks 26, 25, 24, 23
    # For 3x3x2 and 3x2x3, try ranks 17, 16, 15
    # For 3x2x2, try ranks 11, 10, 9
    
    target_ranks = {
        (3, 2, 2): [12, 11, 10, 9],
        (3, 3, 2): [18, 17, 16, 15],
        (3, 2, 3): [18, 17, 16, 15],
        (3, 3, 3): [27, 26, 25, 24, 23],
    }
    
    # Try all combinations of target ranks
    # But this is too many, so let's be more strategic
    # Start with the best known and try to improve one at a time
    
    # First, try to find a good decomposition for 3x3x3 with rank 23 (Strassen)
    U4, V4, W4, r4 = find_low_rank_decomp(3, 3, 3, 23, max_iters=50, n_restarts=5)
    
    if r4 < 27:
        # Great, we found a better decomposition for 3x3x3
        # Now combine with schoolbook for the other sub-problems
        total_rank = 12 + 18 + 18 + r4
        if total_rank < best_total_rank:
            # Build the full decomposition
            U_full = [[0] * total_rank for _ in range(n * m)]
            V_full = [[0] * total_rank for _ in range(m * p)]
            W_full = [[0] * total_rank for _ in range(p * n)]
            
            col = 0
            # Sub-problem 1: 3x2x2, schoolbook
            U1, V1, W1, r1 = find_low_rank_decomp(3, 2, 2, 12, max_iters=10, n_restarts=1)
            for i in range(3):
                for j in range(2):
                    for c in range(r1):
                        U_full[i * m + j][col + c] = U1[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    for c in range(r1):
                        V_full[j * p + k][col + c] = V1[j * 2 + k][c]
            for k in range(2):
                for i in range(3):
                    for c in range(r1):
                        W_full[k * n + i][col + c] = W1[k * 3 + i][c]
            col += r1
            
            # Sub-problem 2: 3x3x2, schoolbook
            U2, V2, W2, r2 = find_low_rank_decomp(3, 3, 2, 18, max_iters=10, n_restarts=1)
            for i in range(3):
                for j in range(3):
                    for c in range(r2):
                        U_full[i * m + (j + 2)][col + c] = U2[i * 3 + j][c]
            for j in range(3):
                for k in range(2):
                    for c in range(r2):
                        V_full[(j + 2) * p + k][col + c] = V2[j * 2 + k][c]
            for k in range(2):
                for i in range(3):
                    for c in range(r2):
                        W_full[k * n + i][col + c] = W2[k * 3 + i][c]
            col += r2
            
            # Sub-problem 3: 3x2x3, schoolbook
            U3, V3, W3, r3 = find_low_rank_decomp(3, 2, 3, 18, max_iters=10, n_restarts=1)
            for i in range(3):
                for j in range(2):
                    for c in range(r3):
                        U_full[i * m + j][col + c] = U3[i * 2 + j][c]
            for j in range(2):
                for k in range(3):
                    for c in range(r3):
                        V_full[j * p + (k + 2)][col + c] = V3[j * 3 + k][c]
            for k in range(3):
                for i in range(3):
                    for c in range(r3):
                        W_full[(k + 2) * n + i][col + c] = W3[k * 3 + i][c]
            col += r3
            
            # Sub-problem 4: 3x3x3, rank r4
            for i in range(3):
                for j in range(3):
                    for c in range(r4):
                        U_full[i * m + (j + 2)][col + c] = U4[i * 3 + j][c]
            for j in range(3):
                for k in range(3):
                    for c in range(r4):
                        V_full[(j + 2) * p + (k + 2)][col + c] = V4[j * 3 + k][c]
            for k in range(3):
                for i in range(3):
                    for c in range(r4):
                        W_full[(k + 2) * n + i][col + c] = W4[k * 3 + i][c]
            col += r4
            
            # Verify
            if verify_decomp(U_full, V_full, W_full, total_rank):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_total_rank = total_rank
    
    if best_total_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        U, V, W = build_schoolbook()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
