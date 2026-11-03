import time
import numpy as np
from itertools import combinations
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s

    # 1. The known rank-17 solution
    # Indices: U is (n*m, R) -> (6, R), V is (m*p, R) -> (9, R), W is (p*n, R) -> (6, R)
    # W index convention: W[k*n + i, r]
    
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    # Strassen for 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 -= M1, C10 -= M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B11-B10) -> C01 += M3
    # M4 = A11(B01-B00) -> C11 += M4
    # M5 = (A00+A01)B11 -> C00 += M5
    # M6 = (A10-A00)(B00+B01) -> C10 -= M6
    # M7 = (A01-A11)(B10+B11) -> C01 -= M7
    
    # Let's map the rank-17 provided in the prompt to ensure consistency.
    # The prompt's rank-17:
    # r=0: U[0,0]=1, U[4,0]=1 (A00, A11). V[0,0]=1, V[4,0]=1 (B00, B11). W[0,0]=1 (C00), W[3,0]=1 (C10? No, W index k*n+i. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3).
    # Wait, let's look at the prompt's W indices.
    # W[0,0]=1, W[3,0]=1.
    # W index = k*n + i. n=2.
    # W[0] -> k=0, i=0.
    # W[3] -> k=1, i=1.
    # So r=0 contributes to C00 and C11. This matches M1 = (A00+A11)(B00+B11) contributing to C00 and C11.
    # But Strassen M1 contributes to C00 (+), C01 (-), C10 (-), C11 (+).
    # The prompt's r=0 only has W[0] and W[3]. It misses C01 and C10.
    # Let's check r=1, r=2, etc.
    # r=1: U[3,1]=1 (A10), U[4,1]=1 (A11). V[0,1]=1 (B00). W[1,1]=1 (C01? k=0, i=1 -> 1), W[3,1]=-1 (C11? k=1, i=1 -> 3).
    # This is M2 = (A10+A11)B00. Contributes to C01 and C11?
    # Standard Strassen: M2 = (A10+A11)B00 -> C10 += M2.
    # Here it contributes to C01 and C11. This is not standard Strassen indexing or I am misinterpreting.
    # Let's just trust the provided rank-17 matrices are valid and use them as the base.

    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1

    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1

    # Verify rank-17 validity
    def verify(U, V, W):
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Must be 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U17, V17, W17):
        # Fallback to schoolbook rank-18 if rank-17 is broken (should not happen)
        U18 = np.zeros((6, 18)); V18 = np.zeros((9, 18)); W18 = np.zeros((6, 18))
        idx = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U18[i*3+j, idx] = 1
                    V18[j*3+k, idx] = 1
                    W18[k*2+i, idx] = 1
                    idx += 1
        return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}

    # 2. Strategy: Try to eliminate one rank term to get rank-16.
    # We have 17 terms. We want to see if we can merge two terms or eliminate one.
    # A rank-1 term r contributes to a set of (i,j,k) triples where U[i,j,r] V[j,k,r] W[k,i,r] != 0.
    # If we can find two terms r1, r2 such that their contributions can be combined into a single term,
    # or if one term is redundant, we can reduce rank.
    
    # However, the inspiration suggests using the 3x3x3 rank-23 decomposition restricted to 2x3x3.
    # But we don't have that code. We have the rank-17 code.
    # Let's try a local search / elimination approach on the rank-17 solution.
    
    # We can try to set one of the 17 columns to zero and see if the remaining 16 can be adjusted
    # to still satisfy the 18 equations. This is a non-linear system.
    # Alternatively, we can try to find a linear combination of two columns that results in a
    # "sparser" or "mergable" term.
    
    # Let's try a simpler heuristic: 
    # For each pair of columns (r1, r2), try to merge them.
    # Merging means finding new U', V', W' for a single column such that
    # U' V' W' approximates U1 V1 W1 + U2 V2 W2.
    # This is generally impossible exactly unless the terms share structure.
    
    # Given the time limit and the difficulty of exact rank reduction via optimization,
    # and the fact that the current score is 1.0588 (which corresponds to rank 17),
    # we should focus on ensuring the rank-17 solution is returned quickly and reliably.
    
    # Is there a known rank-16 for 2x3x3?
    # Lower bound for 2x3x3 is 16.
    # If rank 16 exists, it would be a significant improvement.
    # However, finding it from scratch or via simple modification of rank-17 is non-trivial.
    
    # Let's attempt a quick randomized search for a rank-16 solution initialized near the rank-17.
    # We will use scipy.optimize.minimize on a subset of variables or a penalty method.
    
    # To save time, we will only do this if we have plenty of time left.
    # The rank-17 construction is instantaneous.
    
    if time.time() > deadline - 60:
        # Not enough time for search, return rank-17
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

    # Try to find rank-16
    # We will try to eliminate each of the 17 terms one by one.
    # For each elimination, we fix the other 16 terms and try to solve for the missing contributions?
    # No, that's not how it works. We need to adjust ALL 16 terms.
    
    # Let's try a different approach: 
    # Use the rank-17 solution as a starting point for a rank-16 optimization.
    # We will randomly drop one column and then optimize the remaining 16 columns.
    
    best_rank = 17
    best_U = U17
    best_V = V17
    best_W = W17
    
    # Try dropping each column
    for drop_col in range(17):
        if time.time() > deadline - 10:
            break
            
        U_init = np.delete(U17, drop_col, axis=1)
        V_init = np.delete(V17, drop_col, axis=1)
        W_init = np.delete(W17, drop_col, axis=1)
        
        # We want to minimize sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
        # This is a non-convex problem.
        # We will use a few random restarts with small perturbations.
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            # Perturb the initial guess
            scale = 0.1 * (attempt + 1)
            U_curr = U_init + np.random.randn(6, 16) * scale
            V_curr = V_init + np.random.randn(9, 16) * scale
            W_curr = W_init + np.random.randn(6, 16) * scale
            
            # Simple gradient descent or Nelder-Mead?
            # Nelder-Mead is robust for small dimensions.
            # Dimensions: 6*16 + 9*16 + 6*16 = 96 + 144 + 96 = 336.
            # Nelder-Mead might be slow.
            # Let's try scipy.optimize.minimize with L-BFGS-B.
            
            def objective(x):
                U = x[:96].reshape(6, 16)
                V = x[96:240].reshape(9, 16)
                W = x[240:336].reshape(6, 16)
                
                # Compute tensor T
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # We can use einsum
                # U shape (6, 16), V (9, 16), W (6, 16)
                # T[i, j, k] = np.einsum('ir,jr,kr->ijk', U, V, W) 
                # Wait, indices:
                # U[i*m+j, r] -> i is 0..1, j is 0..2. U row is i*3+j.
                # V[j*p+k, r] -> j is 0..2, k is 0..2. V row is j*3+k.
                # W[k*n+i, r] -> k is 0..2, i is 0..1. W row is k*2+i.
                
                # Let's reshape to 3D tensors
                U_3d = U.reshape(2, 3, 16)
                V_3d = V.reshape(3, 3, 16)
                W_3d = W.reshape(3, 2, 16)
                
                # T[i, j, k] = sum_r U_3d[i, j, r] V_3d[j, k, r] W_3d[k, i, r]
                # This is a bit tricky with einsum directly because of the index mapping.
                # Let's just loop or use matmul tricks.
                # For small sizes, a loop is fast enough.
                
                err = 0.0
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            val = 0.0
                            for r in range(16):
                                val += U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
                            err += (val - 1.0) ** 2
                return err
            
            x0 = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < 1e-6:
                    # Check if it's exactly integer/half-integer? 
                    # The task says "integer/half-integer entries".
                    # We should round and verify.
                    U_sol = res.x[:96].reshape(6, 16)
                    V_sol = res.x[96:240].reshape(9, 16)
                    W_sol = res.x[240:336].reshape(6, 16)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    if verify(U_round, V_round, W_round):
                        best_rank = 16
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    if best_rank == 16:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
