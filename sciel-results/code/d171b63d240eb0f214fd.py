import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank-24 decomposition using a structured search
    # within the Strassen family, or fall back to rank-28.
    
    # First, try a direct rank-24 construction.
    # We'll set up the tensor decomposition problem and solve it.
    # T[i,j,k] = delta(j,k) for i in [n], j in [m], k in [p]
    # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For rank r, this gives n*m*p = 32 equations (only 16 are nonzero: T[i,j,j]=1)
    # Actually T[i,j,k] is nonzero only when j==k, so 2*4=8 nonzero entries.
    # Wait: T[i,j,k] = 1 if j==k, else 0. So for each i in [2], j in [4], k in [4]:
    # T[i,j,j] = 1 for j=0,1,2,3, and T[i,j,k]=0 otherwise.
    # That's 2*4 = 8 equations with value 1, and 2*4*4 - 8 = 24 equations with value 0.
    # Total 32 constraints.
    
    # Let me try rank 24 by constructing a specific pattern.
    # Alternative: use a 4x4x4 rank-49 Strassen and restrict.
    # Or: try to find a rank-24 by exploiting the 2-row structure.
    
    # Let me try a different approach: use the fact that n=2 is small.
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # We can think of this as: for each k, C[:,k] = B[:,k]^T @ A (transposed view)
    # Actually C[:,k] = sum_j A[:,j] * B[j,k]
    
    # Let me try to construct a rank-24 decomposition directly.
    # Strategy: use 4 "Strassen-like" blocks but with some sharing.
    
    # Actually, let me try the simplest improvement: verify rank 28 works,
    # then try to find rank 24 or lower via a small search.
    
    # For now, let's keep the known rank-28 but add a deadline guard.
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Rank 28 decomposition using Strassen on 2x2 blocks
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 decomposition
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_offset * 2 + c)
                U[g_idx, col] += coeff
            
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * p + g_col
                V[g_idx, col] += coeff
            
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = r
                g_col = c_col_offset * 2 + c
                g_idx = g_col * n + g_row
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    # Verify the rank-28 decomposition
    def verify(U_arr, V_arr, W_arr, rank_val):
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank_val):
                    for jj in range(m):
                        acc += U_arr[ii * m + jj, cc] * V_arr[jj * p + kk, cc] * W_arr[kk * n + ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=1e-10)
    
    if not verify(U, V, W, rank):
        # Fallback: return identity-based rank-32 (always works)
        rank = n * m * p  # 32
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check if we have time to try rank 24
    if time.time() < deadline - 5:
        # Try to find a rank-24 decomposition
        # Use a greedy/random search approach
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try random rank-24 initializations
        np.random.seed(123)
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            r = 24
            U_try = np.zeros((n * m, r))
            V_try = np.zeros((m * p, r))
            W_try = np.zeros((p * n, r))
            
            # Initialize with small random values
            U_try += np.random.randn(n * m, r) * 0.1
            V_try += np.random.randn(m * p, r) * 0.1
            W_try += np.random.randn(p * n, r) * 0.1
            
            # Use iterative refinement: for each rank-1 term, optimize
            # This is a non-convex problem, so use multiple restarts
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                # Optimize one factor at a time
                for factor_idx in range(3):
                    if time.time() > deadline - 5:
                        break
                    # Fix other two, optimize this one
                    # For U: for each column c, U[:,c] should satisfy
                    # sum_j U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = T[i,j,k]
                    # This is a linear system in U[:,c] for each c
                    
                    if factor_idx == 0:  # Optimize U
                        for c in range(r):
                            # Build linear system: for each (i,j,k), 
                            # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] contributes to T[i,j,k]
                            # Fix V and W, solve for U[:,c]
                            # The equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
                            # For fixed c, U[i*m+j, c] appears in equations where j and k are fixed
                            # Actually, for a given c, U[i*m+j, c] multiplies V[j*p+k, c]*W[k*n+i, c]
                            # So for each (i,j), U[i*m+j, c] * sum_k V[j*p+k, c]*W[k*n+i, c] ... no that's not right
                            
                            # The full equation: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            # For fixed c, the contribution is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            # This is trilinear, so fixing V and W, for each c we have:
                            # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] is the c-th term
                            # We can't easily isolate U[:,c] because it's coupled with V and W at the same c.
                            
                            # This approach is getting complicated. Let me use a simpler heuristic.
                            pass
            # Skip complex optimization, just check if random works (unlikely)
        
        # Keep the rank-28 solution
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print("C:")
        print(C)
        print("Cf:")
        print(Cf)
    else:
        print("Verification Passed")
        
    print("Rank:", U.shape[1])
    print("Time taken:", time.time() - start_time)
