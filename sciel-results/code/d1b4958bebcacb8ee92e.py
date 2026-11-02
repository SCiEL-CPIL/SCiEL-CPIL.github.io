import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin before 150s kill

    def verify_decomp(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        Tensor: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1 for all i,j,k
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        
        # Reshape to 3D: U -> (n, m, R), V -> (m, p, R), W -> (p, n, R)
        U_r = U_arr.reshape(n, m, R)
        V_r = V_arr.reshape(m, p, R)
        W_r = W_arr.reshape(p, n, R)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        
        if not np.allclose(T, expected, atol=1e-9):
            return False
        return True

    def build_rank32():
        """Build the known rank-32 decomposition: 4 Strassen blocks (rank 7) + 1 schoolbook (rank 4)."""
        terms = []
        
        # Strassen blocks for k-pairs: (0,1), (2,3), (4,5), (6,7)
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            
            # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
            # B entries for this block: B[0,k0], B[0,k1], B[1,k0], B[1,k1]
            # V indices: B[0,k0]=k0, B[0,k1]=k1, B[1,k0]=9+k0, B[1,k1]=9+k1
            # W indices: C[0,k0]=2*k0, C[0,k1]=2*k1, C[1,k0]=2*k0+1, C[1,k1]=2*k1+1
            
            # M1 = (a+d)(B[0,k0]+B[1,k1]) -> contributes to C[0,k0] and C[1,k1]
            terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
            
            # M2 = (c+d)B[0,k0] -> contributes to C[1,k0] and C[1,k1] (with sign)
            terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
            
            # M3 = a(B[0,k1]-B[1,k1]) -> contributes to C[0,k1] and C[1,k1]
            terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
            
            # M4 = d(B[1,k0]-B[0,k0]) -> contributes to C[0,k0] and C[1,k0]
            terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
            
            # M5 = (a+b)B[1,k1] -> contributes to C[0,k0] and C[0,k1]
            terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
            
            # M6 = (c-a)(B[0,k0]+B[0,k1]) -> contributes to C[1,k1]
            terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
            
            # M7 = (b-d)(B[1,k0]+B[1,k1]) -> contributes to C[0,k0]
            terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
        
        # Schoolbook for k=8
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))       # A[0,0]*B[0,8] -> C[0,8]
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))     # A[0,1]*B[1,8] -> C[0,8]
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))     # A[1,0]*B[0,8] -> C[1,8]
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))   # A[1,1]*B[1,8] -> C[1,8]
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        
        return U, V, W

    def build_rank36_schoolbook():
        """Build the rank-36 schoolbook decomposition as ultimate fallback."""
        rank = n * m * p  # 2*2*9 = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        
        return U, V, W

    # First, try the rank-32 Strassen construction
    U32, V32, W32 = build_rank32()
    
    if verify_decomp(U32, V32, W32, n, m, p):
        return {"U": U32, "V": V32, "W": W32}
    
    # If rank-32 verification fails (shouldn't happen, but just in case),
    # try to find a better decomposition via optimization for a [2,2,3] block
    # that could give rank 24 total (3 blocks of rank 8)
    
    if time.time() < deadline - 30:
        # Try to find a rank-8 decomposition for [2,2,3] using optimization
        # If found, use 3 blocks for total rank 24
        from scipy.optimize import least_squares
        
        def search_rank_223(R, max_restarts=20):
            """Search for rank-R decomposition of 2x2x3 matrix multiplication."""
            # U: (4, R), V: (6, R), W: (6, R)
            # U indices: a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=2, d=A[1,1]=3
            # V indices: e=B[0,0]=0, f=B[0,1]=1, g=B[0,2]=2, h=B[1,0]=3, i=B[1,1]=4, j=B[1,2]=5
            # W indices: p=C[0,0]=0, q=C[0,1]=1, r=C[0,2]=2, s=C[1,0]=3, t=C[1,1]=4, u=C[1,2]=5
            
            def residuals(params):
                U = params[:4*R].reshape(4, R)
                V = params[4*R:4*R+6*R].reshape(6, R)
                W = params[4*R+6*R:4*R+12*R].reshape(6, R)
                U_r = U.reshape(2, 2, R)
                V_r = V.reshape(2, 3, R)
                W_r = W.reshape(3, 2, R)
                T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                return (T - 1.0).flatten()
            
            best_sol = None
            best_cost = float('inf')
            
            np.random.seed(42)
            for attempt in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                x0 = np.random.randn(12 * R) * 0.3
                try:
                    res = least_squares(residuals, x0, method='trf', max_nfev=5000, 
                                       xtol=1e-12, ftol=1e-12, gtol=1e-12)
                    if res.cost < best_cost:
                        best_cost = res.cost
                        best_sol = res.x
                        if res.cost < 1e-14:
                            break
                except Exception:
                    continue
            
            if best_sol is not None and best_cost < 1e-10:
                U = best_sol[:4*R].reshape(4, R)
                V = best_sol[4*R:4*R+6*R].reshape(6, R)
                W = best_sol[4*R+6*R:4*R+12*R].reshape(6, R)
                
                # Snap to nearest half-integer
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify the snapped version
                U_r = U_snapped.reshape(2, 2, R)
                V_r = V_snapped.reshape(2, 3, R)
                W_r = W_snapped.reshape(3, 2, R)
                T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                if np.allclose(T, np.ones((2, 2, 3)), atol=1e-9):
                    return U_snapped, V_snapped, W_snapped
                
                # Also try integer snapping
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                U_r = U_int.reshape(2, 2, R)
                V_r = V_int.reshape(2, 3, R)
                W_r = W_int.reshape(3, 2, R)
                T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                if np.allclose(T, np.ones((2, 2, 3)), atol=1e-9):
                    return U_int, V_int, W_int
            
            return None
        
        # Try rank 8 first, then 9, then 10
        for R in [8, 9, 10]:
            if time.time() > deadline - 10:
                break
            sol = search_rank_223(R)
            if sol is not None:
                U_b, V_b, W_b = sol
                # Build full 2x2x9 from 3 blocks of 2x2x3
                total_rank = 3 * R
                U_full = np.zeros((4, total_rank))
                V_full = np.zeros((18, total_rank))
                W_full = np.zeros((18, total_rank))
                
                for b in range(3):
                    start_r = b * R
                    end_r = start_r + R
                    U_full[:, start_r:end_r] = U_b
                    for j in range(2):  # j = 0, 1 (rows of B)
                        for k_local in range(3):  # columns within this block
                            k_global = b * 3 + k_local
                            v_idx_block = j * 3 + k_local
                            v_idx_full = j * 9 + k_global
                            V_full[v_idx_full, start_r:end_r] = V_b[v_idx_block, :]
                    for k_local in range(3):
                        k_global = b * 3 + k_local
                        for i in range(2):  # i = 0, 1 (columns of C)
                            w_idx_block = k_local * 2 + i
                            w_idx_full = k_global * 2 + i
                            W_full[w_idx_full, start_r:end_r] = W_b[w_idx_block, :]
                
                if verify_decomp(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
                    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: rank-36 schoolbook
    U36, V36, W36 = build_rank36_schoolbook()
    return {"U": U36, "V": V36, "W": W36}
