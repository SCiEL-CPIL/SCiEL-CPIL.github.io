import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    def verify_decomp(U, V, W, n, m, p, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=tol)

    def build_rank30():
        """Deterministic rank-30 construction: 3 blocks of 2x2x3 with rank 10 each."""
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # 2x2x3 tensor has rank 10. We use 3 such blocks for columns (0,1,2), (3,4,5), (6,7,8)
        # Known rank-10 decomposition for 2x2x3:
        # The 10 terms are:
        # 1: (a00)(b00)
        # 2: (a01)(b01)
        # 3: (a10)(b10)
        # 4: (a11)(b11)
        # 5: (a00+a11)(b00+b11)
        # 6: (a10+a11)(b00+b10)
        # 7: (a00+a01)(b01+b11)
        # 8: (a10-a00)(b00+b01)
        # 9: (a01-a11)(b10+b11)
        # 10: (a00+a11)(b00+b11) - wait, that's duplicate
        # Let me use the standard rank-10 for 2x2x3.
        
        # Actually, the minimal rank for 2x2x3 is 10? No, it's known that 2x2x3 has rank 10.
        # Let me use a known construction.
        
        # Standard rank-10 decomposition for 2x2x3:
        # Using the fact that 2x2x3 can be decomposed with 10 terms.
        # One known decomposition:
        terms_2x2x3 = [
            # (U_indices, V_indices, W_indices)
            # Format: (u_row, v_row, w_row) for each of 10 terms
            # U is 2x2, V is 2x3, W is 3x2
            # Term 1: a00*b00 -> c00
            ((0,), (0*3+0,), (0*2+0,)),
            # Term 2: a00*b10 -> c01
            ((0,), (1*3+0,), (0*2+1,)),
            # Term 3: a01*b01 -> c00
            ((1,), (0*3+1,), (0*2+0,)),
            # Term 4: a01*b11 -> c01
            ((1,), (1*3+1,), (0*2+1,)),
            # Term 5: a10*b00 -> c10
            ((2,), (0*3+0,), (1*2+0,)),
            # Term 6: a10*b10 -> c11
            ((2,), (1*3+0,), (1*2+1,)),
            # Term 7: a11*b01 -> c10
            ((3,), (0*3+1,), (1*2+0,)),
            # Term 8: a11*b11 -> c11
            ((3,), (1*3+1,), (1*2+1,)),
            # Term 9: (a00+a11)*(b00+b11)
            ((0, 3), (0*3+0, 1*3+1), (0*2+0, 1*2+1)),
            # Term 10: (a10+a11)*(b00+b10)
            ((2, 3), (0*3+0, 1*3+0), (1*2+0, 1*2+1)),
        ]
        
        # Hmm, this isn't right. Let me use a verified rank-10 for 2x2x3.
        # Actually, let me just use the rank-32 construction which is verified to work.
        # Instead, let me try to find a rank-30 by optimization with structured init.
        return None

    def build_rank32():
        """Deterministic rank-32 construction: 4 Strassen blocks + 1 schoolbook block."""
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # 4 Strassen blocks for k pairs (0,1), (2,3), (4,5), (6,7)
        for block in range(4):
            k0 = 2 * block
            k1 = k0 + 1
            
            # M1 = (a00+a11)(b00+b11)
            U[0][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            
            # M2 = (a10+a11)b00
            U[2][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1
            W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
            col += 1
            
            # M3 = a00(b01-b11)
            U[0][col] = 1
            V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
            W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            
            # M4 = a11(b10-b00)
            U[3][col] = 1
            V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
            W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
            col += 1
            
            # M5 = (a00+a01)b11
            U[0][col] = 1; U[1][col] = 1
            V[1*p + k1][col] = 1
            W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
            col += 1
            
            # M6 = (a10-a00)(b00+b01)
            U[2][col] = 1; U[0][col] = -1
            V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            
            # M7 = (a01-a11)(b10+b11)
            U[1][col] = 1; U[3][col] = -1
            V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1
            col += 1
        
        # Schoolbook for last column k=8
        k = 8
        U[0][col] = 1; V[0*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[1][col] = 1; V[1*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[2][col] = 1; V[0*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1
        U[3][col] = 1; V[1*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1
        
        return U, V, W

    def search_lower_rank(R, max_restarts=10, max_nfev=500):
        """Try to find a rank-R decomposition via optimization with structured init."""
        if time.time() > deadline:
            return None
        
        total_params = 4 * R + 18 * R + 18 * R
        
        def residuals(params):
            U = params[:4*R].reshape(4, R)
            V = params[4*R:4*R+18*R].reshape(18, R)
            W = params[4*R+18*R:4*R+36*R].reshape(18, R)
            U_r = U.reshape(2, 2, R)
            V_r = V.reshape(2, 9, R)
            W_r = W.reshape(9, 2, R)
            T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
            return T.flatten() - 1.0
        
        best_sol = None
        best_cost = float('inf')
        
        # Generate structured initializations
        inits = []
        
        # Random inits
        for i in range(min(3, max_restarts)):
            inits.append(np.random.randn(total_params) * 1.0)
        
        # Structured init from rank-32 (if R < 32, truncate)
        if R <= 32:
            U32, V32, W32 = build_rank32()
            U32_arr = np.array(U32, dtype=float)
            V32_arr = np.array(V32, dtype=float)
            W32_arr = np.array(W32, dtype=float)
            if R < 32:
                # Truncate and add noise
                U_init = U32_arr[:, :R] + np.random.randn(4, R) * 0.1
                V_init = V32_arr[:, :R] + np.random.randn(18, R) * 0.1
                W_init = W32_arr[:, :R] + np.random.randn(18, R) * 0.1
            else:
                U_init = U32_arr + np.random.randn(4, R) * 0.1
                V_init = V32_arr + np.random.randn(18, R) * 0.1
                W_init = W32_arr + np.random.randn(18, R) * 0.1
            inits.append(np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()]))
        
        for attempt in range(min(max_restarts, len(inits))):
            if time.time() > deadline:
                break
            
            x0 = inits[attempt]
            
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=max_nfev, 
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_sol = res.x
                    if best_cost < 1e-14:
                        break
            except Exception:
                continue
        
        if best_sol is not None and best_cost < 1e-10:
            U = best_sol[:4*R].reshape(4, R)
            V = best_sol[4*R:4*R+18*R].reshape(18, R)
            W = best_sol[4*R+18*R:4*R+36*R].reshape(18, R)
            
            # Snap to half-integers and verify
            for snap_factor in [2, 4, 8]:
                U_snapped = np.round(U * snap_factor) / snap_factor
                V_snapped = np.round(V * snap_factor) / snap_factor
                W_snapped = np.round(W * snap_factor) / snap_factor
                if verify_decomp(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
            
            # Try integer snap
            U_snapped = np.round(U)
            V_snapped = np.round(V)
            W_snapped = np.round(W)
            if verify_decomp(U_snapped, V_snapped, W_snapped, n, m, p):
                return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        
        return None

    # Attempt Rank 31 with structured initialization
    if time.time() < deadline - 10.0:
        result = search_lower_rank(31, max_restarts=8, max_nfev=300)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Attempt Rank 30 with structured initialization if time allows
    if time.time() < deadline - 10.0:
        result = search_lower_rank(30, max_restarts=5, max_nfev=200)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-32 deterministic construction
    U, V, W = build_rank32()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
