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
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Expected T[i,j,k] = 1
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=tol)

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

    def search_lower_rank(R, max_restarts=5, max_nfev=500):
        """Try to find a rank-R decomposition via optimization. Limited budget."""
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
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            
            x0 = np.random.randn(total_params) * 1.0
            
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=max_nfev, 
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_sol = res.x
                    if best_cost < 1e-12:
                        break
            except Exception:
                continue
        
        if best_sol is not None and best_cost < 1e-10:
            U = best_sol[:4*R].reshape(4, R)
            V = best_sol[4*R:4*R+18*R].reshape(18, R)
            W = best_sol[4*R+18*R:4*R+36*R].reshape(18, R)
            
            # Snap to half-integers and verify
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            if verify_decomp(U_snapped, V_snapped, W_snapped, n, m, p):
                return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
            
            # Try integer snap
            U_snapped = np.round(U)
            V_snapped = np.round(V)
            W_snapped = np.round(W)
            if verify_decomp(U_snapped, V_snapped, W_snapped, n, m, p):
                return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        
        return None

    # Attempt Rank 31 with a very short time budget (5s)
    if time.time() < deadline - 5.0:
        search_deadline = time.time() + 5.0
        # Monkey patch deadline for the search? No, just check inside.
        # The search function uses the global deadline. 
        # To strictly limit it, we can just rely on the few restarts and low nfev.
        result = search_lower_rank(31, max_restarts=5, max_nfev=200)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-32 deterministic construction
    U, V, W = build_rank32()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
