import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    # 2x3x4 matrix multiplication
    target_rank = n * m * p  # 24
    
    # Target tensor T[i, j, k]
    # The bilinear form is C_ik = sum_j A_ij B_jk
    # In tensor decomposition T = sum_r U_r \otimes V_r \otimes W_r
    # T_{ijk} = 1 if the indices correspond to a valid term in the sum.
    # Specifically, for the standard basis, T_{ijk} = 1 for all i, j, k.
    # Wait, let's verify.
    # C = AB.
    # C_{ik} = \sum_j A_{ij} B_{jk}.
    # The tensor T is defined such that \langle T, A \otimes B \rangle = C.
    # Or rather, T is the structure tensor.
    # T_{ijk} A_{ij} B_{jk} = C_{ik}.
    # So T_{ijk} = 1 for all valid i, j, k.
    T = np.ones((n, m, p))
    
    def verify_decomposition(U, V, W, R):
        """Verify if U, V, W form a valid decomposition of T."""
        U3 = np.array(U).reshape(n, m, R)
        V3 = np.array(V).reshape(m, p, R)
        W3 = np.array(W).reshape(p, n, R)
        # A_approx[i, j, k] = sum_c U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
        # Note: W index is k*n+i, so W3 is (p, n, R) -> k, i, c
        A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(A_approx, T, atol=1e-9)

    def try_rank(R, max_restarts=10, max_nfev=50):
        """Try to find a rank-R decomposition using least squares optimization."""
        
        def unpack(params):
            U = params[:R * n * m].reshape(R, n * m).T
            V = params[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
            W = params[R * (n * m + m * p) :].reshape(R, p * n).T
            return U, V, W

        def residuals(params):
            U, V, W = unpack(params)
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return (T - A_approx).flatten()

        num_params = R * (n * m + m * p + p * n)
        
        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            # Initialize with random values
            # Since we suspect integer/half-integer solutions, we can initialize
            # with random choices from a small set of such values.
            init_params = np.random.choice([-1, -0.5, 0, 0.5, 1], size=num_params)
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residuals, init_params, method='trf', max_nfev=max_nfev)
                
                if sol.success or sol.cost < 1e-8:
                    # Check if the solution is close to a valid integer/half-integer decomposition
                    # Try rounding to nearest 0.5
                    rounded_params = np.round(sol.x * 2) / 2.0
                    res_check = residuals(rounded_params)
                    if np.linalg.norm(res_check) < 1e-6:
                        U, V, W = unpack(rounded_params)
                        if verify_decomposition(U, V, W, R):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": R}
                    
                    # Also check if the raw solution is exact enough
                    # The problem statement says "integer/half-integer entries".
                    # So we must round.
                    
            except Exception:
                pass
                
        return None

    # Search for lower ranks
    # Start from target_rank - 1 down to a reasonable limit.
    # If we find a valid one, return it.
    # The rank of 2x3x4 matrix multiplication is known to be 20? Or 21?
    # Let's try from 23 down to 18.
    
    for R in range(target_rank - 1, 17, -1):
        if time.time() - start_time > max_time:
            break
        res = try_rank(R)
        if res:
            return res

    # Fallback: Schoolbook decomposition (rank 24)
    rank = target_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
