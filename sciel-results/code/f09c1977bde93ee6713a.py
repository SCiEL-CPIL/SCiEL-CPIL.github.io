import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T
    # T[a, b, c] = 1 if a, b, c correspond to i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    T_flat = T.flatten()

    def verify(U, V, W):
        res = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(res, T, atol=1e-9)

    def try_rank_als(R, num_restarts=10):
        for restart in range(num_restarts):
            # Random initialization
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Normalize to prevent scale issues
            U = U / np.linalg.norm(U, axis=0, keepdims=True)
            V = V / np.linalg.norm(V, axis=0, keepdims=True)
            W = W / np.linalg.norm(W, axis=0, keepdims=True)
            
            for iteration in range(100):
                # Update U
                # For each a, solve Y U_a^T = X_a^T
                # Y is (mp*pn, R), columns are vec(V[:,r] * W[:,r])? No, vec(V[:,r] W[:,r]?)
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Let Z_r = V[:,r] * W[:,r] (element-wise product, shape mp x pn)
                # Then T[a] = sum_r U[a,r] Z_r
                # Vectorize T[a] to x_a (size mp*pn)
                # Vectorize Z_r to y_r (size mp*pn)
                # x_a = sum_r U[a,r] y_r
                # x = Y U^T
                # U = (Y^T Y)^{-1} Y^T X
                
                # Build Y for V,W update? No, for U update, V and W are fixed.
                # y_r = (V[:,r] * W[:,r]).flatten()
                Y_U = np.zeros((mp * pn, R))
                for r in range(R):
                    Y_U[:, r] = (V[:, r] * W[:, r]).flatten()
                
                # X is T reshaped to (nm, mp*pn)
                X_U = T.reshape(nm, -1)
                
                # Solve X_U = Y_U U^T  =>  U^T = (Y_U^T Y_U)^{-1} Y_U^T X_U^T
                # U = (Y_U^T Y_U)^{-1} Y_U^T X_U^T ... wait, dimensions.
                # X_U is (nm, mp*pn). Y_U is (mp*pn, R).
                # We want U (nm, R).
                # X_U = Y_U U^T? No.
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
                # x_a (mp*pn) = sum_r U[a,r] y_r (mp*pn).
                # So X_U = Y_U U^T? No.
                # If U is (nm, R), then U^T is (R, nm).
                # Y_U (mp*pn, R) * U^T (R, nm) -> (mp*pn, nm).
                # We want (nm, mp*pn).
                # So X_U^T = Y_U U^T.
                # U = (Y_U^T Y_U)^{-1} Y_U^T X_U^T.
                # Note: X_U^T is (mp*pn, nm).
                
                try:
                    A = Y_U.T @ Y_U
                    B = Y_U.T @ X_U.T
                    U_new = np.linalg.solve(A, B).T
                except np.linalg.LinAlgError:
                    U_new = np.linalg.pinv(Y_U) @ X_U.T
                
                # Update V
                Y_V = np.zeros((pn * nm, R))
                for r in range(R):
                    Y_V[:, r] = (W[:, r] * U[:, r]).flatten() # Wait, T[a,b,c] symmetric in structure?
                    # T[a,b,c] = sum U[a,r] V[b,r] W[c,r]
                    # For V update:
                    # T[:,b,:] = sum_r V[b,r] (U[:,r] W[:,r])?
                    # x_b (pn*nm) = sum_r V[b,r] z_r
                    # z_r = (U[:,r] * W[:,r])? No.
                    # T[a,b,c] = V[b,r] * (U[a,r] W[c,r]).
                    # Vector over a,c.
                    # z_r = (U[:,r] * W[:,r])? No, U is (nm, 1) effectively for r, W is (pn, 1).
                    # The tensor slice T[:,b,:] has indices a,c.
                    # T[a,b,c] = V[b,r] U[a,r] W[c,r].
                    # So for fixed b, vector x_b over (a,c) is sum_r V[b,r] (U[:,r] outer W[:,r]).flatten()
                    # Yes.
                    Y_V[:, r] = (np.outer(U[:, r], W[:, r])).flatten()
                X_V = T.transpose(1, 0, 2).reshape(mp, -1) # (mp, nm*pn)
                
                try:
                    A = Y_V.T @ Y_V
                    B = Y_V.T @ X_V.T
                    V_new = np.linalg.solve(A, B).T
                except np.linalg.LinAlgError:
                    V_new = np.linalg.pinv(Y_V) @ X_V.T

                # Update W
                Y_W = np.zeros((nm * mp, R))
                for r in range(R):
                    Y_W[:, r] = (np.outer(U[:, r], V[:, r])).flatten()
                X_W = T.transpose(2, 0, 1).reshape(pn, -1) # (pn, nm*mp)
                
                try:
                    A = Y_W.T @ Y_W
                    B = Y_W.T @ X_W.T
                    W_new = np.linalg.solve(A, B).T
                except np.linalg.LinAlgError:
                    W_new = np.linalg.pinv(Y_W) @ X_W.T

                # Check convergence
                if np.linalg.norm(U_new - U) < 1e-6 and np.linalg.norm(V_new - V) < 1e-6 and np.linalg.norm(W_new - W) < 1e-6:
                    U, V, W = U_new, V_new, W_new
                    break
                
                U, V, W = U_new, V_new, W_new

            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None

    # Try ranks from 19 to 27
    for R in range(19, 28):
        result = try_rank_als(R, num_restarts=10)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    rank = n * m * p
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
# EVOLVE-BLOCK-END
