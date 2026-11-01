"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def verify_np(U, V, W):
        """Vectorized verification using numpy."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
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

    def build_rank32():
        rank = 32
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            U[2][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
            col += 1

            U[0][col] = 1
            V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            U[3][col] = 1
            V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
            col += 1

            U[0][col] = 1; U[1][col] = 1
            V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
            col += 1

            U[2][col] = 1; U[0][col] = -1
            V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1

            U[1][col] = 1; U[3][col] = -1
            V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            U[0][col] = 1; V[0 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_trivial(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U, "V": V, "W": W}

    def try_rank31_search():
        """Try to find a rank-31 decomposition by relaxing one trivial term."""
        if time.time() > deadline - 30:
            return None
        
        # Start from rank-32 solution
        r32 = build_rank32()
        U32 = np.array(r32["U"], dtype=np.float64)
        V32 = np.array(r32["V"], dtype=np.float64)
        W32 = np.array(r32["W"], dtype=np.float64)
        
        # The last 4 terms (cols 28-31) are trivial for k=8
        # Try to see if we can eliminate one by adjusting others
        # Strategy: fix 31 columns, try to solve for the 31st
        # Actually, try a local search: perturb the rank-32 solution and project to rank 31
        
        # Alternative: try to find if any single column can be expressed as a combination
        # of the others in a way that preserves the tensor equation
        
        # Let's try: for each of the 4 trivial columns, try to remove it and solve
        # for adjustments to the other 31 columns
        
        best_result = None
        for remove_col in range(28, 32):
            if time.time() > deadline - 25:
                break
            
            # Keep all columns except remove_col
            keep_cols = [c for c in range(32) if c != remove_col]
            U_keep = U32[:, keep_cols]
            V_keep = V32[:, keep_cols]
            W_keep = W32[:, keep_cols]
            
            # Now we need to find if we can adjust these 31 columns to still satisfy
            # all 36 equations (2*2*9 = 36)
            # This is an underdetermined system (31*3 = 93 vars, 36 equations)
            # but nonlinear. Let's try a local search.
            
            # Flatten the 31 columns into a vector
            x0 = np.concatenate([U_keep.flatten(), V_keep.flatten(), W_keep.flatten()])
            
            def objective(x):
                if time.time() > deadline - 20:
                    return 1e10
                U_mat = x[:4*31].reshape(4, 31)
                V_mat = x[4*31:4*31+18*31].reshape(18, 31)
                W_mat = x[4*31+18*31:].reshape(18, 31)
                
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                            residual += (total - 1.0) ** 2
                return residual
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < 1e-10:
                    U_mat = res.x[:4*31].reshape(4, 31)
                    V_mat = res.x[4*31:4*31+18*31].reshape(18, 31)
                    W_mat = res.x[4*31+18*31:].reshape(18, 31)
                    
                    # Convert to list format
                    U_list = [[float(U_mat[i, c]) for c in range(31)] for i in range(4)]
                    V_list = [[float(V_mat[i, c]) for c in range(31)] for i in range(18)]
                    W_list = [[float(W_mat[i, c]) for c in range(31)] for i in range(18)]
                    
                    if verify_np(U_list, V_list, W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        break
            except Exception:
                pass
        
        return best_result

    # First, try the known rank-32 solution
    result = build_rank32()
    
    if verify(result["U"], result["V"], result["W"]):
        # Try to improve to rank 31
        r31 = try_rank31_search()
        if r31 is not None:
            return r31
        return result
    
    return schoolbook()
# EVOLVE-BLOCK-END
