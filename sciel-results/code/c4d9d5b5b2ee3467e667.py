"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        r = len(U[0])
        # Check: for all i,k: sum_j A[i,j]B[j,k] = sum_t W[k*n+i,t] * (sum_j U[i*m+j,t]*A[i,j]) * (sum_j V[j*p+k,t]*B[j,k])
        # This must hold for all A,B, so check tensor identity:
        # sum_t W[k*n+i,t] * U[i_a*m+j_a,t] * V[j_b*p+k_b,t] = delta(i_a,i)*delta(j_a,j_b)*delta(k_b,k)
        for i in range(n):
            for k in range(p):
                for i_a in range(n):
                    for j_a in range(m):
                        for j_b in range(m):
                            for k_b in range(p):
                                lhs = 0.0
                                for t in range(r):
                                    lhs += W[k*n+i][t] * U[i_a*m+j_a][t] * V[j_b*p+k_b][t]
                                rhs = 1.0 if (i_a == i and j_a == j_b and k_b == k) else 0.0
                                if abs(lhs - rhs) > 1e-10:
                                    return False
        return True
    
    def solve_rank(r, seed=42):
        """Try to find a rank-r decomposition using least squares on quadratic system."""
        rng = np.random.RandomState(seed)
        
        # Variables: U (n*m x r), V (m*p x r), W (p*n x r)
        # Total: (n*m + m*p + p*n) * r = 26*r for our case
        total_vars = (n * m + m * p + p * n) * r
        
        # Build the system of equations
        # For each (i, k, i_a, j_a, j_b, k_b):
        # sum_t W[k*n+i, t] * U[i_a*m+j_a, t] * V[j_b*p+k_b, t] = delta(i_a,i)*delta(j_a,j_b)*delta(k_b,k)
        
        # This is a system of quadratic equations. We'll use scipy's approach.
        # Flatten variables: [U_flat, V_flat, W_flat]
        
        def residuals(x):
            U_flat = x[:n*m*r].reshape(n*m, r)
            V_flat = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_flat = x[n*m*r + m*p*r :].reshape(p*n, r)
            
            res = []
            for i in range(n):
                for k in range(p):
                    for i_a in range(n):
                        for j_a in range(m):
                            for j_b in range(m):
                                for k_b in range(p):
                                    val = 0.0
                                    for t in range(r):
                                        val += W_flat[k*n+i, t] * U_flat[i_a*m+j_a, t] * V_flat[j_b*p+k_b, t]
                                    target = 1.0 if (i_a == i and j_a == j_b and k_b == k) else 0.0
                                    res.append(val - target)
            return np.array(res)
        
        # Initial guess: random
        x0 = rng.randn(total_vars) * 0.1
        
        from scipy.optimize import least_squares
        try:
            result = least_squares(residuals, x0, method='lm', xtol=1e-12, ftol=1e-12, max_nfev=10000)
            if result.success or result.cost < 1e-10:
                x = result.x
                U = x[:n*m*r].reshape(n*m, r).tolist()
                V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r).tolist()
                W = x[n*m*r + m*p*r :].reshape(p*n, r).tolist()
                if verify(U, V, W):
                    return U, V, W
        except Exception:
            pass
        return None
    
    # Try ranks from low to high
    for r in range(14, 25):
        if time.time() > deadline - 30:
            break
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            result = solve_rank(r, seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
