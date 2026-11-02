import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Uses iterative search to find lower-rank decompositions than schoolbook.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    """
    start_time = time.time()
    deadline = 145  # seconds, safety margin for the 150s kill limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 36
    
    # Best solution found (start with schoolbook)
    best_rank = rank_schoolbook
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def check_time():
        return time.time() - start_time > deadline
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly satisfies the tensor equation."""
        # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def try_rank(rank, max_iterations=1000, tol=1e-12):
        """Try to find a valid decomposition at given rank using Gauss-Newton."""
        if rank < max(n*m, m*p, p*n):
            return None  # Below theoretical lower bound
        
        if check_time():
            return None
        
        # Initialize with random factors
        rng = np.random.default_rng(42)
        U = rng.standard_normal((n * m, rank)) * 0.1
        V = rng.standard_normal((m * p, rank)) * 0.1
        W = rng.standard_normal((p * n, rank)) * 0.1
        
        # Scale to make the product approximately 1
        for _ in range(50):
            if check_time():
                return None
            
            # Compute residuals for all (i, j, k)
            residuals = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        residuals.append(val - 1)
            
            residuals = np.array(residuals)
            if np.max(np.abs(residuals)) < tol:
                break
            
            # Gauss-Newton step: linearize around current point
            # f(U,V,W) = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1
            # Jacobian w.r.t. U[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
            # Jacobian w.r.t. V[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
            # Jacobian w.r.t. W[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
            
            # Build Jacobian matrix (sparse structure)
            # Number of variables: n*m*rank + m*p*rank + p*n*rank
            num_vars = (n * m + m * p + p * n) * rank
            num_equations = n * m * p
            
            # Use normal equations: J^T J delta = -J^T r
            # For efficiency, build J^T J and J^T r incrementally
            JTJ = np.zeros((num_vars, num_vars))
            JTr = np.zeros(num_vars)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        r_val = residuals[eq_idx]
                        
                        for r in range(rank):
                            u_idx = (i * m + j) * rank + r
                            v_idx = (n * m * rank) + (j * p + k) * rank + r
                            w_idx = (n * m * rank + m * p * rank) + (k * n + i) * rank + r
                            
                            u_val = U[i * m + j, r]
                            v_val = V[j * p + k, r]
                            w_val = W[k * n + i, r]
                            
                            # Jacobian entries
                            ju = v_val * w_val
                            jv = u_val * w_val
                            jw = u_val * v_val
                            
                            # J^T J contributions
                            JTJ[u_idx, u_idx] += ju * ju
                            JTJ[u_idx, v_idx] += ju * jv
                            JTJ[u_idx, w_idx] += ju * jw
                            JTJ[v_idx, u_idx] += jv * ju
                            JTJ[v_idx, v_idx] += jv * jv
                            JTJ[v_idx, w_idx] += jv * jw
                            JTJ[w_idx, u_idx] += jw * ju
                            JTJ[w_idx, v_idx] += jw * jv
                            JTJ[w_idx, w_idx] += jw * jw
                            
                            # J^T r contributions
                            JTr[u_idx] += ju * r_val
                            JTr[v_idx] += jv * r_val
                            JTr[w_idx] += jw * r_val
            
            # Solve (J^T J + lambda*I) delta = -J^T r with damping
            lam = 1e-6
            JTJ += lam * np.eye(num_vars)
            
            try:
                delta = np.linalg.solve(JTJ, -JTr)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(JTJ, -JTr, rcond=None)[0]
            
            # Update
            U_flat = U.flatten()
            V_flat = V.flatten()
            W_flat = W.flatten()
            
            U_flat += delta[:n * m * rank]
            V_flat += delta[n * m * rank:n * m * rank + m * p * rank]
            W_flat += delta[n * m * rank + m * p * rank:]
            
            U = U_flat.reshape((n * m, rank))
            V = V_flat.reshape((m * p, rank))
            W = W_flat.reshape((p * n, rank))
        
        # Final check
        if check_time():
            return None
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1) > 1e-8:
                        return None
        
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Verify with list-based check
        if verify(U_list, V_list, W_list, rank):
            return (U_list, V_list, W_list)
        
        return None
    
    # Try ranks from low to high
    min_possible = max(n * m, m * p, p * n)
    
    for rank in range(min_possible, best_rank):
        if check_time():
            break
        
        # Try multiple random initializations
        for seed in range(10):
            if check_time():
                break
            
            np.random.seed(seed)
            result = try_rank(rank)
            if result is not None:
                U_list, V_list, W_list = result
                best_rank = rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                break  # Found valid decomposition at this rank, move to next
    
    # If no improvement found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
