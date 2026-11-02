import time
import numpy as np
import scipy.optimize

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def construct():
    n, m, p = DIMENSIONS
    
    # Build target tensor
    target_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                target_tensor[u, v, w] = 1.0
    
    # Schoolbook decomposition (rank 18)
    rank_sb = n * m * p  # 18
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
                
    def to_half_int_list(matrix):
        scaled = np.rint(matrix * 2.0)
        result = scaled / 2.0
        return [[float(v) for v in row] for row in result]

    def validate(U, V, W):
        """Check if U, V, W decompose the tensor exactly."""
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(result, target_tensor, atol=1e-8)

    def residual(x, rank):
        """Residual for least squares optimization."""
        u_size = n * m * rank
        v_size = m * p * rank
        w_size = p * n * rank
        
        U_flat = x[:u_size]
        V_flat = x[u_size:u_size+v_size]
        W_flat = x[u_size+v_size:u_size+v_size+w_size]
        
        U = U_flat.reshape((n*m, rank))
        V = V_flat.reshape((m*p, rank))
        W = W_flat.reshape((p*n, rank))
        
        res = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            res += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        return (res - target_tensor).flatten()

    def try_rank(rank, max_attempts=30):
        """Try to find a decomposition of given rank."""
        if time.time() > START_TIME + TIME_LIMIT - 15:
            return None
            
        u_size = n * m * rank
        v_size = m * p * rank
        w_size = p * n * rank
        total_size = u_size + v_size + w_size
        
        for attempt in range(max_attempts):
            if time.time() > START_TIME + TIME_LIMIT - 5:
                break
                
            # Multiple initialization strategies
            if attempt % 3 == 0:
                # Random initialization
                x0 = np.random.randn(total_size) * 0.5
            elif attempt % 3 == 1:
                # Near-schoolbook initialization (truncated)
                x0 = np.zeros(total_size)
                # Copy first 'rank' columns of schoolbook
                U0 = U_sb[:, :rank].flatten()
                V0 = V_sb[:, :rank].flatten()
                W0 = W_sb[:, :rank].flatten()
                x0[:u_size] = U0
                x0[u_size:u_size+v_size] = V0
                x0[u_size+v_size:] = W0
            else:
                # Random with larger scale
                x0 = np.random.randn(total_size) * 1.0
                
            try:
                sol = scipy.optimize.least_squares(
                    residual, x0, args=(rank,), 
                    method='lm', max_nfev=2000,
                    xtol=1e-12, ftol=1e-12, gtol=1e-12
                )
            except Exception:
                continue
                
            if sol.cost < 1e-10:
                U_sol = sol.x[:u_size].reshape((n*m, rank))
                V_sol = sol.x[u_size:u_size+v_size].reshape((m*p, rank))
                W_sol = sol.x[u_size+v_size:u_size+v_size+w_size].reshape((p*n, rank))
                
                # Snap to half-integers
                U_snapped = np.rint(U_sol * 2) / 2
                V_snapped = np.rint(V_sol * 2) / 2
                W_snapped = np.rint(W_sol * 2) / 2
                
                if validate(U_snapped, V_snapped, W_snapped):
                    return (U_snapped, V_snapped, W_snapped)
                    
        return None

    # Try to find lower rank decompositions
    best_result = None
    for r in [17, 16]:
        if time.time() > START_TIME + TIME_LIMIT - 10:
            break
        res = try_rank(r)
        if res:
            best_result = res
            break
            
    if best_result:
        U, V, W = best_result
        return {"U": to_half_int_list(U), "V": to_half_int_list(V), "W": to_half_int_list(W)}
    else:
        # Return schoolbook
        return {"U": to_half_int_list(U_sb), "V": to_half_int_list(V_sb), "W": to_half_int_list(W_sb)}
