import numpy as np
from scipy.optimize import differential_evolution, minimize

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Create the target tensor T (dense)
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1.0

    def full_error(params, rank):
        """
        Objective function: Sum of squared errors over ALL tensor entries.
        """
        u_size = dim_u * rank
        v_size = dim_v * rank
        
        u_flat = params[:u_size]
        v_flat = params[u_size:u_size+v_size]
        w_flat = params[u_size+v_size:]
        
        U = u_flat.reshape(dim_u, rank)
        V = v_flat.reshape(dim_v, rank)
        W = w_flat.reshape(dim_w, rank)
        
        # Compute T_hat
        T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Error
        err = T_hat - T_target
        return np.sum(err ** 2)

    def check_validity(U, V, W, atol=1e-6):
        T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_hat, T_target, atol=atol)

    # Try ranks from 35 to 42
    # We prioritize lower ranks.
    for rank in range(35, 43):
        # Define bounds for DE
        # Coefficients are likely small integers or half-integers.
        # Bounds [-10, 10] should cover it.
        bounds = [(-10, 10)] * ((dim_u + dim_v + dim_w) * rank)
        
        # Differential Evolution is stochastic. We run it a few times or with specific settings.
        # To save time, we limit maxiter and popsize.
        try:
            res = differential_evolution(
                full_error, 
                bounds, 
                args=(rank,),
                maxiter=100, 
                popsize=10,
                tol=1e-6,
                seed=42,
                polish=False # We will polish manually
            )
            
            # If DE found a good solution, polish it with L-BFGS-B
            if res.fun < 1e-3:
                # Define gradient for L-BFGS-B (optional, but DE provides a good start)
                # We can just use L-BFGS-B with finite differences if gradient is not provided, 
                # but that's slow. Let's use the analytical gradient from the original code 
                # but adapted for full error? 
                # Actually, L-BFGS-B with finite diff on 1000 vars is too slow.
                # Let's just check if res.fun is close enough to 0.
                # Or use a few steps of L-BFGS-B.
                
                U_final = res.x[:dim_u*rank].reshape(dim_u, rank)
                V_final = res.x[dim_u*rank:(dim_u+dim_v)*rank].reshape(dim_v, rank)
                W_final = res.x[(dim_u+dim_v)*rank:].reshape(dim_w, rank)
                
                if check_validity(U_final, V_final, W_final):
                    return {
                        "U": U_final.tolist(),
                        "V": V_final.tolist(),
                        "W": W_final.tolist()
                    }
        except Exception:
            pass

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
