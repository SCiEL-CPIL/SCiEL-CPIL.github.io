"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import scipy
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    
    # The tensor T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # First, create the schoolbook solution as fallback (Rank 12)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    deadline = time.time() + 140  # Leave 10s margin
    
    # Try to find lower rank decompositions
    # Known lower bound is 6, known rank is 8.
    # We try 8 first as it is the known minimal rank.
    for r in [8, 7, 6]:
        if time.time() > deadline:
            break
        
        result = _find_decomposition(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            break  # We found the minimum rank, stop
    
    return best_result


def _find_decomposition(n, m, p, r, deadline):
    """
    Try to find a rank-r decomposition using multiple strategies.
    """
    # Strategy 1: Try known explicit constructions
    explicit = _try_explicit_rank8(n, m, p)
    if explicit is not None and r <= 8:
        if r == 8:
            return explicit
        # For r < 8, we can't use rank-8 solution directly, but let's try optimization
    
    # Strategy 2: Optimization with better initialization and more attempts
    def loss_and_grad(params):
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r : n*m*r + m*p*r]
        W_flat = params[n*m*r + m*p*r :]
        
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        
        vals = np.zeros((n, m, p))
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    vals[i, j, k] = val
                    err = val - 1.0
                    # Gradients
                    grad_U[i*m+j, :] += 2 * err * (V[j*p+k, :] * W[k*n+i, :])
                    grad_V[j*p+k, :] += 2 * err * (U[i*m+j, :] * W[k*n+i, :])
                    grad_W[k*n+i, :] += 2 * err * (U[i*m+j, :] * V[j*p+k, :])
        
        loss = np.sum(err**2)
        
        # Flatten gradients
        grad_flat = np.concatenate([
            grad_U.flatten(), 
            grad_V.flatten(), 
            grad_W.flatten()
        ])
        
        return loss, grad_flat

    # Try multiple random seeds and initialization strategies
    np.random.seed(42)
    
    max_attempts = 200 if r >= 8 else 500
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Different initialization strategies
        if attempt % 4 == 0:
            # Small random
            U_init = np.random.randn(n * m, r) * 0.1
            V_init = np.random.randn(m * p, r) * 0.1
            W_init = np.random.randn(p * n, r) * 0.1
        elif attempt % 4 == 1:
            # Moderate random
            U_init = np.random.randn(n * m, r) * 0.5
            V_init = np.random.randn(m * p, r) * 0.5
            W_init = np.random.randn(p * n, r) * 0.5
        elif attempt % 4 == 2:
            # Sparse random
            U_init = np.zeros((n * m, r))
            V_init = np.zeros((m * p, r))
            W_init = np.zeros((p * n, r))
            for _ in range(r * 2):
                U_init[np.random.randint(n*m), np.random.randint(r)] = np.random.choice([-1, 1])
                V_init[np.random.randint(m*p), np.random.randint(r)] = np.random.choice([-1, 1])
                W_init[np.random.randint(p*n), np.random.randint(r)] = np.random.choice([-1, 1])
        else:
            # Structured initialization based on schoolbook
            U_init = np.zeros((n * m, r))
            V_init = np.zeros((m * p, r))
            W_init = np.zeros((p * n, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i*m+j < n*m and j*p+k < m*p and k*n+i < p*n:
                            c = (i * m + j) * p + k
                            if c < r:
                                U_init[i*m+j, c] = 1.0
                                V_init[j*p+k, c] = 1.0
                                W_init[k*n+i, c] = 1.0
        
        x0 = np.concatenate([
            U_init.flatten(), 
            V_init.flatten(), 
            W_init.flatten()
        ])
        
        try:
            # Use L-BFGS-B for robust optimization
            result = scipy.optimize.minimize(
                loss_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < 1e-8:
                # Extract optimized parameters
                U_opt = result.x[:n*m*r].reshape(n*m, r)
                V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Try to round to various rational values
                # Check denominators 1, 2, 3, 4, 5, 6
                for denom in [1, 2, 3, 4, 5, 6]:
                    U_rounded = np.round(U_opt * denom) / denom
                    V_rounded = np.round(V_opt * denom) / denom
                    W_rounded = np.round(W_opt * denom) / denom
                    
                    # Verify exactness
                    loss_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                                loss_rounded += (val - 1.0)**2
                                
                    if loss_rounded < 1e-10:
                        U_list = [[float(U_rounded[i, c]) for c in range(r)] for i in range(n*m)]
                        V_list = [[float(V_rounded[i, c]) for c in range(r)] for i in range(m*p)]
                        W_list = [[float(W_rounded[i, c]) for c in range(r)] for i in range(p*n)]
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
        except Exception:
            continue
            
    return None


def _try_explicit_rank8(n, m, p):
    """
    Try to find an explicit rank-8 decomposition for 2x2 x 2x3.
    """
    if n != 2 or m != 2 or p != 3:
        return None
    
    # Try a known construction
    # We'll attempt to find a solution by solving a linear system
    # with some variables fixed
    
    # One approach: use the fact that we can write
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # and find 8 rank-1 terms
    
    # Let's try a specific construction
    # Terms 1-3: A[0,0]*B[0,k] for k=0,1,2
    # Terms 4-6: A[1,0]*B[0,k] for k=0,1,2  
    # Terms 7-8: shared terms for A[i,1]*B[1,k]
    
    # Actually, let's try to solve this more systematically
    # We need 8 columns in U, V, W
    
    # Try a parameterized solution
    # Let's use the identity:
    # a*b + c*d = (a+c)*(b+d) - a*d - c*b
    # This can help merge terms
    
    # For each i, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Let's try:
    # Term 1: A[0,0]*B[0,0]
    # Term 2: A[0,0]*B[0,1]  
    # Term 3: A[0,0]*B[0,2]
    # Term 4: A[1,0]*B[0,0]
    # Term 5: A[1,0]*B[0,1]
    # Term 6: A[1,0]*B[0,2]
    # Terms 7-8: handle A[i,1]*B[1,k] with sharing
    
    # This is getting complex. Let's try a numerical approach with fixed structure.
    
    # Try to find a solution where W has a specific form
    # W[k*n+i, c] for k=0,1,2 and i=0,1
    
    # Let's try: W[0,0]=1, W[0,1]=0, W[1,0]=0, W[1,1]=1, W[2,0]=1, W[2,1]=0, etc.
    
    # Actually, let's just return None and let the optimization handle it
    return None
# EVOLVE-BLOCK-END
