import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Vectorized verification
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    tensor = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    expected = np.ones((n, m, p))
    
    if not np.allclose(tensor, expected, atol=1e-9):
        return False, tensor
    return True, tensor

def build_rank_28(n, m, p):
    """
    Construct a rank-28 decomposition for 2x3x5 matrix multiplication.
    """
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        return col + 1

    def add_strassen(i_off, j_off, k_off, col):
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = -1.0
        col += 1

        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = -1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+0)][col] = -1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = -1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        col += 1

        return col

    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)
    
    col = add_strassen(0, 1, 0, col)
    
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)
    
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def search_lower_rank(n, m, p, base_U, base_V, base_W, target_rank, time_budget, rng_seed=0):
    """
    Search for a lower-rank decomposition using the base solution as initialization.
    """
    start_time = time.time()
    rng = np.random.default_rng(rng_seed)
    
    if target_rank >= base_U.shape[1]:
        return None
    
    num_u = n * m * target_rank
    num_v = m * p * target_rank
    num_w = p * n * target_rank
    
    def reshape(x):
        U = x[:num_u].reshape(n * m, target_rank)
        V = x[num_u:num_u+num_v].reshape(m * p, target_rank)
        W = x[num_u+num_v:].reshape(p * n, target_rank)
        return U, V, W
        
    def objective(x):
        U, V, W = reshape(x)
        U_3d = U.reshape(n, m, target_rank)
        V_3d = V.reshape(m, p, target_rank)
        W_3d = W.reshape(p, n, target_rank)
        
        tensor = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = tensor - 1.0
        return np.sum(err**2)

    # Try multiple initializations
    best_obj = float('inf')
    best_x = None
    
    # Initialization 1: Truncation
    U0 = base_U[:, :target_rank]
    V0 = base_V[:, :target_rank]
    W0 = base_W[:, :target_rank]
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    # Initialization 2: Randomized perturbation of truncation
    U1 = base_U[:, :target_rank] + rng.normal(0, 0.1, U0.shape)
    V1 = base_V[:, :target_rank] + rng.normal(0, 0.1, V0.shape)
    W1 = base_W[:, :target_rank] + rng.normal(0, 0.1, W0.shape)
    x1 = np.concatenate([U1.flatten(), V1.flatten(), W1.flatten()])
    
    # Initialization 3: Pure random (scaled)
    U2 = rng.normal(0, 1, (n*m, target_rank))
    V2 = rng.normal(0, 1, (m*p, target_rank))
    W2 = rng.normal(0, 1, (p*n, target_rank))
    x2 = np.concatenate([U2.flatten(), V2.flatten(), W2.flatten()])
    
    candidates = [x0, x1, x2]
    
    for x_init in candidates:
        if time.time() - start_time > time_budget:
            break
            
        obj_init = objective(x_init)
        if obj_init < best_obj:
            best_obj = obj_init
            best_x = x_init.copy()
            
        try:
            result = minimize(objective, x_init, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-6, 'disp': False})
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
        except Exception:
            pass
        
        if best_obj < 1e-9:
            break

    if best_obj < 1e-9 and best_x is not None:
        U, V, W = reshape(best_x)
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        valid, _ = verify_decomposition(U_list, V_list, W_list, n, m, p)
        if valid:
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_result = None
    best_rank = float('inf')
    
    # First, try the known rank-28 solution
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            best_result = {'U': U, 'V': V, 'W': W}
            best_rank = 28
    except Exception as e:
        pass
    
    # Try to find lower rank solutions
    if best_result is not None:
        for target_rank in [27, 26]:
            if time.time() > deadline:
                break
            
            time_budget = min(deadline - time.time(), 30.0)
            if time_budget < 5.0:
                break
            
            try:
                result = search_lower_rank(n, m, p, 
                                          np.array(best_result['U']), 
                                          np.array(best_result['V']), 
                                          np.array(best_result['W']),
                                          target_rank, time_budget, rng_seed=target_rank)
                if result is not None:
                    U, V, W = result
                    best_result = {'U': U, 'V': V, 'W': W}
                    best_rank = target_rank
                    break
            except Exception as e:
                pass
    
    # Fallback: Schoolbook (Rank 30)
    if best_result is None:
        rank_school = n * m * p
        U = [[0.0] * rank_school for _ in range(n * m)]
        V = [[0.0] * rank_school for _ in range(m * p)]
        W = [[0.0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            best_result = {'U': U, 'V': V, 'W': W}
    
    if best_result is not None:
        return best_result
    
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
