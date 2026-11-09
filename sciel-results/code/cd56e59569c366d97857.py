import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _construct_rank50():
    n, m, p = 2, 3, 9
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1; W[w_00][col] += 1; W[w_11][col] += 1
        U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1; W[w_10][col+1] += 1; W[w_11][col+1] -= 1
        U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1; W[w_01][col+2] += 1; W[w_11][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1; W[w_00][col+3] += 1; W[w_10][col+3] += 1
        U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1; W[w_00][col+4] -= 1; W[w_01][col+4] += 1
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1; W[w_11][col+5] += 1
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1; W[w_00][col+6] += 1
        col += 7
    
    k8 = 8
    v_e8 = 0 * p + k8; v_g8 = 1 * p + k8; w_08 = k8 * n + 0; w_18 = k8 * n + 1
    U[u_a][col] += 1; V[v_e8][col] += 1; W[w_08][col] += 1
    U[u_b][col+1] += 1; V[v_g8][col+1] += 1; W[w_08][col+1] += 1
    U[u_c][col+2] += 1; V[v_e8][col+2] += 1; W[w_18][col+2] += 1
    U[u_d][col+3] += 1; V[v_g8][col+3] += 1; W[w_18][col+3] += 1
    col += 4
    
    u_ar0 = 2; u_ar1 = 5
    for k in range(p):
        v_bk = 2 * p + k; w_0k = k * n + 0; w_1k = k * n + 1
        U[u_ar0][col] += 1; V[v_bk][col] += 1; W[w_0k][col] += 1
        U[u_ar1][col+1] += 1; V[v_bk][col+1] += 1; W[w_1k][col+1] += 1
        col += 2

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=2, m=3, p=9, rank=50):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_mat = U_np[:, r].reshape(n, m)
        v_mat = V_np[:, r].reshape(m, p)
        w_mat = W_np[:, r].reshape(p, n).T
        C_computed += (u_mat @ v_mat) * w_mat
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _search_rank49():
    if _time_left() < 30.0:
        return None
    
    n, m, p = 2, 3, 9
    rank = 49
    dim_U = n * m  # 6
    dim_V = m * p  # 27
    dim_W = p * n  # 18
    
    # Target tensor: T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    # We need: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    
    # Flatten the target
    target = np.ones((n, m, p))
    
    def error(params):
        if _time_left() < 5.0:
            return 1e10
        
        U_flat = params[:rank*dim_U].reshape(rank, dim_U).T  # (dim_U, rank)
        V_flat = params[rank*dim_U:rank*dim_U+rank*dim_V].reshape(rank, dim_V).T  # (dim_V, rank)
        W_flat = params[rank*dim_U+rank*dim_V:].reshape(rank, dim_W).T  # (dim_W, rank)
        
        # Compute the tensor
        tensor = np.zeros((n, m, p))
        for r in range(rank):
            u_mat = U_flat[:, r].reshape(n, m)
            v_mat = V_flat[:, r].reshape(m, p)
            w_mat = W_flat[:, r].reshape(p, n)
            # The contribution to T[i,k,j] is u_mat[i,k] * v_mat[k,j] * w_mat[j,i]
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        tensor[i,k,j] += u_mat[i,k] * v_mat[k,j] * w_mat[j,i]
        
        return np.sum((tensor - target) ** 2)
    
    # Initialize
    np.random.seed(123)
    x0 = np.random.randn(rank * (dim_U + dim_V + dim_W)) * 0.1
    
    # Use Nelder-Mead or BFGS
    try:
        result = minimize(error, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result.fun < 1e-6:
            # Extract U, V, W
            U_flat = result.x[:rank*dim_U].reshape(rank, dim_U).T
            V_flat = result.x[rank*dim_U:rank*dim_U+rank*dim_V].reshape(rank, dim_V).T
            W_flat = result.x[rank*dim_U+rank*dim_V:].reshape(rank, dim_W).T
            
            U = U_flat.tolist()
            V = V_flat.tolist()
            W = W_flat.tolist()
            
            # Verify
            if _verify(U, V, W, n, m, p, rank):
                return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
    
    return None

def construct():
    if _time_left() < 5.0:
        return _construct_rank50()
    
    # Try to find rank 49
    result49 = _search_rank49()
    if result49 is not None:
        return result49
    
    # Fallback to rank 50
    return _construct_rank50()
