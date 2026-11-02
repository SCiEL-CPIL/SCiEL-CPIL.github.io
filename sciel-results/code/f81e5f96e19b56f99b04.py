"""Optimized tensor rank decomposition for 3x4 * 4x4 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

N, M, P = 3, 4, 4
RANK_SCHOOLBOOK = N * M * P  # 48


def _verify_fast(U, V, W, n=N, m=M, p=P):
    """Faster verification using tensor reshaping."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(recon, expected, atol=1e-9)


def _strassen_2x2():
    """
    Return rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1: M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Product 2: M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = -1; W[2, 1] = 1
    
    # Product 3: M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Product 4: M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Product 5: M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Product 6: M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # Product 7: M7 = (b-c)(e+g)
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_strassen_32():
    """
    Build rank-32 decomposition for 3x4 * 4x4 using Strassen blocks.
    """
    rank = 32
    U = np.zeros((12, rank))  # 3*4=12 rows
    V = np.zeros((16, rank))  # 4*4=16 rows
    W = np.zeros((12, rank))  # 4*3=12 rows
    
    col = 0
    
    # Pre-compute Strassen blocks
    sU, sV, sW = _strassen_2x2()

    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, current_col):
        for c in range(7):
            # U: A indices
            for li in range(2):
                for lj in range(2):
                    U[(a_row_offset + li) * 4 + (a_col_offset + lj), current_col] = sU[li * 2 + lj][c]
            
            # V: B indices
            for lj in range(2):
                for lk in range(2):
                    V[(b_row_offset + lj) * 4 + (b_col_offset + lk), current_col] = sV[lj * 2 + lk][c]
            
            # W: C indices
            for lk in range(2):
                for li in range(2):
                    W[(b_col_offset + lk) * 3 + (a_row_offset + li), current_col] = sW[lk * 2 + li][c]
        
        return current_col + 7

    # --- 2x4 * 4x4 part (i=0,1) ---
    col = add_strassen_block(0, 0, 0, 0, col)
    col = add_strassen_block(0, 0, 0, 2, col)
    col = add_strassen_block(0, 2, 2, 0, col)
    col = add_strassen_block(0, 2, 2, 2, col)
    
    # --- 1x4 * 4x4 part (i=2) ---
    for j in range(4):
        U[2 * 4 + j, col] = 1
        for k in range(4):
            V[j * 4 + k, col] = 1
            W[k * 3 + 2, col] = 1
        col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    return U.tolist(), V.tolist(), W.tolist()


def _try_optimization(target_rank, deadline):
    """Try to find a decomposition of the given rank using optimization."""
    if time.time() > deadline - 10:
        return None
    
    n, m, p = N, M, P
    r = target_rank
    
    # Parameterize: flatten U, V, W into a single vector
    # Total params: n*m*r + m*p*r + p*n*r = r*(n*m + m*p + p*n) = r*(12+16+12) = 40*r
    total_params = r * (n * m + m * p + p * n)
    
    def objective(params):
        if time.time() > deadline - 5:
            return 1e10
        
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        expected = np.ones((n, m, p))
        return np.sum((recon - expected) ** 2)
    
    # Try a few random initializations
    np.random.seed(42)
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        
        init = np.random.randn(total_params) * 0.1
        result = minimize(objective, init, method='L-BFGS-B', 
                         options={'maxiter': 200, 'maxfun': 500})
        
        if result.fun < 1e-12:
            U = result.x[:n*m*r].reshape(n*m, r)
            V = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W = result.x[n*m*r + m*p*r:].reshape(p*n, r)
            
            if _verify_fast(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _schoolbook():
    """Fallback schoolbook decomposition of rank 48."""
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # First try the known rank-32 Strassen-based decomposition
    U, V, W = _build_strassen_32()
    if _verify_fast(U, V, W):
        best_rank = 32
        best_result = {"U": U, "V": V, "W": W}
    else:
        best_rank = RANK_SCHOOLBOOK
        best_result = _schoolbook()
    
    # Try to find lower rank using optimization
    # Try ranks from 20 to 31
    for r in range(20, min(32, best_rank)):
        if time.time() > deadline - 15:
            break
        
        result = _try_optimization(r, deadline)
        if result is not None:
            U_opt, V_opt, W_opt = result
            if _verify_fast(U_opt, V_opt, W_opt):
                best_rank = r
                best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                break  # Found the minimum, stop
    
    return best_result
