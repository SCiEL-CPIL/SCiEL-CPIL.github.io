"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses Strassen per pair of columns as baseline (rank 42), then attempts
    to reduce rank by finding shared structure via optimization.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    nm = n * m  # 4
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Strassen coefficients for 2x2x2 multiplication
    # U: 4x7, V: 4x7, W: 4x7
    # Using standard Strassen decomposition
    strassen_U = np.array([
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0]
    ], dtype=float)
    
    strassen_V = np.array([
        [1, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 1]
    ], dtype=float)
    
    # W[k*n+i, s] where k is column index (0 or 1), i is row index (0 or 1)
    # Standard Strassen: C[0,0] = M1 + M4 - M5 - M7, C[0,1] = M3 + M5, C[1,0] = M2 + M4, C[1,1] = M1 - M3 + M6 + M7
    # So W[0,0]=1 (k=0,i=0), W[1,0]=0 (k=0,i=1), W[2,0]=1 (k=1,i=0), W[3,0]=1 (k=1,i=1)
    # Wait, let me be more careful with the trace convention W[k*n+i, r]
    # k is the column of B (which becomes the column of C), i is the row of A (which becomes the row of C)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For 2x2x2: T[i,j,k] = 1 for all i,j,k in {0,1}
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # U[i*m+j, r] = coefficient of A[i,j] in Mr
    # U[0,0]=1 (A00 in M1), U[3,0]=1 (A11 in M1)
    # U[2,1]=1 (A10 in M2), U[3,1]=1 (A11 in M2)
    # U[0,2]=1 (A00 in M3)
    # U[3,3]=1 (A11 in M4)
    # U[0,4]=1 (A00 in M5), U[1,4]=1 (A01 in M5)
    # U[2,5]=1 (A10 in M6), U[0,5]=-1 (A00 in M6)
    # U[1,6]=1 (A01 in M7), U[3,6]=-1 (A11 in M7)
    
    strassen_U = np.zeros((4, 7))
    strassen_U[0, 0] = 1; strassen_U[3, 0] = 1
    strassen_U[2, 1] = 1; strassen_U[3, 1] = 1
    strassen_U[0, 2] = 1
    strassen_U[3, 3] = 1
    strassen_U[0, 4] = 1; strassen_U[1, 4] = 1
    strassen_U[2, 5] = 1; strassen_U[0, 5] = -1
    strassen_U[1, 6] = 1; strassen_U[3, 6] = -1
    
    # V[j*p+k, r] = coefficient of B[j,k] in Mr
    # M1: B00+B11 -> V[0,0]=1, V[3,0]=1 (j=0,k=0 and j=1,k=1)
    # M2: B00 -> V[0,1]=1
    # M3: B10-B11 -> V[2,2]=1, V[3,2]=-1
    # M4: B01-B10 -> V[1,3]=1, V[2,3]=-1
    # M5: B11 -> V[3,4]=1
    # M6: B00+B10 -> V[0,5]=1, V[2,5]=1
    # M7: B01+B11 -> V[1,6]=1, V[3,6]=1
    
    strassen_V = np.zeros((4, 7))
    strassen_V[0, 0] = 1; strassen_V[3, 0] = 1
    strassen_V[0, 1] = 1
    strassen_V[2, 2] = 1; strassen_V[3, 2] = -1
    strassen_V[1, 3] = 1; strassen_V[2, 3] = -1
    strassen_V[3, 4] = 1
    strassen_V[0, 5] = 1; strassen_V[2, 5] = 1
    strassen_V[1, 6] = 1; strassen_V[3, 6] = 1
    
    # W[k*n+i, r] = coefficient of Mr in C[i,k]
    # C00 = M1 + M4 - M5 - M7 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=-1
    # C01 = M3 + M5 -> W[1,2]=1, W[1,4]=1
    # C10 = M2 + M4 -> W[2,1]=1, W[2,3]=1
    # C11 = M1 - M3 + M6 + M7 -> W[3,0]=1, W[3,2]=-1, W[3,5]=1, W[3,6]=1
    
    strassen_W = np.zeros((4, 7))
    strassen_W[0, 0] = 1; strassen_W[0, 3] = 1; strassen_W[0, 4] = -1; strassen_W[0, 6] = -1
    strassen_W[1, 2] = 1; strassen_W[1, 4] = 1
    strassen_W[2, 1] = 1; strassen_W[2, 3] = 1
    strassen_W[3, 0] = 1; strassen_W[3, 2] = -1; strassen_W[3, 5] = 1; strassen_W[3, 6] = 1
    
    def build_strassen_decomposition(rank):
        """Build the standard Strassen-per-pair decomposition with given rank."""
        if rank < (p // 2) * 7:
            return None
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        num_pairs = p // 2
        for t in range(num_pairs):
            base_c = t * 7
            k0 = 2 * t
            for s in range(7):
                c = base_c + s
                U[:, c] += strassen_U[:, s]
                # V: for pair t, B columns are k0 and k0+1
                # V[j*p+k, c] where k = k0 or k0+1
                for j in range(m):
                    for k_loc in range(2):
                        global_k = k0 + k_loc
                        v_idx = j * p + global_k
                        V[v_idx, c] += strassen_V[j * 2 + k_loc, s]
                # W: for pair t, C columns are k0 and k0+1
                for k_loc in range(2):
                    global_k = k0 + k_loc
                    for i in range(n):
                        w_idx = global_k * n + i
                        W[w_idx, c] += strassen_W[k_loc * n + i, s]
        
        return U, V, W
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def tensor_error(U, V, W):
        """Compute the squared error of the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0) ** 2
        return err
    
    # Start with rank 42 (Strassen per pair)
    best_rank = 42
    best_U, best_V, best_W = build_strassen_decomposition(42)
    
    # Try to reduce rank
    for candidate_rank in range(36, 42):
        if time.time() > deadline - 30:
            break
        
        # Initialize with Strassen for first 5 pairs (35 terms) + 1 free term
        U_init = np.zeros((nm, candidate_rank))
        V_init = np.zeros((mp, candidate_rank))
        W_init = np.zeros((pn, candidate_rank))
        
        num_pairs_full = min(5, p // 2)
        for t in range(num_pairs_full):
            base_c = t * 7
            k0 = 2 * t
            for s in range(7):
                c = base_c + s
                if c >= candidate_rank:
                    break
                U[:, c] = 0  # reset
                U[:, c] += strassen_U[:, s]
                for j in range(m):
                    for k_loc in range(2):
                        global_k = k0 + k_loc
                        v_idx = j * p + global_k
                        V_init[v_idx, c] += strassen_V[j * 2 + k_loc, s]
                for k_loc in range(2):
                    global_k = k0 + k_loc
                    for i in range(n):
                        w_idx = global_k * n + i
                        W_init[w_idx, c] += strassen_W[k_loc * n + i, s]
        
        # Last term is free
        np.random.seed(42)
        U_init[:, -1] = np.random.randn(nm) * 0.1
        V_init[:, -1] = np.random.randn(mp) * 0.1
        W_init[:, -1] = np.random.randn(pn) * 0.1
        
        def flatten_params(x):
            U_f = x[:nm * candidate_rank].reshape(nm, candidate_rank)
            V_f = x[nm * candidate_rank: nm * candidate_rank + mp * candidate_rank].reshape(mp, candidate_rank)
            W_f = x[nm * candidate_rank + mp * candidate_rank:].reshape(pn, candidate_rank)
            return U_f, V_f, W_f
        
        def objective(x):
            U_f, V_f, W_f = flatten_params(x)
            return tensor_error(U_f, V_f, W_f)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        found = False
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            if restart == 0:
                x_start = x0.copy()
            else:
                np.random.seed(42 + restart)
                x_start = x0 + np.random.randn(len(x0)) * 0.01
            
            try:
                result = minimize(objective, x_start, method='L-BFGS-B',
                                 options={'maxiter': 500, 'gtol': 1e-10, 'maxls': 100})
                
                if result.fun < 1e-12:
                    U_opt = result.x[:nm * candidate_rank].reshape(nm, candidate_rank)
                    V_opt = result.x[nm * candidate_rank: nm * candidate_rank + mp * candidate_rank].reshape(mp, candidate_rank)
                    W_opt = result.x[nm * candidate_rank + mp * candidate_rank:].reshape(pn, candidate_rank)
                    
                    if check_decomposition(U_opt, V_opt, W_opt, tol=1e-10):
                        best_rank = candidate_rank
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
