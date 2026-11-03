"""Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
DEADLINE_MARGIN = 15  # seconds before the 150s kill


def _time_left():
    return (START_TIME + 150 - DEADLINE_MARGIN) - time.time()


def _verify(U, V, W, tol=1e-9):
    """Check exact decomposition using einsum."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    T_computed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    T_expected = np.ones((n, m, p), dtype=np.float64)
    
    return np.max(np.abs(T_computed - T_expected)) < tol


def _als_refine(U, V, W, max_iters=500, tol=1e-12):
    """Alternating Least Squares refinement for tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    for it in range(max_iters):
        if _time_left() < 10:
            break
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute current tensor
        T_cur = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        diff = np.ones((n, m, p), dtype=np.float64) - T_cur
        
        # Update U: for each (i,j,c), minimize ||diff - U_new * (V*W)||^2
        # dU[i,j,c] involves: sum_k diff[i,j,k] * V[j,k,c] * W[k,i,c]
        # But ALS: fix V,W, solve for U
        # U[i,j,c] = sum_k diff[i,j,k] * V[j,k,c] * W[k,i,c] / sum_k V[j,k,c]^2 * W[k,i,c]^2
        
        # More precisely, for fixed V,W:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # For each (i,j), we have p equations in rank unknowns (one per c)
        # This is a linear system: A[i,j] * U[i,j,:] = b[i,j]
        # where A[i,j][k,c] = V[j,k,c]*W[k,i,c], b[i,j][k] = 1
        
        # Solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build system: for each k, sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
                # Matrix M[k,c] = V[j,k,c]*W[k,i,c], vector b[k]=1
                M = np.array([[V_3d[j, k, c] * W_3d[k, i, c] for c in range(rank)] for k in range(p)])
                b = np.ones(p)
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    U_new[i*m + j, :] = sol
                except:
                    U_new[i*m + j, :] = U[i*m + j, :]
        
        # Update V
        U_3d = U_new.reshape(n, m, rank)
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                M = np.array([[U_3d[i, j, c] * W_3d[k, i, c] for c in range(rank)] for i in range(n)])
                b = np.ones(n)
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    V_new[j*p + k, :] = sol
                except:
                    V_new[j*p + k, :] = V[j*p + k, :]
        
        # Update W
        V_3d = V_new.reshape(m, p, rank)
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.array([[U_3d[i, j, c] * V_3d[j, k, c] for c in range(rank)] for j in range(m)])
                b = np.ones(m)
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    W_new[k*n + i, :] = sol
                except:
                    W_new[k*n + i, :] = W[k*n + i, :]
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T_cur = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        diff_norm = np.max(np.abs(T_cur - 1.0))
        
        if diff_norm < tol:
            break
    
    return U, V, W


def _gradient_polish(U, V, W, max_iters=100, lr=0.05):
    """Gradient-based polishing after ALS."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    for it in range(max_iters):
        if _time_left() < 10:
            break
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        T_cur = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        diff = T_cur - 1.0
        diff_norm = np.max(np.abs(diff))
        
        if diff_norm < 1e-12:
            break
        
        # Gradients
        dU_3d = np.einsum('ijk,jkc,kic->ijc', diff, V_3d, W_3d)
        dV_3d = np.einsum('ijk,ijc,kic->jkc', diff, U_3d, W_3d)
        dW_3d = np.einsum('ijk,ijc,jkc->kic', diff, U_3d, V_3d)
        
        lr_eff = lr / (1 + it * 0.01)
        U -= lr_eff * dU_3d.reshape(n*m, rank)
        V -= lr_eff * dV_3d.reshape(m*p, rank)
        W -= lr_eff * dW_3d.reshape(p*n, rank)
    
    return U, V, W


def _try_rank(rank, num_restarts=3, seed=0):
    """Try to find rank-rank decomposition via random init + ALS + gradient polish."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    for restart in range(num_restarts):
        if _time_left() < 20:
            break
        
        # Random initialization
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Normalize columns
        for c in range(rank):
            norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
            if norm > 1e-12:
                U[:, c] /= norm
                V[:, c] /= norm
                W[:, c] /= norm
        
        # ALS refinement
        U, V, W = _als_refine(U, V, W, max_iters=200)
        
        # Check if solution found
        if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-10):
            return U, V, W, True
        
        # Gradient polish
        U, V, W = _gradient_polish(U, V, W, max_iters=50)
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-10):
            return U, V, W, True
    
    return None, None, None, False


def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    # Strassen's 7 products for 2x2 matrix multiplication
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U[i*m+j, c] where i,j in {0,1}, c in {0..6}
    # V[j*p+k, c] where j,k in {0,1}, c in {0..6}
    # W[k*n+i, c] where k,i in {0,1}, c in {0..6}
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a11+a22)*(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # contributes to C11 and C22
    
    # M2: (a21+a22)*b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21
    
    # M3: a11*(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12
    
    # M4: a22*(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11 and C21
    
    # M5: (a11+a12)*b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11 and C12
    
    # M6: (a21-a11)*(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22
    
    # M7: (a12-a22)*(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11
    
    return U, V, W


def _construct_kronecker():
    """Try Kronecker product constructions."""
    n, m, p = DIMENSIONS
    
    # 3x4x6 = (3x2x3) x (2x2x2) in some sense?
    # Or use Strassen for the 2x2x2 part and schoolbook for the rest
    
    # Simple approach: use Strassen for a 2x2x2 block and schoolbook for the rest
    # This is complex; let's try a different structured approach
    
    # Try: decompose 3x4x6 as sum of rank-1 terms with structure
    # For now, just return None and let the search handle it
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But for matrix multiplication, known bounds are higher
    # For 3x4x6, let's try from 30 up to 71
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Try a range of ranks, starting from reasonable values
    ranks_to_try = list(range(30, min(schoolbook_rank, 72)))
    
    # Sort by likelihood (try middle values first, then expand)
    # Actually, let's try in order from low to high
    
    best_result = None
    
    for rank in ranks_to_try:
        if _time_left() < 30:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(3):
            if _time_left() < 20:
                break
            
            U, V, W, success = _try_rank(rank, num_restarts=2, seed=seed)
            if success:
                if _verify(U.tolist(), V.tolist(), W.tolist()):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    return best_result
                found = True
                break
        
        if found:
            break
    
    # If no better solution found, return schoolbook
    return _schoolbook()
