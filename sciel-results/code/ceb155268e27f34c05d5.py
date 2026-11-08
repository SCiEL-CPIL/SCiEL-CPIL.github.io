import numpy as np
import time

DIMENSIONS = [3, 4, 5]
TIME_LIMIT = 140  # safety margin from 150s

def _verify(U, V, W):
    """Check if U,V,W exactly decompose the 3x4x5 matmul tensor."""
    n, m, p = 3, 4, 5
    rank = U.shape[1]
    
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # T = einsum('ur,vr,wr->uvw', U, V, W)
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Expected: T[i*m+l, l*p+k, k*n+i] = 1, else 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_idx = i * m + l
                v_idx = l * p + k
                w_idx = k * n + i
                if abs(T[u_idx, v_idx, w_idx] - 1.0) > 1e-9:
                    return False
    
    # Check that T is zero elsewhere
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_idx = i * m + l
                v_idx = l * p + k
                w_idx = k * n + i
                mask[u_idx, v_idx, w_idx] = True
    
    invalid = ~mask
    if np.any(np.abs(T[invalid]) > 1e-9):
        return False
    
    return True

def _seed_solution():
    """Return the seed 56-rank solution as U,V,W numpy arrays."""
    n, m, p = 3, 4, 5
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_blk[i, r]
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_blk[i, r]
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_blk[i, r]
    
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], *strassen_2x2x2())
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], *strassen_2x2x2())
    add_block([0,1,4,5], [4,9], [12,13], *schoolbook_2x2x1())
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], *strassen_2x2x2())
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], *strassen_2x2x2())
    add_block([2,3,6,7], [14,19], [12,13], *schoolbook_2x2x1())
    add_block([8,9], [0,1,5,6], [2,5], *schoolbook_1x2x2())
    add_block([8,9], [2,3,7,8], [8,11], *schoolbook_1x2x2())
    add_block([8,9], [4,9], [14], *schoolbook_1x2x1())
    add_block([10,11], [10,11,15,16], [2,5], *schoolbook_1x2x2())
    add_block([10,11], [12,13,17,18], [8,11], *schoolbook_1x2x2())
    add_block([10,11], [14,19], [14], *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W

def _try_remove_columns(U, V, W, max_tries=200):
    """Try to remove columns by checking linear dependence."""
    n, m, p = 3, 4, 5
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Try removing each column
    for r in range(rank):
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        if _verify(U_test, V_test, W_test):
            return U_test, V_test, W_test
    return None

def _mix_and_reduce(U, V, W, n_trials=50):
    """Apply random invertible transformations and try to reduce rank."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    for t in range(n_trials):
        # Generate random invertible matrix M
        # Use sparse perturbation to keep numbers small
        M = np.eye(best_rank)
        # Add random sparse entries
        for _ in range(best_rank // 2):
            i, j = np.random.randint(0, best_rank, 2)
            if i != j:
                M[i, j] = np.random.choice([-1, 1])
        
        # Check if M is invertible
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
            
        # Transform
        U_new = best_U @ M
        V_new = best_V @ M_inv
        W_new = best_W
        
        # Try to remove columns
        result = _try_remove_columns(U_new, V_new, W_new)
        if result is not None:
            U_new, V_new, W_new = result
            if U_new.shape[1] < best_rank:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = best_U.shape[1]
                
    return best_U, best_V, best_W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from seed
    U, V, W = _seed_solution()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = U.shape[1]
    
    # Verify seed
    if not _verify(best[0], best[1], best[2]):
        raise ValueError("Seed solution failed verification")
    
    # Iteratively try to reduce rank using mix-and-reduce
    for iteration in range(100):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
        
        U_c, V_c, W_c = best[0].copy(), best[1].copy(), best[2].copy()
        
        # Apply mix and reduce
        U_new, V_new, W_new = _mix_and_reduce(U_c, V_c, W_c, n_trials=20)
        
        if U_new.shape[1] < best_rank:
            best = (U_new, V_new, W_new)
            best_rank = best[0].shape[1]
            
        # If we hit a very low rank, we can stop or continue
        if best_rank <= 50:
            break

    # Final verification
    if not _verify(best[0], best[1], best[2]):
        # Fallback to seed if something went wrong
        best = (U.copy(), V.copy(), W.copy())

    U_out, V_out, W_out = best
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
