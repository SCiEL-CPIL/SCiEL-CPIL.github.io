import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    dimU = n * m  # 6
    dimV = m * p  # 18
    dimW = p * n  # 12
    
    deadline = time.time() + 140
    
    # Try to find a low rank decomposition for 2x3x2 and expand to 2x3x6
    # R(2,3,6) <= 3 * R(2,3,2)
    result = _search_and_expand_232(deadline)
    if result is not None:
        return result
    
    # Fallback to schoolbook 36
    return _schoolbook()

def _search_and_expand_232(deadline):
    n, m, p2 = 2, 3, 2
    dimU = n * m  # 6
    dimV = m * p2 # 6
    dimW = p2 * n # 4
    
    # Build tensor T for 2x3x2
    T = np.zeros((dimU, dimV, dimW), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                a = i * m + j
                b = j * p2 + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try ranks 11 down to 1
    for rank in range(11, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Random restarts
        max_restarts = 100
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart * 123 + rank * 456)
            
            # Initialize with random integers/half-integers in [-1, 1]
            # This biases towards the expected solution structure
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = np.random.choice(vals, size=(dimU, rank))
            V = np.random.choice(vals, size=(dimV, rank))
            W = np.random.choice(vals, size=(dimW, rank))
            
            # Run ALS
            success = _als_232(U, V, W, T, rank, deadline)
            if success:
                # Snap to rationals
                U_s, V_s, W_s = _snap(U, V, W)
                if U_s is not None:
                    # Verify
                    T_check = np.einsum('al,bl,cl->abc', U_s, V_s, W_s)
                    if np.allclose(T_check, T, atol=1e-5):
                        # Expand to 2x3x6
                        return _expand_232_to_236(U_s, V_s, W_s)
                        
    return None

def _als_232(U, V, W, T, rank, deadline):
    dimU, dimV, dimW = T.shape
    
    residual = np.inf
    for it in range(50):
        if time.time() > deadline:
            return False
            
        # Update U
        Z_U = np.einsum('bl,cl->bcl', V, W)
        Z_U_mat = Z_U.transpose(2, 0, 1).reshape(rank, dimV * dimW)
        T_flat = T.reshape(dimU, dimV * dimW)
        U_new, _, _, _ = np.linalg.lstsq(Z_U_mat.T, T_flat.T, rcond=None)
        U = U_new.T
        
        # Update V
        Z_V = np.einsum('al,cl->acl', U, W)
        Z_V_mat = Z_V.transpose(2, 0, 1).reshape(rank, dimU * dimW)
        T_flat_V = T.transpose(1, 0, 2).reshape(dimV, dimU * dimW)
        V_new, _, _, _ = np.linalg.lstsq(Z_V_mat.T, T_flat_V.T, rcond=None)
        V = V_new.T
        
        # Update W
        Z_W = np.einsum('al,bl->abl', U, V)
        Z_W_mat = Z_W.transpose(2, 0, 1).reshape(rank, dimU * dimV)
        T_flat_W = T.transpose(2, 0, 1).reshape(dimW, dimU * dimV)
        W_new, _, _, _ = np.linalg.lstsq(Z_W_mat.T, T_flat_W.T, rcond=None)
        W = W_new.T
        
        # Check residual
        T_decomp = np.einsum('al,bl,cl->abc', U, V, W)
        residual = np.linalg.norm(T - T_decomp)
        if residual < 1e-12:
            return True
            
    return residual < 1e-10

def _snap(U, V, W, tol=1e-5):
    def check_and_snap(mat):
        # Try rounding to nearest 0.5
        snapped = np.round(mat * 2) / 2
        if np.allclose(mat, snapped, atol=tol):
            return snapped
        # Try rounding to nearest integer
        snapped_int = np.round(mat)
        if np.allclose(mat, snapped_int, atol=tol):
            return snapped_int
        return None
    
    U_s = check_and_snap(U)
    V_s = check_and_snap(V)
    W_s = check_and_snap(W)
    
    if U_s is not None and V_s is not None and W_s is not None:
        return U_s, V_s, W_s
    return None, None, None

def _expand_232_to_236(U_232, V_232, W_232):
    n, m, p = 2, 3, 6
    rank_232 = U_232.shape[1]
    rank_total = 3 * rank_232
    
    dimU = n * m  # 6
    dimV = m * p  # 18
    dimW = p * n  # 12
    
    U = np.zeros((dimU, rank_total))
    V = np.zeros((dimV, rank_total))
    W = np.zeros((dimW, rank_total))
    
    # U is shared across blocks (A is the same)
    # But in the tensor sum, each term is independent.
    # T = sum_l (u_l ^ (1) \otimes v_l ^ (1) \otimes w_l ^ (1)) + ...
    # For block 1 (cols 0,1 of B):
    #   U part is U_232
    #   V part is V_232 (padded to 18)
    #   W part is W_232 (padded to 12)
    
    # Block 0: B cols 0,1 -> V indices 0..5, W indices 0..3
    # Block 1: B cols 2,3 -> V indices 6..11, W indices 4..7
    # Block 2: B cols 4,5 -> V indices 12..17, W indices 8..11
    
    for block in range(3):
        offset = block * rank_232
        # U
        U[:, offset:offset+rank_232] = U_232
        
        # V
        V[0:6, offset:offset+rank_232] = V_232
        # V[6:12] and V[12:18] remain 0 for this block
        
        # W
        W[0:4, offset:offset+rank_232] = W_232
        # W[4:8] and W[8:12] remain 0 for this block
        
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

def _schoolbook():
    n, m, p = 2, 3, 6
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
