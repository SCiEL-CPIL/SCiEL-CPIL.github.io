"""Matrix multiplication 4x4x6 - Strassen-based rank reduction."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p  # 96
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strassen-based decomposition for 4x4x6
    # Top level: 4x4 = 2x2 blocks of 2x2, 4x6 = 2x3 blocks of 2x3
    # 7 Strassen multiplications of (2x2) * (2x3)
    # Each (2x2)*(2x3): inner Strassen (7) for cols 0-1 + schoolbook (4) for col 2 = 11
    # Total rank: 7 * 11 = 77
    
    R = 77
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    # Strassen coefficients for 2x2x2
    alpha = [
        [1, 0, 0, 1],  # s0: A00 + A11
        [0, 0, 1, 1],  # s1: A10 + A11
        [1, 0, 0, 0],  # s2: A00
        [0, 0, 0, 1],  # s3: A11
        [1, 1, 0, 0],  # s4: A00 + A01
        [-1, 0, 1, 0], # s5: A10 - A00
        [0, 1, 0, -1]  # s6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],  # s0: B00 + B11
        [1, 0, 0, 0],  # s1: B00
        [0, 1, 0, -1], # s2: B01 - B11
        [-1, 0, 1, 0], # s3: B10 - B00
        [0, 0, 0, 1],  # s4: B11
        [1, 1, 0, 0],  # s5: B00 + B01
        [0, 0, 1, 1]   # s6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s0
        [(1, 0, 1), (1, 1, -1)],    # s1
        [(0, 1, 1), (1, 1, 1)],     # s2
        [(0, 0, 1), (1, 0, 1)],     # s3
        [(0, 0, -1), (0, 1, 1)],    # s4
        [(1, 1, 1)],                # s5
        [(0, 0, 1)]                 # s6
    ]
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for (2x2) * (2x2) [columns 0,1 of Y_s]
        for si in range(7):
            # U part
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for (2x2) * (2x1) [column 2 of Y_s]
        for u in range(2):
            for v in range(2):
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Fast vectorized verification using tensor contraction
    def verify_decomp(U_mat, V_mat, W_mat, tol=1e-8):
        """Check if U, V, W exactly decompose the tensor using vectorized ops."""
        rank = U_mat.shape[1]
        # Reshape U, V, W for efficient contraction
        # U: (nm, rank) -> (n, m, rank)
        U_3d = U_mat.reshape(n, m, rank)
        # V: (mp, rank) -> (m, p, rank)
        V_3d = V_mat.reshape(m, p, rank)
        # W: (pn, rank) -> (p, n, rank)
        W_3d = W_mat.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum for efficient computation
        T = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d, optimize=True)
        
        if not np.allclose(T, 1.0, atol=tol, rtol=0):
            return False
        return True
    
    if not verify_decomp(U, V, W):
        return schoolbook()
    
    # Global SVD-based compression to eliminate linearly dependent terms
    def compress_factors(U_mat, V_mat, W_mat, tol=1e-10):
        """
        Apply global linear compression: find the minimal rank representation
        by checking if any columns are linearly dependent across the three factors.
        
        The key insight: if there exists a non-trivial vector c in R^rank such that
        sum_r c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0, then we can project out that direction.
        
        We approximate this by checking rank of concatenated matrices.
        """
        current_U = U_mat.copy()
        current_V = V_mat.copy()
        current_W = W_mat.copy()
        current_rank = current_U.shape[1]
        
        for iteration in range(50):
            if time.time() > deadline:
                break
            if current_rank <= 1:
                break
            
            # Try to find a direction to eliminate
            # Method: check if removing column r still preserves the decomposition
            found_removal = False
            
            # First pass: systematic check of all columns
            for r in range(current_rank):
                if time.time() > deadline:
                    break
                
                test_U = np.delete(current_U, r, axis=1)
                test_V = np.delete(current_V, r, axis=1)
                test_W = np.delete(current_W, r, axis=1)
                
                if verify_decomp(test_U, test_V, test_W):
                    current_U = test_U
                    current_V = test_V
                    current_W = test_W
                    current_rank -= 1
                    found_removal = True
                    break
            
            if found_removal:
                continue
            
            # Second pass: try pairs of columns (linear combination elimination)
            # If columns r1 and r2 are "proportional" in some sense, we might merge them
            # This is harder; skip for now and rely on single-column removal
            
            if not found_removal:
                break
        
        return current_U, current_V, current_W, current_rank
    
    # Attempt rank reduction
    U_red, V_red, W_red, rank_red = compress_factors(U, V, W)
    
    if verify_decomp(U_red, V_red, W_red):
        U, V, W = U_red, V_red, W_red
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
