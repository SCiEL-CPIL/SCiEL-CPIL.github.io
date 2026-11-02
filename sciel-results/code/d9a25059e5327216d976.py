"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    dim_u = n * m  # 16
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    deadline = 142  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def verify_np(U, V, W, tol=1e-9):
        """Check if the decomposition is exact using numpy."""
        rank = U.shape[1]
        # Build the tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # U: (dim_u, rank), V: (dim_v, rank), W: (dim_w, rank)
        # For each r, outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Sum over r
        T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        # Check T[i,j,k] == 1 for all valid (i,j,k)
        # i = a_row*m + a_col (0..15), j = b_row*p + b_col (0..19), k = c_col*n + c_row (0..19)
        # The tensor should be 1 when a_col == b_row and c_col == a_row... 
        # Actually: C[a,c] = sum_b A[a,b]*B[b,c]
        # T indexed by (a,b,c) should be 1 for all a,b,c (since the multiplication tensor has 1 at each (a,b,c))
        # Wait no. The matrix multiplication tensor T has T[a,b,c] = 1 for all a in [0,n), b in [0,m), c in [0,p)
        # because C[a,c] = sum_b A[a,b] * B[b,c]
        # So T should be all ones.
        if np.max(np.abs(T - 1.0)) > tol:
            return False
        return True
    
    def verify(U, V, W):
        """Check if the decomposition is exact (list-of-lists input)."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        return verify_np(U_np, V_np, W_np)
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p  # 80
    
    # Strassen 2x2 decomposition
    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition as numpy arrays.
        U: (4,7), V: (4,7), W: (4,7)
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # r=0: M1 = (A11+A22)*(B11+B22)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # r=1: M2 = (A21+A22)*B11
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # r=2: M3 = A11*(B12-B22)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1
        
        # r=3: M4 = A22*(B21-B11)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # r=4: M5 = (A11+A12)*B22
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        # r=5: M6 = (A21-A11)*(B11+B12)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # r=6: M7 = (A12-A22)*(B21+B22)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[3,6] = 1
        
        return U, V, W
    
    # Build 4x4*4x5 using Strassen for 2x2*2x2 blocks and schoolbook for 2x2*2x3
    def build_block_decomposition():
        """Build 4x4*4x5 decomposition using Strassen(7) for 2x2*2x2 and schoolbook(12) for 2x2*2x3.
        Total rank: 4*7 + 4*12 = 28 + 48 = 76.
        Returns numpy arrays U(16,76), V(20,76), W(20,76).
        """
        sU, sV, sW = strassen_2x2()
        U_all = np.zeros((dim_u, 76))
        V_all = np.zeros((dim_v, 76))
        W_all = np.zeros((dim_w, 76))
        col = 0
        
        # Define block mappings
        # A blocks (2x2 each): A11=rows0-1,cols0-1; A12=rows0-1,cols2-3; A21=rows2-3,cols0-1; A22=rows2-3,cols2-3
        # B blocks: B11=rows0-1,cols0-1; B12=rows0-1,cols2-4; B21=rows2-3,cols0-1; B22=rows2-3,cols2-4
        # C blocks: C11=rows0-1,cols0-1; C12=rows0-1,cols2-4; C21=rows2-3,cols0-1; C22=rows2-3,cols2-4
        
        def add_strassen_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, c_cols):
            """Add a Strassen decomposition for a 2x2*2x2 block.
            c_cols is the number of columns in the C block (2 for 2x2, but we only use 2x2 here).
            """
            nonlocal col
            # Map Strassen local indices to global indices
            # Strassen U local: [0,1,2,3] = [[r0c0,r0c1],[r1c0,r1c1]]
            # Global U index: (a_row_off + lr) * m + (a_col_off + lc)
            a_map = []
            for lr in range(2):
                for lc in range(2):
                    a_map.append((a_row_off + lr) * m + (a_col_off + lc))
            # Strassen V local: [0,1,2,3] = [[r0c0,r0c1],[r1c0,r1c1]]
            # Global V index: (b_row_off + lr) * p + (b_col_off + lc)
            b_map = []
            for lr in range(2):
                for lc in range(2):
                    b_map.append((b_row_off + lr) * p + (b_col_off + lc))
            # Strassen W local: [0,1,2,3] = [[r0c0,r1c0],[r0c1,r1c1]] (W[k*2+i])
            # W local index = lc * 2 + lr where lc is col, lr is row
            # Global W index: (c_col_off + lc) * n + (c_row_off + lr)
            c_map = []
            for lc in range(2):
                for lr in range(2):
                    c_map.append((c_col_off + lc) * n + (c_row_off + lr))
            
            for r in range(7):
                for li, val in enumerate(sU[:, r]):
                    if val != 0:
                        U_all[a_map[li], col] = val
                for lj, val in enumerate(sV[:, r]):
                    if val != 0:
                        V_all[b_map[lj], col] = val
                for lk, val in enumerate(sW[:, r]):
                    if val != 0:
                        W_all[c_map[lk], col] = val
                col += 1
        
        def add_schoolbook_block(a_row_off, a_col_off, a_rows, a_cols, b_row_off, b_col_off, b_rows, b_cols, c_row_off, c_col_off):
            """Add schoolbook terms for a block multiplication."""
            nonlocal col
            for ar in range(a_rows):
                for bc in range(b_cols):
                    for br in range(b_rows):
                        # C[a_row_off+ar, c_col_off+bc] += A[a_row_off+ar, a_col_off+br] * B[b_row_off+br, b_col_off+bc]
                        U_idx = (a_row_off + ar) * m + (a_col_off + br)
                        V_idx = (b_row_off + br) * p + (b_col_off + bc)
                        W_idx = (c_col_off + bc) * n + (c_row_off + ar)
                        U_all[U_idx, col] = 1
                        V_all[V_idx, col] = 1
                        W_all[W_idx, col] = 1
                        col += 1
        
        # 4 Strassen blocks for 2x2*2x2:
        # C11 += A11*B11
        add_strassen_block(0, 0, 0, 0, 0, 0, 2)
        # C11 += A12*B21
        add_strassen_block(0, 2, 2, 0, 0, 0, 2)
        # C21 += A21*B11
        add_strassen_block(2, 0, 0, 0, 2, 0, 2)
        # C21 += A22*B21
        add_strassen_block(2, 2, 2, 0, 2, 0, 2)
        
        # 4 schoolbook blocks for 2x2*2x3:
        # C12 += A11*B12: A rows 0-1, cols 0-1; B rows 0-1, cols 2-4; C rows 0-1, cols 2-4
        add_schoolbook_block(0, 0, 2, 2, 0, 2, 2, 3, 0, 2)
        # C12 += A12*B22: A rows 0-1, cols 2-3; B rows 2-3, cols 2-4; C rows 0-1, cols 2-4
        add_schoolbook_block(0, 2, 2, 2, 2, 2, 2, 3, 0, 2)
        # C22 += A21*B12: A rows 2-3, cols 0-1; B rows 0-1, cols 2-4; C rows 2-3, cols 2-4
        add_schoolbook_block(2, 0, 2, 2, 0, 2, 2, 3, 2, 2)
        # C22 += A22*B22: A rows 2-3, cols 2-3; B rows 2-3, cols 2-4; C rows 2-3, cols 2-4
        add_schoolbook_block(2, 2, 2, 2, 2, 2, 2, 3, 2, 2)
        
        return U_all, V_all, W_all, col
    
    # Build and verify the block decomposition
    U_block, V_block, W_block, actual_rank = build_block_decomposition()
    if verify_np(U_block, V_block, W_block):
        if actual_rank < best_rank:
            best = {"U": U_block.tolist(), "V": V_block.tolist(), "W": W_block.tolist()}
            best_rank = actual_rank
    
    # Try ALS optimization to reduce rank further
    if elapsed() < deadline - 10:
        # Try to find a lower rank decomposition using ALS
        # Start from the block decomposition and try to reduce rank
        
        def round_to_half_int(x):
            """Round a value to the nearest half-integer."""
            return round(2 * x) / 2.0
        
        def try_als(U_init, V_init, W_init, target_rank, max_iters=200, tol=1e-8):
            """Try ALS optimization to find a rank-target_rank decomposition.
            Returns (U, V, W) as numpy arrays or None if failed.
            """
            U = U_init[:, :target_rank].copy()
            V = V_init[:, :target_rank].copy()
            W = W_init[:, :target_rank].copy()
            
            # Normalize to avoid scale issues
            for r in range(target_rank):
                norms = [np.linalg.norm(U[:,r]), np.linalg.norm(V[:,r]), np.linalg.norm(W[:,r])]
                if all(nv < 1e-10 for nv in norms):
                    continue
                prod = np.prod([nv if nv > 1e-10 else 1e-10 for nv in norms])
                target_norm = prod ** (1.0/3.0)
                for arr, nv in zip([U, V, W], norms):
                    if nv > 1e-10:
                        arr[:, r] *= target_norm / nv
            
            for it in range(max_iters):
                if elapsed() > deadline - 5:
                    return None
                
                # Update U: for each i, U[i,:] should satisfy sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
                # Fix V and W, solve for U
                # T[i,j,k] = sum_r U[i,r] * (V[j,r]*W[k,r])
                # For fixed i: T[i,:,:] = sum_r U[i,r] * outer(V[:,r], W[:,r])
                # This is a linear system: T[i] = U[i] @ (V*W)^T where (V*W)[j,k,r] = V[j,r]*W[k,r]
                
                # Build the design matrix for U update
                # For each i, we need: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k] for all j,k
                # Vectorize: T[i] (flattened) = U[i] @ M_i where M_i[r, j*p+k] = V[j,r]*W[k,r]
                
                # Actually, let's do it more efficiently:
                # The constraint is: for all (i,j,k): sum_r U[i,r]*V[j,r]*W[k,r] = 1
                # Fix V, W. For each i, solve: sum_r U[i,r] * V[j,r]*W[k,r] = 1 for all j,k
                # This is: U[i] @ (V ⊙ W) = 1 where (V⊙W)[j,k,r] = V[j,r]*W[k,r]
                # Flattened: U[i] (1 x rank) times matrix of shape (rank x dim_v*dim_w)
                
                # Build VW matrix: shape (rank, dim_v, dim_w) -> flatten to (rank, dim_v*dim_w)
                VW = np.einsum('jr,kr->jkr', V, W).reshape(target_rank, dim_v * dim_w)
                
                # For each i, solve: U[i] @ VW = ones(dim_v*dim_w)
                # This is underdetermined if target_rank < dim_v*dim_w
                # Use least squares
                for i in range(dim_u):
                    # Solve U[i] @ VW = 1
                    # VW is (rank, dim_v*dim_w), we want U[i] (1 x rank)
                    # U[i] = argmin ||U[i] @ VW - 1||^2
                    # Solution: U[i] = (VW^T @ VW)^{-1} @ VW^T @ 1 = VW^+ @ 1
                    try:
                        U[i] = np.linalg.lstsq(VW.T, np.ones(dim_v * dim_w), rcond=None)[0]
                    except:
                        return None
                
                # Update V similarly
                UW = np.einsum('ir,kr->ikr', U, W).reshape(target_rank, dim_u * dim_w)
                for j in range(dim_v):
                    try:
                        V[j] = np.linalg.lstsq(UW.T, np.ones(dim_u * dim_w), rcond=None)[0]
                    except:
                        return None
                
                # Update W similarly
                UV = np.einsum('ir,jr->ijr', U, V).reshape(target_rank, dim_u * dim_v)
                for k in range(dim_w):
                    try:
                        W[k] = np.linalg.lstsq(UV.T, np.ones(dim_u * dim_v), rcond=None)[0]
                    except:
                        return None
                
                # Check convergence
                T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.max(np.abs(T_check - 1.0))
                if err < tol:
                    break
            
            # Final check
            T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.max(np.abs(T_check - 1.0)) > 1e-6:
                return None
            
            return U, V, W
        
        def try_round_and_verify(U, V, W, target_rank):
            """Round to half-integers and verify."""
            U_r = np.array([[round_to_half_int(x) for x in U[i]] for i in range(dim_u)])
            V_r = np.array([[round_to_half_int(x) for x in V[j]] for j in range(dim_v)])
            W_r = np.array([[round_to_half_int(x) for x in W[k]] for k in range(dim_w)])
            
            if verify_np(U_r, V_r, W_r):
                return U_r, V_r, W_r
            return None
        
        # Try reducing rank from best_rank downward
        current_U = np.array(best["U"], dtype=np.float64)
        current_V = np.array(best["V"], dtype=np.float64)
        current_W = np.array(best["W"], dtype=np.float64)
        
        for target_rank in range(best_rank - 1, max(40, best_rank - 15), -1):
            if elapsed() > deadline - 10:
                break
            
            # Try a few restarts
            found = False
            for restart in range(3):
                if elapsed() > deadline - 8:
                    break
                
                # Perturb initialization
                U_init = current_U[:, :target_rank].copy()
                V_init = current_V[:, :target_rank].copy()
                W_init = current_W[:, :target_rank].copy()
                
                if restart > 0:
                    # Add small random perturbation
                    U_init += 0.01 * np.random.randn(dim_u, target_rank)
                    V_init += 0.01 * np.random.randn(dim_v, target_rank)
                    W_init += 0.01 * np.random.randn(dim_w, target_rank)
                
                result = try_als(U_init, V_init, W_init, target_rank, max_iters=100)
                if result is None:
                    continue
                
                U_opt, V_opt, W_opt = result
                
                # Try rounding
                rounded = try_round_and_verify(U_opt, V_opt, W_opt, target_rank)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    if target_rank < best_rank:
                        best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = target_rank
                    found = True
                    break
            
            if found:
                # Update current for next iteration
                current_U = np.array(best["U"], dtype=np.float64)
                current_V = np.array(best["V"], dtype=np.float64)
                current_W = np.array(best["W"], dtype=np.float64)
            else:
                # If we can't find a solution at this rank, try the next lower rank
                continue
    
    return best

# EVOLVE-BLOCK-END
