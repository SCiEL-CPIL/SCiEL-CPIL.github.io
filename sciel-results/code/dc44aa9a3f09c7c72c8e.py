import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    # Dimensions: U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # n=2, m=4, p=4
    # U: 8 x R
    # V: 16 x R
    # W: 8 x R
    
    # 1. Construct the Target Tensor T
    # T[i, j, k] = 1 if A[i] * B[j] contributes to C[k]
    # i = row_a * m + col_a
    # j = row_b * p + col_b
    # k = col_c * n + row_c (Official trace convention: W[k*n+i, r] implies k is the "output index" 
    # usually C is n x p. Here C is 2x4.
    # Standard: C = A @ B. C[r, c] = sum_k A[r, k] B[k, c].
    # A is 2x4 (n x m). B is 4x4 (m x p). C is 2x4 (n x p).
    # Index i in U corresponds to A's entries: i = r_a * m + c_a.
    # Index j in V corresponds to B's entries: j = r_b * p + c_b.
    # Index k in W corresponds to C's entries: k = c_c * n + r_c? 
    # The prompt says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually, the tensor T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
    # If W is indexed by k, and the convention is W[k*n+i], this likely refers to the storage of the 
    # factor matrix W itself? No, W is a matrix of shape (p*n, rank).
    # Let's look at the seed code's T construction:
    # k = col_c * n + row_c.
    # This means the index k for W corresponds to the output element C[row_c, col_c].
    # So W has rows indexed by output positions.
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for row_a in range(n):
        for col_a in range(m):
            i = row_a * m + col_a
            for row_b in range(m):
                for col_b in range(p):
                    j = row_b * p + col_b
                    for row_c in range(n):
                        for col_c in range(p):
                            k = col_c * n + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T[i, j, k] = 1.0

    T_flat = T.flatten()
    dim_T = T.size # 8 * 16 * 8 = 1024

    # 2. Construct Rank-28 Baseline using Kronecker Product
    # Strassen 2x2x2 decomposition
    # U_s (4x7), V_s (4x7), W_s (4x7)
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)
    
    V_s = np.array([
        [1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1]
    ], dtype=float)
    
    # Derive W_s for 2x2x2
    T_2 = np.zeros((4, 4, 4), dtype=float)
    for ra in range(2):
        for ca in range(2):
            ii = ra * 2 + ca
            for rb in range(2):
                for cb in range(2):
                    jj = rb * 2 + cb
                    for rc in range(2):
                        for cc in range(2):
                            kk = cc * 2 + rc
                            if ca == rb and ra == rc and cb == cc:
                                T_2[ii, jj, kk] = 1.0
    
    # Solve for W_s
    M_sys = np.zeros((64, 28), dtype=float)
    T_2_vec = T_2.flatten()
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                eq = ii * 16 + jj * 4 + kk
                for s in range(7):
                    M_sys[eq, kk * 7 + s] = U_s[ii, s] * V_s[jj, s]
    W_s, _, _, _ = np.linalg.lstsq(M_sys, T_2_vec, rcond=None)
    W_s = W_s.reshape(4, 7)

    # Build 2x4x4 from 2x2x2 blocks
    # We partition the 4x4 matrices into 2x2 blocks.
    # A (2x4) is partitioned into two 2x2 blocks horizontally: A = [A_00, A_01]
    # B (4x4) is partitioned into four 2x2 blocks:
    #   B_00 B_01
    #   B_10 B_11
    # C (2x4) is partitioned into two 2x2 blocks horizontally: C = [C_00, C_01]
    
    # The multiplication is:
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    
    # This requires 4 multiplications of 2x2 matrices.
    # Each 2x2 multiplication uses rank 7.
    # Total rank = 4 * 7 = 28.
    
    U_28 = np.zeros((8, 28), dtype=float)
    V_28 = np.zeros((16, 28), dtype=float)
    W_28 = np.zeros((8, 28), dtype=float)
    
    # Block mapping:
    # A_00: rows 0-1, cols 0-1 of A (i indices 0-3)
    # A_01: rows 0-1, cols 2-3 of A (i indices 4-7)
    # B_00: rows 0-1, cols 0-1 of B (j indices 0-3)
    # B_10: rows 2-3, cols 0-1 of B (j indices 8-11)
    # B_01: rows 0-1, cols 2-3 of B (j indices 4-7)
    # B_11: rows 2-3, cols 2-3 of B (j indices 12-15)
    # C_00: rows 0-1, cols 0-1 of C (k indices 0-3) -> k = c*n+r. c in 0-1, r in 0-1.
    # C_01: rows 0-1, cols 2-3 of C (k indices 4-7) -> c in 2-3, r in 0-1.
    
    # Block 1: C_00 += A_00 B_00
    # U indices: 0-3 (A_00)
    # V indices: 0-3 (B_00)
    # W indices: 0-3 (C_00)
    U_28[0:4, 0:7] = U_s
    V_28[0:4, 0:7] = V_s
    W_28[0:4, 0:7] = W_s
    
    # Block 2: C_00 += A_01 B_10
    # U indices: 4-7 (A_01)
    # V indices: 8-11 (B_10)
    # W indices: 0-3 (C_00)
    U_28[4:8, 7:14] = U_s
    V_28[8:12, 7:14] = V_s
    W_28[0:4, 7:14] = W_s
    
    # Block 3: C_01 += A_00 B_01
    # U indices: 0-3 (A_00)
    # V indices: 4-7 (B_01)
    # W indices: 4-7 (C_01)
    U_28[0:4, 14:21] = U_s
    V_28[4:8, 14:21] = V_s
    W_28[4:8, 14:21] = W_s
    
    # Block 4: C_01 += A_01 B_11
    # U indices: 4-7 (A_01)
    # V indices: 12-15 (B_11)
    # W indices: 4-7 (C_01)
    U_28[4:8, 21:28] = U_s
    V_28[12:16, 21:28] = V_s
    W_28[4:8, 21:28] = W_s
    
    # Verify Rank 28
    def verify(U, V, W):
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # Vectorized check
        # Compute tensor product
        # Use einsum for speed
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(computed - T))
    
    err_28 = verify(U_28, V_28, W_28)
    if err_28 > 1e-9:
        # Fallback to schoolbook if Strassen construction failed (should not happen)
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # 3. Search for Lower Ranks (27, 26, ...)
    # Pre-compute Kronecker matrices for ALS
    # Equation: T_flat = (W^T \otimes V^T) vec(U^T) ? No.
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Let's fix V and W. We want to solve for U.
    # T[:, j, k] = U @ (V[j, :] * W[k, :])^T
    # This is a linear system for U.
    # Let M_U be the matrix such that vec(U) = M_U^+ vec(T)
    # Actually, it's easier to construct the big sparse matrix A_U for the full system.
    # A_U has size (N_T) x (N_U). N_T = 1024, N_U = 8*R.
    # Row idx: t_idx = (i * 16 + j) * 8 + k
    # Col idx: u_idx = i * R + r
    # A_U[t_idx, u_idx] = V[j, r] * W[k, r]
    
    # To make this fast, we can construct A_U using Kronecker products of sparse matrices.
    # However, V and W change in each iteration.
    # Instead of building the full 1024 x (8R) matrix every time, we can exploit structure.
    # But for R <= 28, 1024 x 224 is small. Building it in Python loop is slow.
    # Let's try to build it using vectorized operations or sparse Kronecker.
    
    # Alternative: Use the fact that the system is linear.
    # We can precompute the "structure" of the indices.
    
    # Let's define a function to build A_U given V, W
    def build_A_U(V, W, R):
        # V is 16 x R, W is 8 x R
        # A_U is 1024 x (8*R)
        # We can build this by iterating over r (R times)
        # For a fixed r, A_U[:, r_block] = (W[:, r] \otimes V[:, r]) ... no.
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # Let's group by r.
        # For each r, we have a rank-1 contribution to the tensor.
        # The matrix A_U for this r is:
        # Rows are (i,j,k). Cols are i.
        # A_U_r[i, j, k, i'] = V[j, r] W[k, r] * delta_{i, i'}
        # This is (W[:, r] \otimes V[:, r]) \otimes I_8 ?
        # Let's check dimensions.
        # W[:, r] is 8x1. V[:, r] is 16x1.
        # Kronecker product: (8*16) x (1*1) = 128x1.
        # We need to map this to the full 1024 rows.
        # The row index is (i, j, k).
        # If we fix r, the term is U[i, r] * V[j, r] * W[k, r].
        # So for a fixed r, the column block of A_U corresponding to U[:, r] is:
        # A_U_block_r has shape 1024 x 8.
        # A_U_block_r[(i,j,k), i'] = V[j, r] * W[k, r] if i == i' else 0.
        # This can be constructed as:
        # Let M_r be a 128x1 matrix: M_r[j, k] = V[j, r] * W[k, r]. (Flattened j,k)
        # Then A_U_block_r = np.kron(np.eye(8), M_r) ?
        # np.kron(I_8, M_r) has size (8*128) x (1*8) = 1024 x 8.
        # Row index in kron: i * 128 + (j*8+k).
        # Our T_flat index is (i*16+j)*8+k = i*128 + j*8 + k.
        # Yes! The ordering matches.
        
        A_U = sp.lil_matrix((1024, 8 * R), dtype=float)
        for r in range(R):
            M_r = (V[:, r] * W[:, r]).reshape(16, 8) # 16x8
            # Flatten M_r to 128x1
            M_r_flat = M_r.flatten().reshape(128, 1)
            Block = sp.kron(sp.eye(8), sp.csr_matrix(M_r_flat))
            A_U[:, r*8:(r+1)*8] = Block
        return A_U.tocsr()

    def build_A_V(U, W, R):
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # Solve for V.
        # A_V shape: 1024 x (16*R)
        # Col idx: j*R + r
        # A_V[(i,j,k), j*R+r] = U[i,r] * W[k,r]
        # For fixed r, block is 1024 x 16.
        # A_V_block_r[(i,j,k), j'] = U[i,r] * W[k,r] if j == j' else 0.
        # Structure: (W[:, r] \otimes I_16 \otimes U[:, r])?
        # Let's check indices.
        # Row: i, j, k.
        # Col: j.
        # Term: U[i,r] W[k,r] delta_{j,j'}.
        # This is a tensor product of U[i,r] (size 8), delta_j (size 16), W[k,r] (size 8).
        # We can construct it as:
        # M_r = U[:, r].reshape(8,1) \otimes W[:, r].reshape(8,1) ? No, j is in the middle.
        # Let's use kron.
        # We want a matrix where row (i,j,k) and col j' is non-zero only if j=j'.
        # Value is U[i,r] W[k,r].
        # Let's construct a 1024 x 16 matrix.
        # We can iterate j' (16 times).
        # For a fixed j', the column is:
        # Col_j'[(i,j,k)] = U[i,r] W[k,r] if j == j' else 0.
        # This is:
        # Let V_j' be a 1024x1 vector.
        # V_j'[i, j, k] = (j==j') * U[i,r] * W[k,r].
        # This can be built as:
        # M_r = np.kron(np.eye(16), np.kron(U[:, r], W[:, r])) ?
        # np.kron(U[:, r], W[:, r]) has size 64x1. (i,k)
        # np.kron(I_16, M_64) has size 1024x16.
        # Row index: j * 64 + (i*8+k).
        # Our T_flat index: (i*16+j)*8+k = i*128 + j*8 + k.
        # Mismatch!
        # T_flat index is i-major, then j, then k.
        # Kron(I_16, M_64) is j-major, then i, then k.
        # We need to permute or build differently.
        
        # Let's just build it using sparse matrix construction with indices.
        # It's safer and R is small.
        A_V = sp.lil_matrix((1024, 16 * R), dtype=float)
        for r in range(R):
            u_r = U[:, r]
            w_r = W[:, r]
            for j in range(16):
                col_idx = j * R + r
                # Rows where j is the second index
                # i in 0..7, k in 0..7
                # row_idx = (i * 16 + j) * 8 + k
                for i in range(8):
                    for k in range(8):
                        row_idx = (i * 16 + j) * 8 + k
                        A_V[row_idx, col_idx] = u_r[i] * w_r[k]
        return A_V.tocsr()

    def build_A_W(U, V, R):
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # Solve for W.
        # A_W shape: 1024 x (8*R)
        # Col idx: k*R + r
        # A_W[(i,j,k), k*R+r] = U[i,r] * V[j,r]
        # For fixed r, block is 1024 x 8.
        # A_W_block_r[(i,j,k), k'] = U[i,r] V[j,r] if k == k' else 0.
        # Similar to V, but k is last.
        A_W = sp.lil_matrix((1024, 8 * R), dtype=float)
        for r in range(R):
            u_r = U[:, r]
            v_r = V[:, r]
            for k in range(8):
                col_idx = k * R + r
                for i in range(8):
                    for j in range(16):
                        row_idx = (i * 16 + j) * 8 + k
                        A_W[row_idx, col_idx] = u_r[i] * v_r[j]
        return A_W.tocsr()

    def solve_als(U, V, W, R, max_iters=20):
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                return U, V, W, False
            
            # Solve U
            A_U = build_A_U(V, W, R)
            U_flat, _, _, _ = spla.lstsq(A_U, T_flat, cond=None)
            U = U_flat.reshape(8, R)
            U = np.round(U * 2) / 2
            
            # Solve V
            A_V = build_A_V(U, W, R)
            V_flat, _, _, _ = spla.lstsq(A_V, T_flat, cond=None)
            V = V_flat.reshape(16, R)
            V = np.round(V * 2) / 2
            
            # Solve W
            A_W = build_A_W(U, V, R)
            W_flat, _, _, _ = spla.lstsq(A_W, T_flat, cond=None)
            W = W_flat.reshape(8, R)
            W = np.round(W * 2) / 2
            
            err = verify(U, V, W)
            if err < 1e-9:
                return U, V, W, True
                
        return U, V, W, False

    best_solution = (U_28, V_28, W_28)
    best_rank = 28

    # Try ranks 27 down to 20
    for R in range(27, 19, -1):
        if time.time() - start_time > deadline:
            break
            
        found = False
        # Multiple restarts
        for restart in range(5):
            if time.time() - start_time > deadline:
                break
                
            # Initialize with random or truncated Strassen
            if restart == 0:
                # Truncate Strassen
                U_init = U_28[:, :R].copy()
                V_init = V_28[:, :R].copy()
                W_init = W_28[:, :R].copy()
            else:
                # Random initialization with small values
                U_init = np.random.normal(0, 0.5, (8, R))
                V_init = np.random.normal(0, 0.5, (16, R))
                W_init = np.random.normal(0, 0.5, (8, R))
                # Round to half-integers to stay in lattice
                U_init = np.round(U_init * 2) / 2
                V_init = np.round(V_init * 2) / 2
                W_init = np.round(W_init * 2) / 2
            
            U_sol, V_sol, W_sol, success = solve_als(U_init, V_init, W_init, R, max_iters=30)
            
            if success:
                best_rank = R
                best_solution = (U_sol, V_sol, W_sol)
                found = True
                break # Found solution for this rank, try lower rank
        
        if found:
            continue # Try lower rank
        else:
            # If not found, we might still try lower ranks, but it's less likely.
            # We'll keep trying.
            pass

    U_out, V_out, W_out = best_solution
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
