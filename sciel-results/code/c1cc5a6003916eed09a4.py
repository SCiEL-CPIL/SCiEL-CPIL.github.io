"""Matrix multiplication 3x5x5 - Rank-minimal exact decomposition search."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Time budget with safety margin
    start_time = time.time()
    deadline = 120.0
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some valid i,j,k
    # Otherwise 0
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
    
    # Schoolbook decomposition (rank = n*m*p = 75)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Verify a decomposition
    def verify(U, V, W):
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.outer(U[:, r], V[:, r]).reshape(nm, mp, 1) * W[:, r].reshape(1, 1, pn)
        return np.array_equal(T_recon, T)
    
    # Strassen 2x2x2 decomposition
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    def strassen_2x2_factors():
        """
        Returns U, V, W factor matrices for 2x2x2 Strassen decomposition.
        U has shape (4, 7) - rows are (i,j) pairs for A
        V has shape (4, 7) - rows are (j,k) pairs for B  
        W has shape (4, 7) - rows are (k,i) pairs for C
        
        Indexing: u = i*2+j, v = j*2+k, w = k*2+i
        """
        # 7 multiplications
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        
        # M1 = (A11+A22)*(B11+B22)
        # A factors: A11 at (0,0), A22 at (1,1)
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        # B factors: B11 at (0,0), B22 at (1,1)
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        # C factors: C11 += M1, C22 += M1
        # C11 is at (0,0) -> w = 0*2+0 = 0
        # C22 is at (1,1) -> w = 1*2+1 = 3
        W[0, 0] = 1  # C11
        W[3, 0] = 1  # C22
        
        # M2 = (A21+A22)*B11
        U[2, 1] = 1  # A21
        U[3, 1] = 1  # A22
        V[0, 1] = 1  # B11
        # C21 += M2, C22 -= M2
        # C21 is at (0,1) -> w = 0*2+1 = 1
        # C22 is at (1,1) -> w = 1*2+1 = 3
        W[1, 1] = 1  # C21
        W[3, 1] = -1  # C22
        
        # M3 = A11*(B12-B22)
        U[0, 2] = 1  # A11
        V[1, 2] = 1  # B12
        V[3, 2] = -1  # B22
        # C12 += M3, C22 += M3
        # C12 is at (1,0) -> w = 1*2+0 = 2
        # C22 is at (1,1) -> w = 1*2+1 = 3
        W[2, 2] = 1  # C12
        W[3, 2] = 1  # C22
        
        # M4 = A22*(B21-B11)
        U[3, 3] = 1  # A22
        V[2, 3] = 1  # B21
        V[0, 3] = -1  # B11
        # C11 += M4, C21 += M4
        # C11 is at (0,0) -> w = 0*2+0 = 0
        # C21 is at (0,1) -> w = 0*2+1 = 1
        W[0, 3] = 1  # C11
        W[1, 3] = 1  # C21
        
        # M5 = (A11+A12)*B22
        U[0, 4] = 1  # A11
        U[1, 4] = 1  # A12
        V[3, 4] = 1  # B22
        # C11 -= M5, C12 += M5
        # C11 is at (0,0) -> w = 0*2+0 = 0
        # C12 is at (1,0) -> w = 1*2+0 = 2
        W[0, 4] = -1  # C11
        W[2, 4] = 1  # C12
        
        # M6 = (A21-A11)*(B11+B12)
        U[2, 5] = 1  # A21
        U[0, 5] = -1  # A11
        V[0, 5] = 1  # B11
        V[1, 5] = 1  # B12
        # This M6 doesn't appear in standard Strassen C output
        # Actually, let me recheck the Strassen formulas
        # Standard Strassen:
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        # So M6 is not used! Let me fix this.
        # Actually, I made an error. Let me redo the Strassen decomposition properly.
        
        # Let me restart with correct Strassen:
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        
        # M1 = (A11+A22)*(B11+B22)
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        W[0, 0] = 1  # C11
        W[3, 0] = 1  # C22
        
        # M2 = (A21+A22)*B11
        U[2, 1] = 1  # A21
        U[3, 1] = 1  # A22
        V[0, 1] = 1  # B11
        W[1, 1] = 1  # C21
        W[3, 1] = -1  # C22
        
        # M3 = A11*(B12-B22)
        U[0, 2] = 1  # A11
        V[1, 2] = 1  # B12
        V[3, 2] = -1  # B22
        W[2, 2] = 1  # C12
        W[3, 2] = 1  # C22
        
        # M4 = A22*(B21-B11)
        U[3, 3] = 1  # A22
        V[2, 3] = 1  # B21
        V[0, 3] = -1  # B11
        W[0, 3] = 1  # C11
        W[1, 3] = 1  # C21
        
        # M5 = (A11+A12)*B22
        U[0, 4] = 1  # A11
        U[1, 4] = 1  # A12
        V[3, 4] = 1  # B22
        W[0, 4] = -1  # C11
        W[2, 4] = 1  # C12
        
        # M6 = (A21-A11)*(B11+B12) -- not used in C, skip
        # M7 = (A12-A22)*(B21+B22)
        U[1, 6] = 1  # A12
        U[3, 6] = -1  # A22
        V[2, 6] = 1  # B21
        V[3, 6] = 1  # B22
        W[0, 6] = -1  # C11
        W[3, 6] = 1  # C22
        
        return U, V, W
    
    # Build 3x5x5 decomposition using block structure
    # Split 5x5 into blocks: use 2x2 blocks where possible
    # 5 = 2 + 2 + 1, so we have a 3x3 block structure
    
    def build_block_decomposition():
        """
        Build a decomposition using 3x3 block structure with Strassen for 2x2 blocks.
        5x5 matrix split into blocks of sizes [2, 2, 1]
        """
        block_sizes = [2, 2, 1]
        num_blocks = 3
        
        # Precompute Strassen factors for 2x2 blocks
        S_U, S_V, S_W = strassen_2x2_factors()
        
        # For each pair of blocks (i_block, j_block), we need to compute
        # C[i_block, k_block] = sum_j_block A[i_block, j_block] * B[j_block, k_block]
        
        # We'll build the full U, V, W matrices
        # Total rank will be sum over all block pairs of the rank needed for that pair
        
        # For 2x2 blocks, use Strassen (rank 7)
        # For 2x1, 1x2, 1x1 blocks, use schoolbook
        
        # Let's build it systematically
        U_list = []
        V_list = []
        W_list = []
        
        # Block indices: 0, 1, 2 with sizes 2, 2, 1
        # For each (i, j, k) block triple where A[i,j] * B[j,k] contributes to C[i,k]
        
        for i_block in range(num_blocks):
            for j_block in range(num_blocks):
                for k_block in range(num_blocks):
                    ai_size = block_sizes[i_block]
                    aj_size = block_sizes[j_block]
                    bj_size = block_sizes[j_block]
                    bk_size = block_sizes[k_block]
                    
                    # This is A[i_block, j_block] (size ai_size x aj_size) * B[j_block, k_block] (size bj_size x bk_size)
                    # contributing to C[i_block, k_block] (size bk_size x ai_size)
                    
                    # For simplicity, use schoolbook for all except when both are 2x2
                    if ai_size == 2 and aj_size == 2 and bj_size == 2 and bk_size == 2:
                        # Use Strassen
                        rank_this = 7
                        U_block = np.zeros((ai_size * aj_size, rank_this), dtype=np.int64)
                        V_block = np.zeros((bj_size * bk_size, rank_this), dtype=np.int64)
                        W_block = np.zeros((bk_size * ai_size, rank_this), dtype=np.int64)
                        
                        # Map Strassen 2x2 factors to this block
                        # Strassen U has shape (4, 7) where row = i*2+j
                        # Our block has same structure
                        U_block = S_U.copy()
                        V_block = S_V.copy()
                        W_block = S_W.copy()
                    else:
                        # Schoolbook
                        rank_this = ai_size * aj_size
                        U_block = np.zeros((ai_size * aj_size, rank_this), dtype=np.int64)
                        V_block = np.zeros((bj_size * bk_size, rank_this), dtype=np.int64)
                        W_block = np.zeros((bk_size * ai_size, rank_this), dtype=np.int64)
                        
                        col = 0
                        for ii in range(ai_size):
                            for jj in range(aj_size):
                                u_idx = ii * aj_size + jj
                                v_idx = jj * bk_size  # B[j_block, k_block] starts at row jj*bk_size
                                # Wait, B[j_block, k_block] is a block of B, so within the full B matrix
                                # B is 5x5, split into blocks. B[j_block, k_block] is at rows (sum of previous block sizes in j), cols (sum of previous in k)
                                # This is getting complex. Let me use a different approach.
                                pass
                        # For now, just use simple 1x1 blocks for non-2x2 cases
                        # Actually, let me simplify: use schoolbook for everything except 2x2x2 blocks
                        rank_this = ai_size * bj_size * bk_size
                        U_block = np.zeros((ai_size * aj_size, rank_this), dtype=np.int64)
                        V_block = np.zeros((bj_size * bk_size, rank_this), dtype=np.int64)
                        W_block = np.zeros((bk_size * ai_size, rank_this), dtype=np.int64)
                        
                        col = 0
                        for ii in range(ai_size):
                            for jj in range(aj_size):
                                for kk in range(bk_size):
                                    u_idx = ii * aj_size + jj
                                    # v_idx within B[j_block, k_block]
                                    v_idx = jj * bk_size + kk
                                    # w_idx within C[i_block, k_block]
                                    w_idx = kk * ai_size + ii
                                    U_block[u_idx, col] = 1
                                    V_block[v_idx, col] = 1
                                    W_block[w_idx, col] = 1
                                    col += 1
                    
                    U_list.append(U_block)
                    V_list.append(V_block)
                    W_list.append(W_block)
        
        # Concatenate
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        
        return U, V, W
    
    # Actually, the block approach above is too complex. Let me use a simpler strategy:
    # Directly construct a decomposition that exploits Strassen-like savings
    
    # Strategy: Use the fact that 5x5 can be viewed as having structure
    # For 3x5x5, we have 3 rows of A, each of length 5
    
    # Let me try a different approach: use a known result that 5x5 matrix multiplication
    # can be done with fewer than 25 multiplications
    
    # For now, let's return the schoolbook as baseline
    best_U, best_V, best_W, best_rank = get_schoolbook()
    
    # Check if we have time to do any search
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Try a structured approach: exploit that 5 = 2+2+1
    # Use Strassen for the 2x2 blocks and schoolbook for the rest
    
    def build_strassen_block_decomp():
        """
        Build decomposition using block structure 5 = 2+2+1
        """
        # Block sizes
        bs = [2, 2, 1]
        nb = 3
        
        # Compute cumulative offsets
        offsets = [0]
        for i in range(nb - 1):
            offsets.append(offsets[-1] + bs[i])
        
        # Strassen 2x2 factors
        S_U, S_V, S_W = strassen_2x2_factors()
        
        U_parts = []
        V_parts = []
        W_parts = []
        
        for i_b in range(nb):
            for j_b in range(nb):
                for k_b in range(nb):
                    ai = bs[i_b]
                    aj = bs[j_b]
                    bj = bs[j_b]
                    bk = bs[k_b]
                    
                    # A[i_b, j_b] is ai x aj
                    # B[j_b, k_b] is bj x bk
                    # C[i_b, k_b] is bk x ai
                    # (Note: C[i,k] has shape bk x ai because C = A*B, C[i,k] = sum_j A[i,j]*B[j,k])
                    
                    if ai == 2 and bj == 2 and bk == 2:
                        # Use Strassen: rank 7
                        U_part = S_U.copy()
                        V_part = S_V.copy()
                        W_part = S_W.copy()
                    else:
                        # Schoolbook: rank ai*bj*bk
                        rank = ai * bj * bk
                        U_part = np.zeros((ai * aj, rank), dtype=np.int64)
                        V_part = np.zeros((bj * bk, rank), dtype=np.int64)
                        W_part = np.zeros((bk * ai, rank), dtype=np.int64)
                        
                        col = 0
                        for ii in range(ai):
                            for jj in range(aj):
                                for kk in range(bk):
                                    u_idx = ii * aj + jj
                                    v_idx = jj * bk + kk
                                    w_idx = kk * ai + ii
                                    U_part[u_idx, col] = 1
                                    V_part[v_idx, col] = 1
                                    W_part[w_idx, col] = 1
                                    col += 1
                    
                    U_parts.append(U_part)
                    V_parts.append(V_part)
                    W_parts.append(W_part)
        
        U = np.hstack(U_parts)
        V = np.hstack(V_parts)
        W = np.hstack(W_parts)
        
        return U, V, W
    
    # Try the Strassen block decomposition
    try:
        U_str, V_str, W_str = build_strassen_block_decomp()
        if verify(U_str, V_str, W_str):
            if U_str.shape[1] < best_rank:
                best_U, best_V, best_W, best_rank = U_str, V_str, W_str, U_str.shape[1]
    except Exception as e:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
