import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Strategy: Try to find rank 16 decomposition for 3x2x3 block
    # Then tile 4 such blocks (m split 2+2, p split 3+3)
    # Total rank: 4 * 16 = 64
    
    n_sub = 3
    m_sub = 2
    p_sub = 3
    
    target_block_rank = 16
    
    # Build target tensor for 3x2x3
    # T[i, j, k] = 1 if i in 0..2, j in 0..1, k in 0..2
    # Actually, the tensor is defined by the multiplication.
    # A is 3x2, B is 2x3, C is 3x3.
    # C_ij = sum_k A_ik B_kj
    # Tensor T is sum over i,j,k of C_ij A_ik B_kj
    # We want U, V, W such that U_i * V_j * W_k = 1 for the appropriate terms.
    # Wait, the standard tensor is T(i,j,k) = delta_{i,j,k} in the flattened sense?
    # No, the tensor is the structure tensor of the algebra.
    
    # We will use ALS to find U, V, W for the 3x2x3 block
    # Flattened T is of shape (3*2, 3) ? No.
    # T is shape (n_sub, m_sub, p_sub).
    # We want rank-r decomposition.
    
    def find_block_decomp(n_b, m_b, p_b, target_rank, seed):
        # Initialize
        np.random.seed(seed)
        U_b = np.random.randn(n_b * m_b, target_rank)
        V_b = np.random.randn(m_b * p_b, target_rank)
        W_b = np.random.randn(p_b * n_b, target_rank)
        
        # Target tensor
        # T[i, j, k] is 1 if the term A[i, j] B[j, k] C[k, i] is present?
        # No, the tensor is T = sum_{i,j,k} e_i^A e_j^B e_k^C
        # where the multiplication is C = AB.
        # The entries of the tensor are 1 for all (i,j,k) where the indices match the multiplication.
        # Actually, for matrix multiplication, the tensor is the identity in the "tensor" sense?
        # No, the tensor is sum_{k} A_{ik} B_{kj} C_{ki}?
        # The decomposition is sum_c U_{i, c} V_{j, c} W_{k, c} = 1 if (i,j,k) is a valid multiplication term?
        # No, the condition is:
        # sum_c U_{i,m+j, c} V_{j,p+k, c} W_{k,n+i, c} = 1 for all i,j,k.
        
        # So the target "tensor" T_flat has T_flat[i*m+j, j*p+k, k*n+i] = 1?
        # No, it's a 3-way tensor T[i, j, k] where the indices are flattened?
        # Let's define the 3-way tensor T of size (n*m, m*p, p*n).
        # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
        # Since i,j,k are unique for each a,b,c in the support, T is a permutation matrix-like tensor.
        
        T_block = np.zeros((n_b * m_b, m_b * p_b, p_b * n_b))
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    T_block[i * m_b + j, j * p_b + k, k * n_b + i] = 1.0
        
        # ALS
        for it in range(100):
            # Fix V, W, update U
            # U[a, :] = T[:, a, :] @ (V * W).T ?
            # sum_c U[a,c] V[b,c] W[c_index, c] = T[a,b,c_index]
            # For fixed b, c_index: U[a, :] dot (V[b, :] * W[c_index, :]) = T[a,b,c_index]
            # This is a linear system for U[:, :].
            # Let M_U[b, c_index, :] = V[b, :] * W[c_index, :]
            # Then U @ M_U.T = T_flat ?
            
            # Vectorize T and M
            # T is (n_m, m_p, p_n). Flatten last two?
            # T_flat (n_m, m_p * p_n)
            # M_U (m_p * p_n, rank)
            T_flat_U = T_block.reshape(n_b * m_b, m_b * p_b * p_b * n_b)
            M_U = np.einsum('bc,dc->bd', V_b, W_b).reshape(m_b * p_b, p_b * n_b, target_rank)
            # M_U is (m_p, p_n, rank). We need (m_p * p_n, rank)
            M_U_flat = M_U.reshape(m_b * p_b * p_b * n_b, target_rank)
            
            try:
                U_b = np.linalg.lstsq(M_U_flat.T, T_flat_U.T, rcond=None)[0].T
            except:
                pass
            
            # Fix U, W, update V
            T_flat_V = T_block.reshape(m_b * p_b, n_b * m_b * p_b * n_b).T # (m_p, n_m * p_n)
            # Actually, sum_c U[a,c] V[b,c] W[c_idx,c] = T[a,b,c_idx]
            # V[b, :] dot (U[a, :] * W[c_idx, :]) = T[a,b,c_idx]
            M_V = np.einsum('ac,dc->ad', U_b, W_b).reshape(n_b * m_b, p_b * n_b, target_rank)
            M_V_flat = M_V.reshape(n_b * m_b * p_b * n_b, target_rank)
            T_flat_V = T_block.transpose(1, 0, 2).reshape(m_b * p_b, n_b * m_b * p_b * n_b)
            
            try:
                V_b = np.linalg.lstsq(M_V_flat.T, T_flat_V.T, rcond=None)[0].T
            except:
                pass
            
            # Fix U, V, update W
            T_flat_W = T_block.transpose(2, 0, 1).reshape(p_b * n_b, n_b * m_b * m_b * p_b)
            M_W = np.einsum('ac,bc->ab', U_b, V_b).reshape(n_b * m_b, m_b * p_b, target_rank)
            M_W_flat = M_W.reshape(n_b * m_b * m_b * p_b, target_rank)
            
            try:
                W_b = np.linalg.lstsq(M_W_flat.T, T_flat_W.T, rcond=None)[0].T
            except:
                pass
        
        # Snap to integers
        U_snap = np.rint(U_b)
        V_snap = np.rint(V_b)
        W_snap = np.rint(W_b)
        
        # Verify block
        U_list = [list(map(int, U_snap[i])) for i in range(n_b * m_b)]
        V_list = [list(map(int, V_snap[i])) for i in range(m_b * p_b)]
        W_list = [list(map(int, W_snap[i])) for i in range(p_b * n_b)]
        
        if verify(U_list, V_list, W_list, n_b, m_b, p_b):
            return U_list, V_list, W_list
        return None
    
    # Try to find 3x2x3 block
    block_decomp = None
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        res = find_block_decomp(3, 2, 3, 16, seed)
        if res:
            block_decomp = res
            print(f"Found 3x2x3 block rank 16 with seed {seed}")
            break
    
    if block_decomp:
        U_blk, V_blk, W_blk = block_decomp
        # Tile: m split 2+2, p split 3+3
        # m_blocks: [0,1], [2,3]
        # p_blocks: [0,1,2], [3,4,5]
        
        U_full = [[0] * 64 for _ in range(n * m)]
        V_full = [[0] * 64 for _ in range(m * p)]
        W_full = [[0] * 64 for _ in range(p * n)]
        
        col_offset = 0
        for mi, m_blk in enumerate([[0, 1], [2, 3]]):
            for pi, p_blk in enumerate([[0, 1, 2], [3, 4, 5]]):
                # Copy block
                for r in range(16):
                    c = col_offset + r
                    for i in range(3):
                        for j in range(2):
                            U_full[i * m + m_blk[j]][c] = U_blk[i * 2 + j][r]
                    for j in range(2):
                        for k in range(3):
                            V_full[m_blk[j] * p + p_blk[k]][c] = V_blk[j * 3 + k][r]
                    for k in range(3):
                        for i in range(3):
                            W_full[p_blk[k] * n + i][c] = W_blk[k * 3 + i][r]
                col_offset += 16
        
        if verify(U_full, V_full, W_full, n, m, p):
            if 64 < best_rank:
                best = {"U": U_full, "V": V_full, "W": W_full}
                best_rank = 64
                print("Full decomposition rank 64 verified")
    
    # Fallback to 66 if 64 failed
    if best_rank > 66:
         # Reconstruct the 66 rank solution (previous block logic)
         # ... (reuse code from previous best)
         pass
         
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    def verify_block(U, V, W, n_b, m_b, p_b):
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i * m_b + j, c] * V_arr[j * p_b + k, c] * W_arr[k * n_b + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def find_block_decomp(n_b, m_b, p_b, target_rank, seed):
        np.random.seed(seed)
        U_b = np.random.randn(n_b * m_b, target_rank)
        V_b = np.random.randn(m_b * p_b, target_rank)
        W_b = np.random.randn(p_b * n_b, target_rank)
        
        T_block = np.zeros((n_b * m_b, m_b * p_b, p_b * n_b))
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    T_block[i * m_b + j, j * p_b + k, k * n_b + i] = 1.0
        
        for it in range(150):
            T_flat_U = T_block.reshape(n_b * m_b, m_b * p_b * p_b * n_b)
            M_U = np.einsum('bc,dc->bd', V_b, W_b).reshape(m_b * p_b, p_b * n_b, target_rank)
            M_U_flat = M_U.reshape(m_b * p_b * p_b * n_b, target_rank)
            try:
                U_b = np.linalg.lstsq(M_U_flat.T, T_flat_U.T, rcond=None)[0].T
            except: pass
            
            T_flat_V = T_block.transpose(1, 0, 2).reshape(m_b * p_b, n_b * m_b * p_b * n_b)
            M_V = np.einsum('ac,dc->ad', U_b, W_b).reshape(n_b * m_b, p_b * n_b, target_rank)
            M_V_flat = M_V.reshape(n_b * m_b * p_b * n_b, target_rank)
            try:
                V_b = np.linalg.lstsq(M_V_flat.T, T_flat_V.T, rcond=None)[0].T
            except: pass
            
            T_flat_W = T_block.transpose(2, 0, 1).reshape(p_b * n_b, n_b * m_b * m_b * p_b)
            M_W = np.einsum('ac,bc->ab', U_b, V_b).reshape(n_b * m_b, m_b * p_b, target_rank)
            M_W_flat = M_W.reshape(n_b * m_b * m_b * p_b, target_rank)
            try:
                W_b = np.linalg.lstsq(M_W_flat.T, T_flat_W.T, rcond=None)[0].T
            except: pass
        
        U_snap = np.rint(U_b)
        V_snap = np.rint(V_b)
        W_snap = np.rint(W_b)
        
        U_list = [list(map(int, U_snap[i])) for i in range(n_b * m_b)]
        V_list = [list(map(int, V_snap[i])) for i in range(m_b * p_b)]
        W_list = [list(map(int, W_snap[i])) for i in range(p_b * n_b)]
        
        if verify_block(U_list, V_list, W_list, n_b, m_b, p_b):
            return U_list, V_list, W_list
        return None

    # Try 3x2x3 rank 16
    block_decomp = None
    for seed in range(20):
        if time.time() > deadline - 20: break
        res = find_block_decomp(3, 2, 3, 16, seed)
        if res:
            block_decomp = res
            break

    if block_decomp:
        U_blk, V_blk, W_blk = block_decomp
        U_full = [[0] * 64 for _ in range(n * m)]
        V_full = [[0] * 64 for _ in range(m * p)]
        W_full = [[0] * 64 for _ in range(p * n)]
        
        col_offset = 0
        for m_blk in [[0, 1], [2, 3]]:
            for p_blk in [[0, 1, 2], [3, 4, 5]]:
                for r in range(16):
                    c = col_offset + r
                    for i in range(3):
                        for j in range(2):
                            U_full[i * m + m_blk[j]][c] = U_blk[i * 2 + j][r]
                    for j in range(2):
                        for k in range(3):
                            V_full[m_blk[j] * p + p_blk[k]][c] = V_blk[j * 3 + k][r]
                    for k in range(3):
                        for i in range(3):
                            W_full[p_blk[k] * n + i][c] = W_blk[k * 3 + i][r]
                col_offset += 16
        
        if verify(U_full, V_full, W_full, n, m, p):
            if 64 < best_rank:
                best = {"U": U_full, "V": V_full, "W": W_full}
                best_rank = 64

    # Fallback: 66 rank block decomposition
    if best_rank > 66:
        U_66 = [[0] * 66 for _ in range(n * m)]
        V_66 = [[0] * 66 for _ in range(m * p)]
        W_66 = [[0] * 66 for _ in range(p * n)]
        
        # Strassen 2x2x2
        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        col = 0
        n_blocks = [[0, 1], [2]]
        m_blocks = [[0, 1], [2, 3]]
        p_blocks = [[0, 1], [2, 3], [4, 5]]
        
        for n_blk in n_blocks:
            for m_blk in m_blocks:
                for p_blk in p_blocks:
                    if len(n_blk) == 2:
                        for u_vals, v_vals, w_vals in strassen_terms:
                            c = col
                            for idx, val in enumerate(u_vals):
                                if val: U_66[n_blk[0]*m + m_blk[0] + (idx%2)*1 + (idx//2)*0][c] += val # Simplified mapping
                            # Correct mapping:
                            # u_vals[0] -> A[0,0] -> U[n_blk[0]*m + m_blk[0]]
                            # u_vals[1] -> A[0,1] -> U[n_blk[0]*m + m_blk[1]]
                            # u_vals[2] -> A[1,0] -> U[n_blk[1]*m + m_blk[0]]
                            # u_vals[3] -> A[1,1] -> U[n_blk[1]*m + m_blk[1]]
                            U_66[n_blk[0]*m + m_blk[0]][c] = u_vals[0]
                            U_66[n_blk[0]*m + m_blk[1]][c] = u_vals[1]
                            U_66[n_blk[1]*m + m_blk[0]][c] = u_vals[2]
                            U_66[n_blk[1]*m + m_blk[1]][c] = u_vals[3]
                            
                            V_66[m_blk[0]*p + p_blk[0]][c] = v_vals[0]
                            V_66[m_blk[0]*p + p_blk[1]][c] = v_vals[1]
                            V_66[m_blk[1]*p + p_blk[0]][c] = v_vals[2]
                            V_66[m_blk[1]*p + p_blk[1]][c] = v_vals[3]
                            
                            W_66[p_blk[0]*n + n_blk[0]][c] = w_vals[0]
                            W_66[p_blk[0]*n + n_blk[1]][c] = w_vals[1]
                            W_66[p_blk[1]*n + n_blk[0]][c] = w_vals[2]
                            W_66[p_blk[1]*n + n_blk[1]][c] = w_vals[3]
                            col += 1
                    else:
                        # 1x2x2 rank 4
                        for k_idx in range(2):
                            for j_idx in range(2):
                                c = col
                                U_66[n_blk[0]*m + k_idx][c] = 1 # A[0, k_idx]
                                V_66[m_blk[0]*p + k_idx + j_idx*2][c] = 0 # Wait, V index is j*p+k?
                                # V row: m_blk[j] * p + p_blk[k]
                                # Here m_blk has size 2.
                                # V_66[ (m_blk[0] + k_idx)*p + (p_blk[0] + j_idx) ] ? No.
                                # V is m x p. Block is m_blk x p_blk.
                                # V row index in block: k_idx * 2 + j_idx?
                                # V[k, j] -> row k * p + j? No, V is m x p. Row is m_idx * p + p_idx.
                                # Here m_idx is k_idx (0 or 1). p_idx is j_idx (0 or 1).
                                V_66[(m_blk[0] + k_idx)*p + (p_blk[0] + j_idx)][c] = 1
                                W_66[(p_blk[0] + j_idx)*n + n_blk[0]][c] = 1
                                col += 1
        
        if verify(U_66, V_66, W_66, n, m, p):
            if 66 < best_rank:
                best = {"U": U_66, "V": V_66, "W": W_66}
                best_rank = 66

    return best
# EVOLVE-BLOCK-END
