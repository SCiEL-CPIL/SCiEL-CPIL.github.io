import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = 3, 4, 8

    # Schoolbook rank
    rank_sb = n * m * p  # 96
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, idx] = 1
                V_sb[j * p + k, idx] = 1
                W_sb[k * n + i, idx] = 1
                idx += 1
                
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    def verify(U, V, W, rank, tol=1e-9):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Reshape
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        rng = np.random.default_rng(12345)
        for _ in range(3):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            
            M = U_3d * A[:, :, np.newaxis]
            N = V_3d * B[:, :, np.newaxis]
            S = np.einsum('ijc,jkc->ikc', M, N)
            S_T = np.transpose(S, (1, 0, 2))
            C_dec = np.sum(S_T * W_3d, axis=2)
            
            if np.max(np.abs(C - C_dec)) > tol:
                return False
        return True

    def to_clean_list(mat_np):
        result = []
        for row in mat_np:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                elif abs(v * 2 - round(v * 2)) < 1e-9:
                    new_row.append(round(v * 2) / 2)
                else:
                    new_row.append(v)
            result.append(new_row)
        return result

    # Strategy: Block Decomposition using Strassen for 2x2x2 sub-blocks
    # We can partition the 3x4x8 multiplication into blocks.
    # Let's try to decompose it into 2x2x2 blocks and use Strassen.
    # However, the dimensions don't align perfectly with 2x2x2.
    # 3x4x8 can be seen as:
    # i in 0..2, j in 0..3, k in 0..7
    
    # Let's use a known low-rank decomposition for 2x2x2 (Strassen) and combine.
    # Strassen for 2x2x2:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    # This gives rank 7 for 2x2x2.
    # Schoolbook for 2x2x2 is 8.
    # We have 3x4x8.
    # We can partition A (3x4) into blocks of 2x2? No, 3 is not divisible by 2.
    # But we can partition A into a 2x2 block and a 1x4 block?
    # Or partition B (4x8) into 4x2 blocks?
    
    # Let's try a different approach:
    # Use the fact that 3x4x8 = (2+1)x4x8.
    # We can split the rows of A into 2 and 1.
    # A = [A1; A2] where A1 is 2x4, A2 is 1x4.
    # C = A1*B + A2*B.
    # A1*B is 2x8. We can compute this using Strassen if we split B into 4x2 blocks?
    # B is 4x8. Split into 4x2, 4x2, 4x2, 4x2.
    # A1 (2x4) * B1 (4x2) -> 2x2. Use Strassen? No, Strassen is for 2x2*2x2.
    # Here we have 2x4 * 4x2. This is not a square block multiplication.
    
    # Let's look at the tensor rank bounds.
    # For 2x2x2, rank 7.
    # For 2x2x3, what is the rank?
    # For 3x3x3, rank 23.
    
    # Maybe we can use a known decomposition for 3x4x8?
    # Or just use a heuristic:
    # If we use Strassen for every 2x2x2 sub-problem, we save 1 rank per 8 ranks.
    # How many 2x2x2 blocks can we fit?
    # If we partition 3x4x8 into 2x2x2 blocks, we can fit:
    # i: 0-1 (2), j: 0-1 (2), k: 0-1 (2) -> 1 block
    # i: 0-1 (2), j: 0-1 (2), k: 2-3 (2) -> 1 block
    # i: 0-1 (2), j: 0-1 (2), k: 4-5 (2) -> 1 block
    # i: 0-1 (2), j: 0-1 (2), k: 6-7 (2) -> 1 block
    # i: 0-1 (2), j: 2-3 (2), k: 0-1 (2) -> 1 block
    # i: 0-1 (2), j: 2-3 (2), k: 2-3 (2) -> 1 block
    # i: 0-1 (2), j: 2-3 (2), k: 4-5 (2) -> 1 block
    # i: 0-1 (2), j: 2-3 (2), k: 6-7 (2) -> 1 block
    # i: 2 (1), j: 0-1 (2), k: 0-1 (2) -> 1x2x2 block
    # i: 2 (1), j: 0-1 (2), k: 2-3 (2) -> 1x2x2 block
    # i: 2 (1), j: 0-1 (2), k: 4-5 (2) -> 1x2x2 block
    # i: 2 (1), j: 0-1 (2), k: 6-7 (2) -> 1x2x2 block
    # i: 2 (1), j: 2-3 (2), k: 0-1 (2) -> 1x2x2 block
    # i: 2 (1), j: 2-3 (2), k: 2-3 (2) -> 1x2x2 block
    # i: 2 (1), j: 2-3 (2), k: 4-5 (2) -> 1x2x2 block
    # i: 2 (1), j: 2-3 (2), k: 6-7 (2) -> 1x2x2 block
    
    # Total 16 blocks.
    # 8 blocks of 2x2x2. Rank 7 each. Total 56.
    # 8 blocks of 1x2x2. What is the rank of 1x2x2?
    # C[i,k] = sum_j A[i,j] B[j,k]. i is fixed.
    # This is just a 2x2 matrix multiplication for each i? No, i is fixed, so A is 1x2, B is 2x2, C is 1x2.
    # Rank of 1x2x2 tensor:
    # T[i,j,k] = delta_{i,0} (since i=0 only).
    # It's just the standard 2x2 matrix multiplication tensor restricted to one row.
    # Actually, the tensor is T_{0jk} = 1 for all j,k.
    # This is a rank 4 tensor? No, it's just the identity map for the 4 parameters?
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # This is a bilinear map from (1x2, 2x2) to (1x2).
    # The tensor has 4 non-zero entries: T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1.
    # Wait, the indices are (i,j,k). i=0. j=0,1. k=0,1.
    # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1.
    # This tensor can be decomposed into 4 rank-1 tensors?
    # T = e_0 (in i) (e_0 (in j) e_0 (in k) + e_0 (in j) e_1 (in k) + e_1 (in j) e_0 (in k) + e_1 (in j) e_1 (in k))
    # This is not rank 1.
    # But we can write it as:
    # T = e_0 e_0 e_0 + e_0 e_0 e_1 + e_0 e_1 e_0 + e_0 e_1 e_1
    # This is a sum of 4 rank-1 tensors.
    # Can it be done in less?
    # The matrix of the tensor (flattening i and j, keeping k) is:
    # [1 1]
    # [1 1]
    # Rank is 1. So the tensor rank is at most 1? No, tensor rank is different.
    # For 1x2x2, the tensor is just the 2x2 matrix B multiplied by the vector A.
    # It's equivalent to 2x2 matrix multiplication but with A being a row vector.
    # The rank of the 1x2x2 tensor is 4?
    # Let's check.
    # If we use schoolbook, rank is 4.
    # Can we do better?
    # Strassen doesn't apply directly to 1x2x2.
    # So we assume rank 4 for 1x2x2.
    
    # Total rank = 8 * 7 + 8 * 4 = 56 + 32 = 88.
    # This is better than 96.
    
    # Let's implement this block decomposition.
    
    def build_block_decomposition():
        U_blocks = []
        V_blocks = []
        W_blocks = []
        
        # Strassen 2x2x2 coefficients
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        # c11 = m1+m4-m5+m7
        # c12 = m3+m5
        # c21 = m2+m4
        # c22 = m1-m2+m3+m6
        
        # U, V, W for Strassen 2x2x2
        # U is 2x2x7, V is 2x2x7, W is 2x2x7
        U_strassen = np.zeros((2, 2, 7))
        V_strassen = np.zeros((2, 2, 7))
        W_strassen = np.zeros((2, 2, 7))
        
        # m1
        U_strassen[0,0,0] = 1; U_strassen[1,1,0] = 1
        V_strassen[0,0,0] = 1; V_strassen[1,1,0] = 1
        W_strassen[0,0,0] = 1; W_strassen[1,1,0] = 1
        W_strassen[0,1,0] = -1; W_strassen[1,0,0] = 1 # c11, c22, c12, c21 contributions
        
        # Let's be precise.
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # W[c, i, j] where c is the output index, i, j are input indices?
        # In our convention, W is (p, n, rank) -> (k, i, c).
        # So W[k, i, c] is the coefficient for output C[i, k] from product m_c.
        # For 2x2x2, i=0,1, k=0,1.
        
        # m1: c11+=1, c22+=1
        W_strassen[0,0,0] = 1
        W_strassen[1,1,0] = 1
        
        # m2: c21+=1
        W_strassen[0,1,1] = 1
        
        # m3: c12+=1, c22+=1
        W_strassen[1,0,2] = 1
        W_strassen[1,1,2] = 1
        
        # m4: c11+=1, c21+=1
        W_strassen[0,0,3] = 1
        W_strassen[0,1,3] = 1
        
        # m5: c12+=1, c11-=1
        W_strassen[1,0,4] = 1
        W_strassen[0,0,4] = -1
        
        # m6: c22+=1
        W_strassen[1,1,5] = 1
        
        # m7: c11+=1
        W_strassen[0,0,6] = 1
        
        # U and V
        # m1: (a11+a22)(b11+b22)
        U_strassen[0,0,0] = 1; U_strassen[1,1,0] = 1
        V_strassen[0,0,0] = 1; V_strassen[1,1,0] = 1
        
        # m2: (a21+a22)b11
        U_strassen[1,0,1] = 1; U_strassen[1,1,1] = 1
        V_strassen[0,0,1] = 1
        
        # m3: a11(b12-b22)
        U_strassen[0,0,2] = 1
        V_strassen[0,1,2] = 1; V_strassen[1,1,2] = -1
        
        # m4: a22(b21-b11)
        U_strassen[1,1,3] = 1
        V_strassen[1,0,3] = 1; V_strassen[0,0,3] = -1
        
        # m5: (a11+a12)b22
        U_strassen[0,0,4] = 1; U_strassen[0,1,4] = 1
        V_strassen[1,1,4] = 1
        
        # m6: (a21-a11)(b11+b12)
        U_strassen[1,0,5] = 1; U_strassen[0,0,5] = -1
        V_strassen[0,0,5] = 1; V_strassen[0,1,5] = 1
        
        # m7: (a12-a22)(b21+b22)
        U_strassen[0,1,6] = 1; U_strassen[1,1,6] = -1
        V_strassen[1,0,6] = 1; V_strassen[1,1,6] = 1
        
        # 1x2x2 schoolbook
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        U_1x2x2 = np.zeros((1, 2, 4))
        V_1x2x2 = np.zeros((2, 2, 4))
        W_1x2x2 = np.zeros((2, 1, 4))
        
        U_1x2x2[0,0,0] = 1; V_1x2x2[0,0,0] = 1; W_1x2x2[0,0,0] = 1
        U_1x2x2[0,0,1] = 1; V_1x2x2[0,1,1] = 1; W_1x2x2[1,0,1] = 1
        U_1x2x2[0,1,2] = 1; V_1x2x2[1,0,2] = 1; W_1x2x2[0,0,2] = 1
        U_1x2x2[0,1,3] = 1; V_1x2x2[1,1,3] = 1; W_1x2x2[1,0,3] = 1
        
        # Now assemble the full 3x4x8 decomposition
        # We will create large U, V, W matrices and fill them block by block.
        total_rank = 88
        U_full = np.zeros((3*4, total_rank))
        V_full = np.zeros((4*8, total_rank))
        W_full = np.zeros((8*3, total_rank))
        
        rank_offset = 0
        
        # 8 blocks of 2x2x2
        # i in [0,1], j in [0,1] or [2,3], k in [0,1], [2,3], [4,5], [6,7]
        for i_block in range(2): # 0, 1
            i_start = i_block * 2
            for j_block in range(2): # 0, 1
                j_start = j_block * 2
                for k_block in range(4): # 0, 1, 2, 3
                    k_start = k_block * 2
                    
                    # Fill U
                    for di in range(2):
                        for dj in range(2):
                            for c in range(7):
                                U_full[(i_start+di)*4 + (j_start+dj), rank_offset+c] = U_strassen[di, dj, c]
                    
                    # Fill V
                    for dj in range(2):
                        for dk in range(2):
                            for c in range(7):
                                V_full[(j_start+dj)*8 + (k_start+dk), rank_offset+c] = V_strassen[dj, dk, c]
                    
                    # Fill W
                    for dk in range(2):
                        for di in range(2):
                            for c in range(7):
                                W_full[(k_start+dk)*3 + (i_start+di), rank_offset+c] = W_strassen[dk, di, c]
                    
                    rank_offset += 7
        
        # 8 blocks of 1x2x2
        # i in [2], j in [0,1] or [2,3], k in [0,1], [2,3], [4,5], [6,7]
        for j_block in range(2): # 0, 1
            j_start = j_block * 2
            for k_block in range(4): # 0, 1, 2, 3
                k_start = k_block * 2
                
                # Fill U
                for dj in range(2):
                    for c in range(4):
                        U_full[2*4 + (j_start+dj), rank_offset+c] = U_1x2x2[0, dj, c]
                
                # Fill V
                for dj in range(2):
                    for dk in range(2):
                        for c in range(4):
                            V_full[(j_start+dj)*8 + (k_start+dk), rank_offset+c] = V_1x2x2[dj, dk, c]
                
                # Fill W
                for dk in range(2):
                    for c in range(4):
                        W_full[(k_start+dk)*3 + 2, rank_offset+c] = W_1x2x2[dk, 0, c]
                
                rank_offset += 4
        
        # Verify
        if verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), total_rank):
            return {"U": to_clean_list(U_full), "V": to_clean_list(V_full), "W": to_clean_list(W_full)}
        
        return None

    res = build_block_decomposition()
    if res:
        best = res
        best_rank = 88

    # Final cleaning
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                if isinstance(v, (int, np.integer)):
                    new_row.append(int(v))
                elif isinstance(v, float):
                    if abs(v - round(v)) < 1e-9:
                        new_row.append(int(round(v)))
                    elif abs(v * 2 - round(v * 2)) < 1e-9:
                        new_row.append(round(v * 2) / 2)
                    else:
                        new_row.append(v)
                else:
                    new_row.append(v)
            result.append(new_row)
        return result

    U_final = clean_matrix(best["U"])
    V_final = clean_matrix(best["V"])
    W_final = clean_matrix(best["W"])
    
    return {"U": U_final, "V": V_final, "W": W_final}
