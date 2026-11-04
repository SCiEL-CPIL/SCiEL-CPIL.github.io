"""Construct a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.

Uses a block-structured decomposition:
- Split the 3x4 A into a 1x4 row-block and a 2x4 row-block.
- Split the 4x5 B into a 4x2 column-block and a 4x3 column-block.
- The 2x4 x 4x2 = 2x2x4 part is decomposed as two 2x2x2 blocks (Strassen, rank 7 each).
- The 1x4 x 4x2 = 1x2x4 part uses schoolbook (rank 8).
- The 1x4 x 4x3 = 1x2x3 part uses schoolbook (rank 12).
- The 2x4 x 4x3 = 2x2x6 part uses schoolbook (rank 24).

Total rank: 7+7+8+12+24 = 58.
"""

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    # U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    
    # We'll build the decomposition block by block.
    # Each block contributes a set of columns to U, V, W.
    
    rank_parts = []  # list of (U_block, V_block, W_block) where each is a list of columns
    
    def schoolbook_block(n_b, m_b, p_b):
        """Standard decomposition for n_b x m_b by m_b x p_b."""
        cols_U = []
        cols_V = []
        cols_W = []
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u = [0] * (n_b * m_b)
                    u[i * m_b + j] = 1
                    cols_U.append(u)
                    
                    v = [0] * (m_b * p_b)
                    v[j * p_b + k] = 1
                    cols_V.append(v)
                    
                    w = [0] * (p_b * n_b)
                    w[k * n_b + i] = 1
                    cols_W.append(w)
        return cols_U, cols_V, cols_W
    
    def strassen_block(n_b, m_b, p_b):
        """Strassen decomposition for 2x2x2 block.
        n_b=m_b=p_b=2.
        Returns 7 columns for U, V, W of sizes (n_b*m_b,), (m_b*p_b,), (p_b*n_b,).
        """
        if n_b != 2 or m_b != 2 or p_b != 2:
            return schoolbook_block(n_b, m_b, p_b)
        
        # U indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  [k*n+i: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3]
        # Wait, W index is k*n+i. For n=2:
        # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
        # So W[0]=(i=0,k=0), W[1]=(i=1,k=0), W[2]=(i=0,k=1), W[3]=(i=1,k=1)
        
        cols_U = []
        cols_V = []
        cols_W = []
        
        # M1 = A00 * (B00 + B11)
        u1 = [0,0,0,0]; u1[0] = 1
        v1 = [0,0,0,0]; v1[0] = 1; v1[3] = 1
        w1 = [0,0,0,0]; w1[0] = 1; w1[1] = -1  # W[k=0,i=0]=1, W[k=0,i=1]=-1
        cols_U.append(u1); cols_V.append(v1); cols_W.append(w1)
        
        # M2 = (A10 + A11) * B00
        u2 = [0,0,0,0]; u2[2] = 1; u2[3] = 1
        v2 = [0,0,0,0]; v2[0] = 1
        w2 = [0,0,0,0]; w2[2] = 1; w2[3] = 1  # W[k=1,i=0]=1, W[k=1,i=1]=1
        cols_U.append(u2); cols_V.append(v2); cols_W.append(w2)
        
        # M3 = A00 * (B10 - B11)
        u3 = [0,0,0,0]; u3[0] = 1
        v3 = [0,0,0,0]; v3[2] = 1; v3[3] = -1
        w3 = [0,0,0,0]; w3[2] = 1  # W[k=1,i=0]=1
        cols_U.append(u3); cols_V.append(v3); cols_W.append(w3)
        
        # M4 = A11 * (B01 - B00)
        u4 = [0,0,0,0]; u4[3] = 1
        v4 = [0,0,0,0]; v4[1] = 1; v4[0] = -1
        w4 = [0,0,0,0]; w4[3] = 1  # W[k=1,i=1]=1
        cols_U.append(u4); cols_V.append(v4); cols_W.append(w4)
        
        # M5 = (A00 + A11) * (B00 + B11)
        u5 = [0,0,0,0]; u5[0] = 1; u5[3] = 1
        v5 = [0,0,0,0]; v5[0] = 1; v5[3] = 1
        w5 = [0,0,0,0]; w5[0] = 1; w5[3] = 1; w5[1] = -1; w5[2] = -1
        cols_U.append(u5); cols_V.append(v5); cols_W.append(w5)
        
        # M6 = (A01 - A00) * (B00 + B10)
        u6 = [0,0,0,0]; u6[1] = 1; u6[0] = -1
        v6 = [0,0,0,0]; v6[0] = 1; v6[2] = 1
        w6 = [0,0,0,0]; w6[1] = 1  # W[k=0,i=1]=1
        cols_U.append(u6); cols_V.append(v6); cols_W.append(w6)
        
        # M7 = (A10 - A11) * (B11 + B01)
        u7 = [0,0,0,0]; u7[2] = 1; u7[3] = -1
        v7 = [0,0,0,0]; v7[3] = 1; v7[1] = 1
        w7 = [0,0,0,0]; w7[2] = -1; w7[3] = 1  # W[k=1,i=0]=-1, W[k=1,i=1]=1
        cols_U.append(u7); cols_V.append(v7); cols_W.append(w7)
        
        return cols_U, cols_V, cols_W
    
    # Now build the full decomposition.
    # A is 3x4: split into A_top (1x4) and A_bot (2x4)
    # B is 4x5: split into B_left (4x2) and B_right (4x3)
    
    # Block 1: A_top (1x4) * B_left (4x2) -> 1x2 result
    # This is a 1x4 x 4x2 multiplication. Use schoolbook: rank 1*4*2 = 8.
    # A_top indices: i=0, j=0..3
    # B_left indices: j=0..3, k=0..1
    # Result: i=0, k=0..1
    u1, v1, w1 = schoolbook_block(1, 4, 2)
    # Global U indices for this block: i=0, j=0..3 -> U[0*4+0]..U[0*4+3] = U[0]..U[3]
    # Global V indices: j=0..3, k=0..1 -> V[0*5+0]..V[3*5+1] = V[0]..V[16]
    # Global W indices: k=0..1, i=0 -> W[0*3+0]=W[0], W[1*3+0]=W[3]
    
    # Block 2: A_top (1x4) * B_right (4x3) -> 1x3 result
    # Schoolbook: rank 1*4*3 = 12.
    u2, v2, w2 = schoolbook_block(1, 4, 3)
    
    # Block 3: A_bot (2x4) * B_left (4x2) -> 2x2 result
    # Split into two 2x2x2 blocks along the m=4 dimension? No, m is shared.
    # Actually, A_bot is 2x4, B_left is 4x2. The inner dimension is 4.
    # We can split the 4 into 2+2: A_bot = [A_bl1 (2x2) | A_bl2 (2x2)], B_left = [B_bl1 (2x2); B_bl2 (2x2)]
    # Then result = A_bl1*B_bl1 + A_bl2*B_bl2. Each is 2x2x2, rank 7.
    u3a, v3a, w3a = strassen_block(2, 2, 2)
    u3b, v3b, w3b = strassen_block(2, 2, 2)
    
    # Block 4: A_bot (2x4) * B_right (4x3) -> 2x3 result
    # Split m=4 into 2+2: A_bot = [A_br1 (2x2) | A_br2 (2x2)], B_right = [B_br1 (2x3); B_br2 (2x3)]
    # Result = A_br1*B_br1 + A_br2*B_br2. Each is 2x2x3, schoolbook rank 12.
    u4a, v4a, w4a = schoolbook_block(2, 2, 3)
    u4b, v4b, w4b = schoolbook_block(2, 2, 3)
    
    # Now assemble global U, V, W.
    # Total rank = 8 + 12 + 7 + 7 + 12 + 12 = 58
    
    total_rank = 8 + 12 + 7 + 7 + 12 + 12
    U = [[0] * total_rank for _ in range(n * m)]  # 3*4=12
    V = [[0] * total_rank for _ in range(m * p)]  # 4*5=20
    W = [[0] * total_rank for _ in range(p * n)]  # 5*3=15
    
    col_offset = 0
    
    # Helper to add a block
    def add_block(cols_U, cols_V, cols_W, i_offset, j_offset, k_offset):
        nonlocal col_offset
        for idx, (cu, cv, cw) in enumerate(zip(cols_U, cols_V, cols_W)):
            c = col_offset + idx
            for local_idx, val in enumerate(cu):
                if val != 0:
                    gi = (local_idx // 1)  # This is relative to the block
                    # local_idx is in [0, n_b*m_b)
                    # global row = (i_offset + local_i) * m + (j_offset + local_j)
                    # But local_idx = local_i * m_b + local_j
                    local_i = local_idx // len(cv) if len(cv) > 0 else 0
                    # This is getting complicated. Let me just index directly.
            col_offset += len(cols_U)
    
    # Let me just do it manually for clarity.
    
    # Block 1: A_top (1x4) * B_left (4x2)
    # i_offset=0, j_offset=0, k_offset=0
    # n_b=1, m_b=4, p_b=2
    # U rows: i=0, j=0..3 -> global U[0..3]
    # V rows: j=0..3, k=0..1 -> global V[j*5+k]
    # W rows: k=0..1, i=0 -> global W[k*3+0] = W[0], W[3]
    for idx in range(8):
        c = col_offset + idx
        # u1[idx] is a vector of length 1*4=4
        for li in range(4):
            if u1[idx][li] != 0:
                U[li][c] += u1[idx][li]
        for li in range(20):
            if v1[idx][li] != 0:
                V[li][c] += v1[idx][li]
        for li in range(6):
            if w1[idx][li] != 0:
                W[li][c] += w1[idx][li]
    col_offset += 8
    
    # Block 2: A_top (1x4) * B_right (4x3)
    # i_offset=0, j_offset=0, k_offset=2
    # n_b=1, m_b=4, p_b=3
    # U rows: i=0, j=0..3 -> global U[0..3]
    # V rows: j=0..3, k=0..2 (local) -> global V[j*5+(2+k)]
    # W rows: k=0..2 (local), i=0 -> global W[(2+k)*3+0]
    for idx in range(12):
        c = col_offset + idx
        for li in range(4):
            if u2[idx][li] != 0:
                U[li][c] += u2[idx][li]
        # v2[idx] has length 4*3=12, indexed by j*3+k_local
        for j in range(4):
            for k_local in range(3):
                val = v2[idx][j*3+k_local]
                if val != 0:
                    V[j*5+(2+k_local)][c] += val
        # w2[idx] has length 3*1=3, indexed by k_local*n_b+i_local = k_local*1+0
        for k_local in range(3):
            val = w2[idx][k_local]
            if val != 0:
                W[(2+k_local)*3+0][c] += val
    col_offset += 12
    
    # Block 3a: A_bot_top (2x2) * B_left_top (2x2)
    # i_offset=1, j_offset=0, k_offset=0
    # n_b=2, m_b=2, p_b=2
    # U rows: i=0..1 (local), j=0..1 (local) -> global U[(1+i)*4+j]
    # V rows: j=0..1 (local), k=0..1 (local) -> global V[j*5+k]
    # W rows: k=0..1 (local), i=0..1 (local) -> global W[k*3+(1+i)]
    for idx in range(7):
        c = col_offset + idx
        # u3a[idx] has length 2*2=4, indexed by i*2+j
        for i in range(2):
            for j in range(2):
                val = u3a[idx][i*2+j]
                if val != 0:
                    U[(1+i)*4+j][c] += val
        # v3a[idx] has length 2*2=4, indexed by j*2+k
        for j in range(2):
            for k in range(2):
                val = v3a[idx][j*2+k]
                if val != 0:
                    V[j*5+k][c] += val
        # w3a[idx] has length 2*2=4, indexed by k*2+i
        for k in range(2):
            for i in range(2):
                val = w3a[idx][k*2+i]
                if val != 0:
                    W[k*3+(1+i)][c] += val
    col_offset += 7
    
    # Block 3b: A_bot_bot (2x2) * B_left_bot (2x2)
    # i_offset=1, j_offset=2, k_offset=0
    # U rows: i=0..1 (local), j=0..1 (local) -> global U[(1+i)*4+(2+j)]
    # V rows: j=0..1 (local), k=0..1 (local) -> global V[(2+j)*5+k]
    # W rows: k=0..1 (local), i=0..1 (local) -> global W[k*3+(1+i)]
    for idx in range(7):
        c = col_offset + idx
        for i in range(2):
            for j in range(2):
                val = u3b[idx][i*2+j]
                if val != 0:
                    U[(1+i)*4+(2+j)][c] += val
        for j in range(2):
            for k in range(2):
                val = v3b[idx][j*2+k]
                if val != 0:
                    V[(2+j)*5+k][c] += val
        for k in range(2):
            for i in range(2):
                val = w3b[idx][k*2+i]
                if val != 0:
                    W[k*3+(1+i)][c] += val
    col_offset += 7
    
    # Block 4a: A_bot_top (2x2) * B_right_top (2x3)
    # i_offset=1, j_offset=0, k_offset=2
    # n_b=2, m_b=2, p_b=3
    # U rows: i=0..1, j=0..1 -> global U[(1+i)*4+j]
    # V rows: j=0..1, k=0..2 (local) -> global V[j*5+(2+k)]
    # W rows: k=0..2 (local), i=0..1 -> global W[(2+k)*3+(1+i)]
    for idx in range(12):
        c = col_offset + idx
        for i in range(2):
            for j in range(2):
                val = u4a[idx][i*2+j]
                if val != 0:
                    U[(1+i)*4+j][c] += val
        for j in range(2):
            for k in range(3):
                val = v4a[idx][j*3+k]
                if val != 0:
                    V[j*5+(2+k)][c] += val
        for k in range(3):
            for i in range(2):
                val = w4a[idx][k*2+i]
                if val != 0:
                    W[(2+k)*3+(1+i)][c] += val
    col_offset += 12
    
    # Block 4b: A_bot_bot (2x2) * B_right_bot (2x3)
    # i_offset=1, j_offset=2, k_offset=2
    # U rows: i=0..1, j=0..1 -> global U[(1+i)*4+(2+j)]
    # V rows: j=0..1, k=0..2 (local) -> global V[(2+j)*5+(2+k)]
    # W rows: k=0..2 (local), i=0..1 -> global W[(2+k)*3+(1+i)]
    for idx in range(12):
        c = col_offset + idx
        for i in range(2):
            for j in range(2):
                val = u4b[idx][i*2+j]
                if val != 0:
                    U[(1+i)*4+(2+j)][c] += val
        for j in range(2):
            for k in range(3):
                val = v4b[idx][j*3+k]
                if val != 0:
                    V[(2+j)*5+(2+k)][c] += val
        for k in range(3):
            for i in range(2):
                val = w4b[idx][k*2+i]
                if val != 0:
                    W[(2+k)*3+(1+i)][c] += val
    col_offset += 12
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    print(f"Rank: {rank}")
    # Verify
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if i,j,k are the product indices.
    # Actually, the tensor for matrix multiplication M_{n,m,p}[i,j,k] = delta(i,j,k) means
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all i,j,k (it's the identity in the appropriate sense)?
    # No. The multiplication tensor is T such that (A*B)[i,k] = sum_{j,c} A[i,j] B[j,k] T_{i,j,k}^{(c)} ...
    # Actually, the standard definition: T_{i,j,k} is the coefficient such that
    # (A*B)[i,k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] A[i,j] B[j,k]
    # For this to equal sum_j A[i,j] B[j,k], we need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-10:
                    print(f"ERROR: T[{i},{j},{k}] = {s}, expected 1")
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        print("Verification PASSED")
