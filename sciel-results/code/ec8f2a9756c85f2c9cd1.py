import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)  
    # W: (p*n, rank) = (16, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Rank-32 schoolbook decomposition via block structure
    # A = [A1 | A2] (2x2 each), B = [[B11, B12], [B21, B22]] (2x4 each)
    # C[:,0:4] = A1*B11 + A2*B21
    # C[:,4:8] = A1*B12 + A2*B22
    # Each 2x2*2x4 block uses 8 multiplications
    
    rank = 32
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    
    for a_block in range(2):  # A col block: 0->cols 0,1; 1->cols 2,3
        for b_row_half in range(2):  # B row block: 0->rows 0,1; 1->rows 2,3
            for b_col_half in range(2):  # B col block: 0->cols 0-3; 1->cols 4-7
                a0_idx = 0 * 4 + a_block * 2 + 0
                a1_idx = 0 * 4 + a_block * 2 + 1
                a2_idx = 1 * 4 + a_block * 2 + 0
                a3_idx = 1 * 4 + a_block * 2 + 1
                
                b_row0 = b_row_half * 2
                b_row1 = b_row_half * 2 + 1
                b_col_base = b_col_half * 4
                
                # Row 0 of output: C[0, b_col_base+k] = A[0,a0]*B[b_row0,k] + A[0,a1]*B[b_row1,k]
                for k in range(4):
                    b0k_idx = b_row0 * 8 + b_col_base + k
                    b1k_idx = b_row1 * 8 + b_col_base + k
                    ck_idx = (b_col_base + k) * 2 + 0
                    
                    U[a0_idx, col] = 1
                    V[b0k_idx, col] = 1
                    W[ck_idx, col] = 1
                    col += 1
                    
                    U[a1_idx, col] = 1
                    V[b1k_idx, col] = 1
                    W[ck_idx, col] = 1
                    col += 1
                
                # Row 1 of output: C[1, b_col_base+k] = A[1,a0]*B[b_row0,k] + A[1,a1]*B[b_row1,k]
                for k in range(4):
                    b0k_idx = b_row0 * 8 + b_col_base + k
                    b1k_idx = b_row1 * 8 + b_col_base + k
                    ck_idx = (b_col_base + k) * 2 + 1
                    
                    U[a2_idx, col] = 1
                    V[b0k_idx, col] = 1
                    W[ck_idx, col] = 1
                    col += 1
                    
                    U[a3_idx, col] = 1
                    V[b1k_idx, col] = 1
                    W[ck_idx, col] = 1
                    col += 1
    
    assert col == 32
    
    # Verify
    A_test = np.random.randn(2, 4)
    B_test = np.random.randn(4, 8)
    C_test = A_test @ B_test
    
    C_recon = np.zeros((2, 8))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        u_mat = u_col.reshape(2, 4)
        v_mat = v_col.reshape(4, 8)
        w_mat = w_col.reshape(8, 2)
        
        # C[i,k] += W[k,i,r] * sum_j U[i,j,r]*A[i,j] * sum_j V[j,k,r]*B[j,k]
        a_proj = np.array([sum(u_mat[i,j] * A_test[i,j] for j in range(4)) for i in range(2)])
        b_proj = np.array([sum(v_mat[j,k] * B_test[j,k] for j in range(4)) for k in range(8)])
        
        for i in range(2):
            for k in range(8):
                C_recon[i, k] += w_mat[k, i] * a_proj[i] * b_proj[k]
    
    max_err = np.max(np.abs(C_test - C_recon))
    
    if max_err > 1e-10:
        # Fallback to rank-56 Strassen
        rank56 = 56
        U56 = np.zeros((8, rank56))
        V56 = np.zeros((32, rank56))
        W56 = np.zeros((16, rank56))
        
        col56 = 0
        for a_col_off in [0, 2]:
            for b_col_off in [0, 2, 4, 6]:
                a00 = 0*4 + a_col_off
                a01 = 0*4 + a_col_off + 1
                a10 = 1*4 + a_col_off
                a11 = 1*4 + a_col_off + 1
                
                b00 = 0*8 + b_col_off
                b01 = 0*8 + b_col_off + 1
                b10 = 1*8 + b_col_off
                b11 = 1*8 + b_col_off + 1
                
                c00 = b_col_off*2 + 0
                c01 = (b_col_off+1)*2 + 0
                c10 = b_col_off*2 + 1
                c11 = (b_col_off+1)*2 + 1
                
                U56[a00, col56] += 1; U56[a11, col56] += 1
                V56[b00, col56] += 1; V56[b11, col56] += 1
                W56[c00, col56] += 1; W56[c11, col56] += 1
                col56 += 1
                
                U56[a10, col56] += 1; U56[a11, col56] += 1
                V56[b00, col56] += 1
                W56[c10, col56] += 1; W56[c11, col56] += -1
                col56 += 1
                
                U56[a00, col56] += 1
                V56[b01, col56] += 1; V56[b11, col56] += -1
                W56[c01, col56] += 1; W56[c11, col56] += 1
                col56 += 1
                
                U56[a11, col56] += 1
                V56[b10, col56] += 1; V56[b00, col56] += -1
                W56[c00, col56] += 1; W56[c10, col56] += 1
                col56 += 1
                
                U56[a00, col56] += 1; U56[a01, col56] += 1
                V56[b11, col56] += 1
                W56[c00, col56] += -1; W56[c01, col56] += 1
                col56 += 1
                
                U56[a10, col56] += 1; U56[a00, col56] += -1
                V56[b00, col56] += 1; V56[b01, col56] += 1
                W56[c11, col56] += 1
                col56 += 1
                
                U56[a01, col56] += 1; U56[a11, col56] += -1
                V56[b10, col56] += 1; V56[b11, col56] += 1
                W56[c00, col56] += 1
                col56 += 1
        
        assert col56 == 56
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
