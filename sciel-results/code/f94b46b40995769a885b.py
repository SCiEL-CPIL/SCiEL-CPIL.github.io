import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get Strassen rank-7 decomposition for 2x2 * 2x2
    # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices for 2x2 matrix in prompt convention:
        # U index i*m+j -> i*2+j
        # V index j*p+k -> j*2+k
        # W index k*n+i -> k*2+i
        
        # A entries:
        # a = A[0,0] -> U[0]
        # b = A[0,1] -> U[1]
        # c = A[1,0] -> U[2]
        # d = A[1,1] -> U[3]
        
        # B entries:
        # e = B[0,0] -> V[0]
        # f = B[0,1] -> V[1]
        # g = B[1,0] -> V[2]
        # h = B[1,1] -> V[3]
        
        # C entries (output):
        # C[0,0] -> W[0] (k=0, i=0)
        # C[0,1] -> W[1] (k=0, i=1) -- Wait, W index is k*n+i.
        # In 2x2 case, n=2.
        # C_{ki} = sum A_{ij} B_{jk}.
        # C_{00} (k=0, i=0) -> W[0*2+0]=W[0]
        # C_{01} (k=1, i=0) -> W[1*2+0]=W[2] ? 
        # Let's re-verify the W index mapping.
        # Prompt: W[k*n+i].
        # Loop: i in range(n), j in range(m), k in range(p).
        # For 2x2: n=2, m=2, p=2.
        # i=0,1. j=0,1. k=0,1.
        # W index = k*2 + i.
        # If k=0, i=0 -> W[0].
        # If k=0, i=1 -> W[1].
        # If k=1, i=0 -> W[2].
        # If k=1, i=1 -> W[3].
        
        # Standard C = AB.
        # C_{ki} = sum_j A_{ij} B_{jk}.
        # C_{00} = A_{00}B_{00} + A_{01}B_{10}.
        # C_{01} = A_{00}B_{01} + A_{01}B_{11}.
        # C_{10} = A_{10}B_{00} + A_{11}B_{10}.
        # C_{11} = A_{10}B_{01} + A_{11}B_{11}.
        
        # So W[0] corresponds to C_{00}.
        # W[1] corresponds to C_{01} (k=0, i=1).
        # W[2] corresponds to C_{10} (k=1, i=0).
        # W[3] corresponds to C_{11} (k=1, i=1).
        
        # Strassen M's:
        # M1 = (A00+A11)(B00+B11)
        # M2 = A11(B00-B01)
        # M3 = (A00-A11)B11
        # M4 = A00(B11-B00)
        # M5 = (A01+A11)B00
        # M6 = (A00-A01)(B01+B11)
        # M7 = (A10+A00)(B00-B10)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M5 + M6
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M6
        
        # We need to decompose each M_k into rank-1 terms?
        # No, Strassen IS the rank-7 decomposition.
        # Each M_k is a product of a linear combo of A and a linear combo of B.
        # So each M_k contributes 1 to the rank.
        
        # Let's define the coefficients for U, V, W for each of the 7 terms.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1 -> U[0]=1, U[3]=1
        # V: B00=1, B11=1 -> V[0]=1, V[3]=1
        # W: C00 += 1, C11 += 1 -> W[0]=1, W[3]=1
        
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[:, 0] = [1, 0, 0, 1]
        
        # Term 2: M2 = A11(B00-B01)
        # U: A11=1 -> U[3]=1
        # V: B00=1, B01=-1 -> V[0]=1, V[1]=-1
        # W: C10 += 1, C11 += -1 -> W[2]=1, W[3]=-1
        
        U[:, 1] = [0, 0, 0, 1]
        V[:, 1] = [1, -1, 0, 0]
        W[:, 1] = [0, 0, 1, -1]
        
        # Term 3: M3 = (A00-A11)B11
        # U: A00=1, A11=-1 -> U[0]=1, U[3]=-1
        # V: B11=1 -> V[3]=1
        # W: C10 += 1, C11 += 1 -> W[2]=1, W[3]=1
        
        U[:, 2] = [1, 0, 0, -1]
        V[:, 2] = [0, 0, 0, 1]
        W[:, 2] = [0, 0, 1, 1]
        
        # Term 4: M4 = A00(B11-B00)
        # U: A00=1 -> U[0]=1
        # V: B11=1, B00=-1 -> V[3]=1, V[0]=-1
        # W: C00 += 1, C11 += 1 -> W[0]=1, W[3]=1
        
        U[:, 3] = [1, 0, 0, 0]
        V[:, 3] = [-1, 0, 0, 1]
        W[:, 3] = [1, 0, 0, 1]
        
        # Term 5: M5 = (A01+A11)B00
        # U: A01=1, A11=1 -> U[1]=1, U[3]=1
        # V: B00=1 -> V[0]=1
        # W: C00 += -1, C01 += 1 -> W[0]=-1, W[1]=1
        
        U[:, 4] = [0, 1, 0, 1]
        V[:, 4] = [1, 0, 0, 0]
        W[:, 4] = [-1, 1, 0, 0]
        
        # Term 6: M6 = (A00-A01)(B01+B11)
        # U: A00=1, A01=-1 -> U[0]=1, U[1]=-1
        # V: B01=1, B11=1 -> V[1]=1, V[3]=1
        # W: C01 += 1, C11 += 1 -> W[1]=1, W[3]=1
        
        U[:, 5] = [1, -1, 0, 0]
        V[:, 5] = [0, 1, 0, 1]
        W[:, 5] = [0, 1, 0, 1]
        
        # Term 7: M7 = (A10+A00)(B00-B10)
        # U: A10=1, A00=1 -> U[2]=1, U[0]=1
        # V: B00=1, B10=-1 -> V[0]=1, V[2]=-1
        # W: C00 += 1, C10 += 1 -> W[0]=1, W[2]=1
        
        U[:, 6] = [1, 0, 1, 0]
        V[:, 6] = [1, 0, -1, 0]
        W[:, 6] = [1, 0, 1, 0]
        
        return U, V, W

    # Get the base 2x2 decomposition
    U2, V2, W2 = strassen_2x2()
    
    # Now build 4x4 * 4x4 decomposition using blocks
    # A is 4x4, split into 2x2 blocks of 2x2 matrices.
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # Indices in U (size 16):
    # Block (br, bc) starts at row br*4, col bc*4?
    # U index is i*4+j.
    # A11: i in 0,1; j in 0,1 -> indices 0,1,4,5
    # A12: i in 0,1; j in 2,3 -> indices 2,3,6,7
    # A21: i in 2,3; j in 0,1 -> indices 8,9,12,13
    # A22: i in 2,3; j in 2,3 -> indices 10,11,14,15
    
    # Similarly for V (B) and W (C).
    
    # Strassen 7 terms for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = A22(B11-B12)
    # M3 = (A11-A22)B22
    # M4 = A11(B22-B11)
    # M5 = (A12+A22)B11
    # M6 = (A11-A12)(B12+B22)
    # M7 = (A21+A11)(B11-B21)
    
    # Each Mk is a 2x2 * 2x2 multiplication.
    # We use the U2, V2, W2 decomposition for each.
    
    rank_4x4 = 49
    U4 = np.zeros((16, rank_4x4))
    V4 = np.zeros((16, rank_4x4))
    W4 = np.zeros((16, rank_4x4))
    
    # Define block index offsets for U, V, W
    # U blocks:
    # A11: [0:2, 0:2] -> rows 0,1,4,5
    # A12: [0:2, 2:4] -> rows 2,3,6,7
    # A21: [2:4, 0:2] -> rows 8,9,12,13
    # A22: [2:4, 2:4] -> rows 10,11,14,15
    
    # V blocks (B):
    # B11: [0:2, 0:2] -> cols 0,1,4,5 (in V matrix, index j*4+k)
    # B12: [0:2, 2:4] -> cols 2,3,6,7
    # B21: [2:4, 0:2] -> cols 8,9,12,13
    # B22: [2:4, 2:4] -> cols 10,11,14,15
    
    # W blocks (C):
    # C11: [0:2, 0:2] -> W index k*4+i.
    # C is 4x4.
    # C11 (top-left): k in 0,1; i in 0,1.
    # Indices: 0,1,4,5.
    # C12 (top-right): k in 0,1; i in 2,3.
    # Indices: 2,3,6,7.
    # C21 (bottom-left): k in 2,3; i in 0,1.
    # Indices: 8,9,12,13.
    # C22 (bottom-right): k in 2,3; i in 2,3.
    # Indices: 10,11,14,15.
    
    # Helper to add a 2x2 block decomposition to the 4x4 matrices
    def add_block(U_target, V_target, W_target, col_offset, 
                  u_blocks, v_blocks, w_blocks, 
                  u_coeffs, v_coeffs, w_coeffs):
        # u_blocks: dict mapping block name to list of indices
        # u_coeffs: dict mapping block name to coefficient
        # We iterate over the 7 terms of the base Strassen
        for t in range(7):
            col = col_offset + t
            # U contribution
            for b_name, coeff in u_coeffs.items():
                if coeff != 0:
                    idxs = u_blocks[b_name]
                    for idx in idxs:
                        U_target[idx, col] += coeff * U2[idx % 4, t] # Wait, U2 index is local 0..3
            # V contribution
            for b_name, coeff in v_coeffs.items():
                if coeff != 0:
                    idxs = v_blocks[b_name]
                    for idx in idxs:
                        V_target[idx, col] += coeff * V2[idx % 4, t]
            # W contribution
            for b_name, coeff in w_coeffs.items():
                if coeff != 0:
                    idxs = w_blocks[b_name]
                    for idx in idxs:
                        W_target[idx, col] += coeff * W2[idx % 4, t]
                        
    # Define block indices
    u_blocks = {
        '11': [0, 1, 4, 5],
        '12': [2, 3, 6, 7],
        '21': [8, 9, 12, 13],
        '22': [10, 11, 14, 15]
    }
    v_blocks = {
        '11': [0, 1, 4, 5],
        '12': [2, 3, 6, 7],
        '21': [8, 9, 12, 13],
        '22': [10, 11, 14, 15]
    }
    w_blocks = {
        '11': [0, 1, 4, 5],
        '12': [2, 3, 6, 7],
        '21': [8, 9, 12, 13],
        '22': [10, 11, 14, 15]
    }
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    u_c1 = {'11': 1, '22': 1}
    v_c1 = {'11': 1, '22': 1}
    w_c1 = {'11': 1, '22': 1}
    add_block(U4, V4, W4, 0, u_blocks, v_blocks, w_blocks, u_c1, v_c1, w_c1)
    
    # Term 2: M2 = A22(B11-B12)
    u_c2 = {'22': 1}
    v_c2 = {'11': 1, '12': -1}
    w_c2 = {'21': 1, '22': -1} # C21 += M2, C22 += -M2
    add_block(U4, V4, W4, 7, u_blocks, v_blocks, w_blocks, u_c2, v_c2, w_c2)
    
    # Term 3: M3 = (A11-A22)B22
    u_c3 = {'11': 1, '22': -1}
    v_c3 = {'22': 1}
    w_c3 = {'21': 1, '22': 1} # C21 += M3, C22 += M3
    add_block(U4, V4, W4, 14, u_blocks, v_blocks, w_blocks, u_c3, v_c3, w_c3)
    
    # Term 4: M4 = A11(B22-B11)
    u_c4 = {'11': 1}
    v_c4 = {'22': 1, '11': -1}
    w_c4 = {'11': 1, '22': 1} # C11 += M4, C22 += M4
    add_block(U4, V4, W4, 21, u_blocks, v_blocks, w_blocks, u_c4, v_c4, w_c4)
    
    # Term 5: M5 = (A12+A22)B11
    u_c5 = {'12': 1, '22': 1}
    v_c5 = {'11': 1}
    w_c5 = {'11': -1, '12': 1} # C11 += -M5, C12 += M5
    add_block(U4, V4, W4, 28, u_blocks, v_blocks, w_blocks, u_c5, v_c5, w_c5)
    
    # Term 6: M6 = (A11-A12)(B12+B22)
    u_c6 = {'11': 1, '12': -1}
    v_c6 = {'12': 1, '22': 1}
    w_c6 = {'12': 1, '22': 1} # C12 += M6, C22 += M6
    add_block(U4, V4, W4, 35, u_blocks, v_blocks, w_blocks, u_c6, v_c6, w_c6)
    
    # Term 7: M7 = (A21+A11)(B11-B21)
    u_c7 = {'21': 1, '11': 1}
    v_c7 = {'11': 1, '21': -1}
    w_c7 = {'11': 1, '21': 1} # C11 += M7, C21 += M7
    add_block(U4, V4, W4, 42, u_blocks, v_blocks, w_blocks, u_c7, v_c7, w_c7)
    
    # Now we have U4, V4, W4 for 4x4 * 4x4.
    # We need 4x4 * 4x8.
    # Split B into B1 (4x4) and B2 (4x4).
    # U is same for both.
    # V1 uses first 16 rows of V (indices 0..15).
    # V2 uses second 16 rows of V? No, V is 16x49 for one 4x4.
    # For 4x8, V shape is (4*8, rank) = (32, rank).
    # B1 corresponds to V rows 0..15.
    # B2 corresponds to V rows 16..31.
    
    # W1 corresponds to C1 (4x4) -> W rows 0..15.
    # W2 corresponds to C2 (4x4) -> W rows 16..31.
    
    rank_total = 98
    U_final = np.zeros((16, rank_total))
    V_final = np.zeros((32, rank_total))
    W_final = np.zeros((32, rank_total))
    
    # Copy U (same for both)
    U_final[:, 0:49] = U4
    U_final[:, 49:98] = U4
    
    # Copy V1
    V_final[0:16, 0:49] = V4
    # V2 is zero in first part
    V_final[16:32, 0:49] = 0
    
    # Copy W1
    W_final[0:16, 0:49] = W4
    # W2 is zero in first part
    W_final[16:32, 0:49] = 0
    
    # Copy V2
    V_final[0:16, 49:98] = 0
    V_final[16:32, 49:98] = V4
    
    # Copy W2
    W_final[0:16, 49:98] = 0
    W_final[16:32, 49:98] = W4
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
