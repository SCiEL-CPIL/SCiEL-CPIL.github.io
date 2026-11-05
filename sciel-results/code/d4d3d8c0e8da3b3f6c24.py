import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for the 120s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Laderman's rank-15 decomposition for 3x3 matrix multiplication
    # Tensor condition: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k
    # This is the trace convention: T_{ijk} = 1 (since trace(ABC) = sum A_ij B_jk C_ki)
    
    def get_laderman_3x3():
        """
        Returns U, V, W matrices of shape (9, 15) for Laderman's 3x3 decomposition.
        U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        """
        U = np.zeros((9, 15))
        V = np.zeros((9, 15))
        W = np.zeros((9, 15))
        
        # Laderman's 15 terms
        # Each term is (u_matrix, v_matrix, w_matrix) where each is 3x3
        # u[i,j] * v[j,k] * w[k,i] summed over r should give 1 for all i,j,k
        
        # I'll use a verified Laderman decomposition
        # Reference: Laderman (1976), also in Strassen's book
        
        # Term 0:
        U[0*3+0, 0] = 1; V[0*3+0, 0] = 1; W[0*3+0, 0] = 1
        # Term 1:
        U[0*3+1, 1] = 1; V[1*3+1, 1] = 1; W[1*3+0, 1] = 1
        # Term 2:
        U[1*3+1, 2] = 1; V[1*3+2, 2] = 1; W[2*3+1, 2] = 1
        # Term 3:
        U[1*3+2, 3] = 1; V[2*3+2, 3] = 1; W[2*3+1, 3] = 1
        # Term 4:
        U[2*3+2, 4] = 1; V[2*3+0, 4] = 1; W[0*3+2, 4] = 1
        
        # The above is just schoolbook for the diagonal. I need the full Laderman.
        # Let me use a different approach: I'll construct it from known formulas.
        
        # Actually, let me use a well-known explicit Laderman decomposition.
        # The 15 terms can be written as:
        
        # Reset
        U = np.zeros((9, 15))
        V = np.zeros((9, 15))
        W = np.zeros((9, 15))
        
        # I'll use the decomposition from the literature where:
        # M1 = A11*(B11+B22), M2 = A22*(B11+B22), etc.
        # But for the tensor form, I need explicit coefficients.
        
        # Let me try a different strategy: use the fact that Laderman's algorithm
        # can be expressed in terms of 15 bilinear forms.
        
        # Actually, I'll use a verified set of coefficients from a reference implementation.
        # Here's one known Laderman decomposition:
        
        # u_r, v_r, w_r for r=0..14
        # Using the convention: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
        
        # Term 0: u=[[1,0,0],[0,0,0],[0,0,0]], v=[[1,0,0],[0,0,0],[0,0,0]], w=[[1,0,0],[0,0,0],[0,0,0]]
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        
        # Term 1: u=[[0,1,0],[0,0,0],[0,0,0]], v=[[0,0,0],[1,0,0],[0,0,0]], w=[[0,0,0],[0,1,0],[0,0,0]]
        U[1,1]=1; V[4,1]=1; W[4,1]=1
        
        # This is getting too complex to do manually. Let me use a different approach.
        # I'll use the schoolbook rank-27 for 3x3 and Strassen rank-7 for 2x2 blocks.
        # This gives rank 7*27 = 189, which is what the current program does.
        
        # But I want to try Laderman. Let me use a programmatic approach to verify.
        # For now, let me just return the schoolbook 3x3 and use Strassen 2x2.
        
        # Actually, let me implement Laderman properly using known formulas.
        # From Strassen's "Gaussian Elimination and Matrix Multiplication":
        
        # The 15 multiplications in Laderman's algorithm are:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # But this is for 2x2 blocks, not 3x3. For 3x3, Laderman's is different.
        
        # Let me just use schoolbook 3x3 (rank 27) with Strassen 2x2 (rank 7) = rank 189.
        # This is safe and known to work.
        
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = i*9 + j*3 + k
                    U[i*3+j, col] = 1.0
                    V[j*3+k, col] = 1.0
                    W[k*3+i, col] = 1.0
        return U, V, W

    U3, V3, W3 = get_laderman_3x3()
    r3 = U3.shape[1]  # Should be 27 for schoolbook
    
    # If we could use Laderman (rank 15), total would be 7*15=105
    # With schoolbook 3x3, total is 7*27=189
    
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    # Strassen Formulas for 2x2 block multiplication
    # Blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Block (bi, bj) means row block bi, col block bj
    # U index: (row_block*3 + i) * 6 + (col_block*3 + j)
    # V index: (row_block*3 + j) * 6 + (col_block*3 + k)  
    # W index: (col_block*3 + k) * 6 + (row_block*3 + i)
    # Wait, W[k*n+i] means W index = k*6 + i where k is col of C, i is row of C
    # So W index for block (ci, cj) at position (i,j) within block: (cj*3+j)*6 + (ci*3+i)
    
    def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
        """
        a_blocks, b_blocks, c_blocks: list of (block_idx, coeff)
        block_idx 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        """
        for r in range(r3):
            u_vec = U3[:, r]
            v_vec = V3[:, r]
            w_vec = W3[:, r]
            
            # Update U6: U[i*6+j, col] for i,j in 0..5
            for bi, coeff in a_blocks:
                row_block = bi // 2
                col_block = bi % 2
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_vec[i*3+j]
                        if val != 0:
                            global_i = row_block * 3 + i
                            global_j = col_block * 3 + j
                            U6[global_i * 6 + global_j, start_col + r] += val
            
            # Update V6: V[j*6+k, col] for j,k in 0..5
            for bi, coeff in b_blocks:
                row_block = bi // 2
                col_block = bi % 2
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_vec[j*3+k]
                        if val != 0:
                            global_j = row_block * 3 + j
                            global_k = col_block * 3 + k
                            V6[global_j * 6 + global_k, start_col + r] += val
            
            # Update W6: W[k*6+i, col] for k,i in 0..5
            # W index = k*6 + i where k is col of C, i is row of C
            for bi, coeff in c_blocks:
                row_block = bi // 2  # This is the row of C block
                col_block = bi % 2   # This is the col of C block
                for i in range(3):   # i within C block (row)
                    for k in range(3): # k within C block (col)
                        val = coeff * w_vec[k*3+i]
                        if val != 0:
                            global_k = col_block * 3 + k
                            global_i = row_block * 3 + i
                            W6[global_k * 6 + global_i, start_col + r] += val

    # P1 = (A11 + A22) * (B11 + B22)
    # C11 += P1, C22 += P1
    add_strassen_term(0, 
                      a_blocks=[(0, 1), (3, 1)], 
                      b_blocks=[(0, 1), (3, 1)], 
                      c_blocks=[(0, 1), (3, 1)])
    
    # P2 = (A21 + A22) * B11
    # C12 += P2, C22 -= P2
    add_strassen_term(r3, 
                      a_blocks=[(2, 1), (3, 1)], 
                      b_blocks=[(0, 1)], 
                      c_blocks=[(1, 1), (3, -1)])
    
    # P3 = A11 * (B12 - B22)
    # C11 += P3
    add_strassen_term(2*r3, 
                      a_blocks=[(0, 1)], 
                      b_blocks=[(1, 1), (3, -1)], 
                      c_blocks=[(0, 1)])
    
    # P4 = A22 * (B21 - B11)
    # C11 -= P4, C22 += P4
    add_strassen_term(3*r3, 
                      a_blocks=[(3, 1)], 
                      b_blocks=[(2, 1), (0, -1)], 
                      c_blocks=[(0, -1), (3, 1)])
    
    # P5 = (A11 + A12) * B22
    # C21 += P5
    add_strassen_term(4*r3, 
                      a_blocks=[(0, 1), (1, 1)], 
                      b_blocks=[(3, 1)], 
                      c_blocks=[(2, 1)])
    
    # P6 = (A21 - A11) * (B11 + B12)
    # C21 += P6, C22 += P6
    add_strassen_term(5*r3, 
                      a_blocks=[(2, 1), (0, -1)], 
                      b_blocks=[(0, 1), (1, 1)], 
                      c_blocks=[(2, 1), (3, 1)])
    
    # P7 = (A12 - A22) * (B21 + B22)
    # C11 -= P7, C12 += P7
    add_strassen_term(6*r3, 
                      a_blocks=[(1, 1), (3, -1)], 
                      b_blocks=[(2, 1), (3, 1)], 
                      c_blocks=[(0, -1), (1, 1)])

    # Verify the decomposition
    if time.time() < deadline:
        try:
            U3g = U6.reshape(6, 6, -1)
            V3g = V6.reshape(6, 6, -1)
            W3g = W6.reshape(6, 6, -1)
            
            # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for all i,j,k
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            target = np.ones((6, 6, 6))
            if np.allclose(reconstructed, target, atol=1e-9):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
            else:
                # Debug: print where it fails
                diff = reconstructed - target
                print(f"Max diff: {np.max(np.abs(diff))}")
        except Exception as e:
            print(f"Verification error: {e}")

    # Fallback to schoolbook if Strassen construction failed or time expired
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                idx = i*6 + j
                jk = j*6 + k
                ki = k*6 + i
                col = i*36 + j*6 + k
                U_f[idx, col] = 1.0
                V_f[jk, col] = 1.0
                W_f[ki, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
