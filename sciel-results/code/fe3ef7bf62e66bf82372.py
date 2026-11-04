import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy: Kronecker Product of Strassen(2x2) and Schoolbook(3x3)
    # Rank = 7 * 27 = 189
    
    # 1. Construct Schoolbook Tensor for 3x3 matrices
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i] for tensor T[i,j,k]
    n3 = 3
    r3 = n3 * n3 * n3  # 27
    U3 = np.zeros((n3*n3, r3))
    V3 = np.zeros((n3*n3, r3))
    W3 = np.zeros((n3*n3, r3))
    
    for i in range(n3):
        for j in range(n3):
            for k in range(n3):
                col = i*n3*n3 + j*n3 + k
                U3[i*n3+j, col] = 1.0
                V3[j*n3+k, col] = 1.0
                W3[k*n3+i, col] = 1.0
                
    # 2. Construct Strassen Tensor for 2x2 matrices
    # We define the 7 products P1..P7.
    # U represents the linear combination of A blocks.
    # V represents the linear combination of B blocks.
    # W represents the linear combination of P results to form C blocks.
    
    n2 = 2
    r2 = 7
    U2 = np.zeros((n2*n2, r2))
    V2 = np.zeros((n2*n2, r2))
    W2 = np.zeros((n2*n2, r2))
    
    # Block indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U index: row*2 + col
    # V index: row*2 + col
    # W index: row*2 + col
    
    # P1 = (A00 + A11) * (B00 + B11)
    # A00 is U[0], A11 is U[3]
    # B00 is V[0], B11 is V[3]
    # C00 += P1, C11 += P1
    # W[0] += P1, W[3] += P1
    U2[0, 0] += 1; U2[3, 0] += 1
    V2[0, 0] += 1; V2[3, 0] += 1
    W2[0, 0] += 1; W2[3, 0] += 1
    
    # P2 = (A10 + A11) * B00
    # A10 is U[2], A11 is U[3]
    # B00 is V[0]
    # C10 += P2, C11 -= P2
    U2[2, 1] += 1; U2[3, 1] += 1
    V2[0, 1] += 1
    W2[2, 1] += 1; W2[3, 1] -= 1
    
    # P3 = A00 * (B01 - B11)
    # A00 is U[0]
    # B01 is V[1], B11 is V[3]
    # C01 += P3, C11 += P3
    U2[0, 2] += 1
    V2[1, 2] += 1; V2[3, 2] -= 1
    W2[1, 2] += 1; W2[3, 2] += 1
    
    # P4 = A11 * (B10 - B00)
    # A11 is U[3]
    # B10 is V[2], B00 is V[0]
    # C00 += P4, C10 += P4
    U2[3, 3] += 1
    V2[2, 3] += 1; V2[0, 3] -= 1
    W2[0, 3] += 1; W2[2, 3] += 1
    
    # P5 = (A00 + A01) * B11
    # A00 is U[0], A01 is U[1]
    # B11 is V[3]
    # C01 += P5, C00 -= P5
    U2[0, 4] += 1; U2[1, 4] += 1
    V2[3, 4] += 1
    W2[1, 4] += 1; W2[0, 4] -= 1
    
    # P6 = (A10 - A00) * (B00 + B01)
    # A10 is U[2], A00 is U[0]
    # B00 is V[0], B01 is V[1]
    # C11 += P6
    U2[2, 5] += 1; U2[0, 5] -= 1
    V2[0, 5] += 1; V2[1, 5] += 1
    W2[3, 5] += 1
    
    # P7 = (A01 - A11) * (B10 + B11)
    # A01 is U[1], A11 is U[3]
    # B10 is V[2], B11 is V[3]
    # C00 += P7
    U2[1, 6] += 1; U2[3, 6] -= 1
    V2[2, 6] += 1; V2[3, 6] += 1
    W2[0, 6] += 1

    # 3. Combine via Kronecker Product
    # The total tensor is T_total = T_Strassen (outer) x T_Schoolbook (inner)
    # Dimensions:
    # U_total: (n2*n2 * n3*n3, r2*r3) = (4 * 9, 7 * 27) = (36, 189)
    # V_total: (n2*n2 * n3*n3, r2*r3) = (4 * 9, 7 * 27) = (36, 189)
    # W_total: (n2*n2 * n3*n3, r2*r3) = (4 * 9, 7 * 27) = (36, 189)
    
    # The Kronecker product of matrices U2 and U3 gives the structure where
    # each entry U2[i,j] scales the entire matrix U3.
    # We need to ensure the indices map correctly to the 6x6 layout.
    # 6x6 layout: Row = R2*3 + R3, Col = C2*3 + C3
    # Index = (R2*3 + R3)*6 + (C2*3 + C3)
    # = R2*18 + R3*6 + C2*3 + C3
    # 
    # np.kron(U2, U3) creates a matrix of size (4*9, 7*27).
    # Row index in kron: i2 * 9 + i3. 
    # i2 = R2*2 + C2. i3 = R3*3 + C3.
    # This matches the required ordering if we interpret the row index of the 36x189 matrix 
    # as (BlockRow*3 + LocalRow)*6 + (BlockCol*3 + LocalCol).
    # Let's verify:
    # i2 = 2*R2 + C2.
    # Row index = (2*R2 + C2)*9 + (3*R3 + C3)
    # = 18*R2 + 9*C2 + 3*R3 + C3.
    # 
    # Required global index for A[R2*3+R3, C2*3+C3] is (R2*3+R3)*6 + (C2*3+C3)
    # = 18*R2 + 6*R3 + 6*C2 + 3*C3.
    # 
    # The kron index (18*R2 + 9*C2 + 3*R3 + C3) does NOT match the standard row-major 
    # flattening of the 6x6 matrix (18*R2 + 6*R3 + 6*C2 + 3*C3).
    # 
    # We need to permute the rows of the Kronecker product to match the 6x6 layout.
    # Alternatively, we can construct U, V, W by iterating over the 7 Strassen products 
    # and placing the 3x3 schoolbook blocks in the correct positions.
    # Given the time constraints and the need for robustness, let's stick to the explicit 
    # placement method which is safer for index mapping, but use vectorized operations 
    # inside the loop to be distinct from the previous loop-heavy implementation.
    
    total_rank = r2 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    for s in range(r2):
        # For each Strassen product s, we have coefficients in U2[:, s], V2[:, s], W2[:, s]
        # We need to add the corresponding 3x3 schoolbook tensors scaled by these coefficients.
        
        u_coeff = U2[:, s]
        v_coeff = V2[:, s]
        w_coeff = W2[:, s]
        
        # U2 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Map to 6x6 row/col block offsets
        # Row block r_b, Col block c_b -> Global Row Offset = r_b * 3, Global Col Offset = c_b * 3
        
        for r_b in range(2):
            for c_b in range(2):
                idx2 = r_b * 2 + c_b
                u_val = u_coeff[idx2]
                v_val = v_coeff[idx2]
                w_val = w_coeff[idx2]
                
                if u_val == 0 and v_val == 0 and w_val == 0:
                    continue
                
                # Determine the slice of the 36x189 matrix corresponding to this block
                # Row indices: (r_b * 3) to (r_b * 3 + 2)
                # Col indices: (c_b * 3) to (c_b * 3 + 2)
                # Global Row Index in 36-vector: (r_b * 3 + i) * 6 + (c_b * 3 + j)
                
                # We will add u_val * U3 to the U6 block, etc.
                # U3 is 9x27. We need to map its 9 rows to the specific 9 rows in U6.
                
                rows_u = []
                cols_u = []
                for i in range(3):
                    for j in range(3):
                        gr = r_b * 3 + i
                        gc = c_b * 3 + j
                        rows_u.append(gr * 6 + gc)
                
                # Similarly for V and W. Note W indices are (k, i) -> (col, row) of C.
                # In 6x6 C, C[k, i].
                # Global W index: k * 6 + i.
                # k is in block r_b? No, k is row of C, which corresponds to row of B?
                # Wait, C = A B.
                # A is n x m. B is m x p. C is p x n.
                # Here n=m=p=6.
                # C[k, i] = sum_j A[i, j] B[j, k].
                # W tensor index is (k, i).
                # So for block (r_b, c_b) of C:
                # r_b is row block of C (which is k block).
                # c_b is col block of C (which is i block).
                
                rows_v = []
                for j in range(3): # j is row of B (col of A)
                    for k in range(3): # k is col of B (row of C)
                        # B block (r_b, c_b) means B row block r_b, col block c_b.
                        # So j is in block r_b, k is in block c_b.
                        gr = r_b * 3 + j
                        gc = c_b * 3 + k
                        rows_v.append(gr * 6 + gc)
                        
                rows_w = []
                for k in range(3): # k is row of C (block r_b)
                    for i in range(3): # i is col of C (block c_b)
                        gr = r_b * 3 + k
                        gc = c_b * 3 + i
                        rows_w.append(gr * 6 + gc)
                        
                # Now update the matrices
                # U3 is 9x27. We need to add u_val * U3 to U6[rows_u, s*27:(s+1)*27]
                # Since U3 is sparse (only 1s), we can just add u_val to the non-zero spots.
                # But vectorized addition is faster and cleaner.
                
                U6[np.ix_(rows_u, range(s*r3, (s+1)*r3))] += u_val * U3
                V6[np.ix_(rows_v, range(s*r3, (s+1)*r3))] += v_val * V3
                W6[np.ix_(rows_w, range(s*r3, (s+1)*r3))] += w_val * W3

    # Verify
    if time.time() < deadline:
        try:
            U3g = U6.reshape(6, 6, -1)
            V3g = V6.reshape(6, 6, -1)
            W3g = W6.reshape(6, 6, -1)
            
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            target = np.ones((6, 6, 6))
            if np.allclose(reconstructed, target, atol=1e-9):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        except Exception as e:
            pass

    # Fallback to schoolbook
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
