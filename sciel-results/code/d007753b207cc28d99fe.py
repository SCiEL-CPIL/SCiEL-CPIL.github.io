import time
import numpy as np

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    The decomposition is derived from the Kronecker product of two Strassen decompositions for 2x2 matrices.
    
    Interface:
    U: (16, 49) matrix where row i*m+j corresponds to A_{i,j}
    V: (16, 49) matrix where row j*p+k corresponds to B_{j,k}
    W: (16, 49) matrix where row k*n+i corresponds to C_{i,k} (Note: k is row of B/col of C, i is row of A/row of C)
       Wait, the convention is W[k*n + i, r].
       In standard matrix multiplication C = AB, C_{i,k} = sum_j A_{i,j} B_{j,k}.
       The tensor T is defined such that <U, A> <V, B> <W, C> = <T, A \otimes B \otimes C>.
       The component T_{i,j,k} corresponds to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
       The vector W_r is the "weight" of the output C in the rank-r term.
       If the rank-r term computes a scalar M_r, and C = sum_r M_r * W_r, then W_r is a vector over the space of C.
       The index of W in the returned dict is specified as W[k*n + i]. This implies the index corresponds to the position (i, k) in the matrix C, flattened with k as the major index?
       Let's check: "W[k*n+i, r]".
       If n=4, and we have C_{i,k}, the linear index is k*4 + i.
       This means the vector W is ordered by column k, then row i.
       So W[col, row] in matrix terms? Or is it just a specific flattening order?
       Yes, index = k * 4 + i.
       
       The Strassen decomposition for 2x2 gives us 7 terms.
       For 4x4, we have 49 terms.
       Term (r, s) corresponds to u_{rs} \otimes v_{rs} \otimes w_{rs}.
       
       Let's hardcode the 2x2 Strassen vectors.
       A indices (row-major): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
       B indices (row-major): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
       C indices: We need to map to the specific W index convention.
       C_{i,k} -> index k*4 + i (for 2x2, k*2+i).
       
       Strassen 2x2 Multipliers:
       M1 = (A00 + A11)(B00 + B11)
       M2 = (A10 + A11)(B00)
       M3 = (A00)(B01 - B11)
       M4 = (A11)(B10 - B00)
       M5 = (A00 + A01)(B11)
       M6 = (A10 - A00)(B00 + B01)
       M7 = (A01 - A11)(B10 + B11)
       
       Output C:
       C00 = M1 + M4 - M5 + M7
       C01 = M3 + M5
       C10 = M2 + M4
       C11 = M1 - M2 + M3 + M6
       
       Vectors U (coeffs for A):
       U1: [1, 0, 0, 1]
       U2: [0, 0, 1, 1]
       U3: [1, 0, 0, 0]
       U4: [0, 0, 0, 1]
       U5: [1, 1, 0, 0]
       U6: [-1, 0, 1, 0]
       U7: [0, 1, 0, -1]
       
       Vectors V (coeffs for B):
       V1: [1, 0, 0, 1]
       V2: [1, 0, 0, 0]
       V3: [0, 1, 0, -1]
       V4: [0, 0, 1, -1]
       V5: [0, 0, 0, 1]
       V6: [1, 1, 0, 0]
       V7: [0, 0, 1, 1]
       
       Vectors W (coeffs for C, ordered by C00, C01, C10, C11 initially, then mapped to W-index):
       M1: C00 += 1, C11 += 1  -> W1_raw: [1, 0, 0, 1]
       M2: C10 += 1, C11 -= 1  -> W2_raw: [0, 0, 1, -1]
       M3: C01 += 1, C11 += 1  -> W3_raw: [0, 1, 0, 1]
       M4: C00 += 1, C10 += 1  -> W4_raw: [1, 0, 1, 0]
       M5: C00 -= 1, C01 += 1  -> W5_raw: [-1, 1, 0, 0]
       M6: C11 += 1             -> W6_raw: [0, 0, 0, 1]
       M7: C00 += 1             -> W7_raw: [1, 0, 0, 0]
       
       The 4x4 decomposition is the Kronecker product of these.
       U_4x4[idx, r*7+s] = U_2x2[i, r] * U_2x2[j, s] where idx = i*2+j
       V_4x4[idx, r*7+s] = V_2x2[i, r] * V_2x2[j, s] where idx = i*2+j
       
       For W, we need to be careful with the index mapping.
       The 4x4 matrix C has indices I (row 0..3) and K (col 0..3).
       The requested W index is K*4 + I.
       
       The Kronecker product of W vectors:
       W_4x4_vec is a vector of length 16.
       The elements of the 4x4 matrix C are formed by the outer product of the 2x2 block outputs?
       Actually, the tensor T_4x4 = T_2x2 \otimes T_2x2.
       The index of the tensor T_4x4 is (I, J, K) where I is row of A, J is col of A/row of B, K is col of B.
       This produces C_{I, K}.
       
       Let I = 2*i + i'
       Let K = 2*k + k'
       Where i, i' in {0,1} are block row and local row.
       k, k' in {0,1} are block col and local col.
       
       The tensor entry T_4x4[I, J, K] corresponds to the product of:
       T_2x2[i, j, k] * T_2x2[i', j', k']
       
       The vector W_4x4 for term (r,s) has components W_4x4[I, K] (flattened to index K*4+I).
       W_4x4[I, K] = W_2x2[i, k] * W_2x2[i', k']
       where W_2x2 is the vector of coefficients for the 2x2 output matrix.
       
       So, for each term (r,s):
       For each i' in 0..1, i in 0..1, k' in 0..1, k in 0..1:
       I = 2*i + i'
       K = 2*k + k'
       w_val = W_2x2[r][i, k] * W_2x2[s][i', k']  (using raw indices i,k for 2x2)
       
       Note: W_2x2[r] is a vector of length 4, indexed by (row, col) of the 2x2 matrix.
       Let's define W_2x2[r] as a 2x2 array for easier access.
       """
    
    # 1. Define 2x2 Strassen components
    # U: 7x4
    U2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V: 7x4
    V2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # W: 7x4 (Raw matrix indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1))
    # M1: C00=1, C11=1
    W2_raw = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # 2. Construct 4x4 U and V using Kronecker products
    # U4 is 16x49
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    
    # 3. Construct 4x4 W
    # W4 is 16x49
    # Index mapping: W[idx, col] where idx = K*4 + I
    # I = 2*i + i', K = 2*k + k'
    # val = W2_raw[r][i*2+k] * W2_raw[s][i'*2+k']
    W4 = np.zeros((16, 49), dtype=np.int64)
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            
            # Fill U4 and V4
            for i in range(2):
                for j in range(2):
                    idx = i * 2 + j
                    U4[idx, col] = U2[r][i*2+j] * U2[s][i*2+j] 
                    # Wait, the Kronecker product of vectors u_r and u_s.
                    # If u_r is a vector of length 4 (indices 0,1,2,3 corresponding to 00,01,10,11),
                    # then (u_r \otimes u_s) is a vector of length 16.
                    # Index I (0..15) corresponds to (i, i') where I = 2*i + i'.
                    # The value is u_r[i*2 + j']? No.
                    # Let's stick to the definition:
                    # T_4x4[I, J, K] = T_2x2[i, j, k] T_2x2[i', j', k']
                    # where I=2i+i', J=2j+j', K=2k+k'.
                    # The vector U for term (r,s) has component at I equal to U2[r][i*2+j_block? No.
                    # U2[r] is a vector over the 2x2 matrix space.
                    # Let's say U2[r] is a 2x2 matrix M_r.
                    # Then the 4x4 matrix for term r,s is M_r \otimes M_s.
                    # The entries of M_r \otimes M_s at (I, J) are (M_r)_{i,j} (M_s)_{i',j'}.
                    # This matches the tensor structure.
                    
                    # So for U4:
                    # U4[I, col] = (M_r)_{i,j} * (M_s)_{i',j'} where I = 2i + i'.
                    # My previous loop `U4[idx, col] = U2[r][i*2+j] * U2[s][i*2+j]` was wrong because it used the same local indices for both block and local.
                    # It should be:
                    # U4[I, col] = U2[r][i*2 + j] * U2[s][i'*2 + j'] ? No.
                    # The index in U2 is for the 2x2 matrix.
                    # Let's define U2[r] as a 2x2 array.
            
            # Let's restart the filling logic with 2x2 arrays for clarity.
            Ur = np.array(U2[r]).reshape(2, 2)
            Vs = np.array(V2[s]).reshape(2, 2) # Just in case, though we need V_r and V_s
            Vr = np.array(V2[r]).reshape(2, 2)
            V2_s = np.array(V2[s]).reshape(2, 2)
            
            # U4
            U_r_arr = np.array(U2[r]).reshape(2, 2)
            U_s_arr = np.array(U2[s]).reshape(2, 2)
            # Kronecker product
            U_kron = np.kron(U_r_arr, U_s_arr)
            # Flatten U_kron to fill U4 column col
            U4[:, col] = U_kron.flatten()
            
            # V4
            V_r_arr = np.array(V2[r]).reshape(2, 2)
            V_s_arr = np.array(V2[s]).reshape(2, 2)
            V_kron = np.kron(V_r_arr, V_s_arr)
            V4[:, col] = V_kron.flatten()
            
            # W4
            # W2_raw[r] corresponds to coefficients for C_2x2.
            # Let's treat W2_raw[r] as a 2x2 matrix of coefficients.
            W_r_arr = np.array(W2_raw[r]).reshape(2, 2)
            W_s_arr = np.array(W2_raw[s]).reshape(2, 2)
            
            # The 4x4 matrix C is formed by C_4x4 = C_r \otimes C_s ?
            # The tensor decomposition says T_4x4 = sum T_2x2(r) \otimes T_2x2(s).
            # The "W" part of the tensor is the vector of coefficients for the output C.
            # If the term is M_{rs}, then C_4x4 += M_{rs} * (W_r \otimes W_s).
            # So the coefficient matrix for C is W_r \otimes W_s.
            W_kron = np.kron(W_r_arr, W_s_arr)
            
            # Now we need to map the entries of W_kron (row I, col K) to the linear index K*4 + I.
            for i in range(4):
                for k in range(4):
                    val = W_kron[i, k]
                    W4[k * 4 + i, col] = val

    # Convert to lists
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Quick validation
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Random test
    np.random.seed(0)
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)
    C_expected = A @ B
    
    # Compute C_decomposed
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Decomposed: C_{i,k} = sum_r (sum_j A_{i,j} U_{i,j,r}) (sum_k B_{j,k} V_{j,k,r}) ... wait.
    # The tensor is T = sum_r u_r \otimes v_r \otimes w_r.
    # <A, U_r> <B, V_r> is the scalar M_r.
    # Then C = sum_r M_r W_r.
    
    # M_r = sum_{i,j} A_{i,j} U_{i,j,r}
    # U is 16x49. Index i*4+j.
    # V is 16x49. Index j*4+k.
    # W is 16x49. Index k*4+i.
    
    # Let's compute M
    A_flat = A.flatten() # row major: i*4+j
    B_flat = B.flatten() # row major: j*4+k
    
    M = A_flat @ U # Shape (49,)
    M = M * (B_flat @ V) # Element-wise product of the two sums? No.
    # M_r = (A_flat @ U[:, r]) * (B_flat @ V[:, r])
    
    AU = A_flat @ U # (49,)
    BV = B_flat @ V # (49,)
    M = AU * BV     # (49,)
    
    # C_flat_expected = sum_r M_r * W[:, r]
    # But W is stored as W[k*4+i, r].
    # So C_flat_decomposed = M @ W
    C_flat_calc = M @ W
    
    # C_flat_calc is indexed by k*4 + i.
    # We need to reshape it to (4,4) C[i,k].
    C_calc = C_flat_calc.reshape(4, 4, order='F') # Fortran order: first index varies fastest.
    # k*4 + i -> if we reshape in Fortran order, index 0 is (0,0), 1 is (1,0)...
    # Let's check:
    # Linear index L = k*4 + i.
    # In Fortran order (column-major), index L corresponds to row L % 4, col L // 4.
    # So row = i, col = k.
    # So C_calc[i, k] is the value.
    
    if np.allclose(C_calc, C_expected, atol=1e-6):
        print("Validation Passed")
    else:
        print("Validation Failed")
        print(C_calc)
        print(C_expected)

# EVOLVE-BLOCK-START
def construct():
    return construct()
# EVOLVE-BLOCK-END
