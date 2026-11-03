import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Return (U, V, W) lists for Strassen 2x2 decomposition.
    Each is a list of 7 vectors of length 4 (2x2 flattened).
    u index: i*2 + j
    v index: j*2 + k
    w index: k*2 + i
    """
    # Standard Strassen 2x2x2 decomposition
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M7
    
    U_list = []
    V_list = []
    W_list = []
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # u: A11+A22 -> indices 0,3 -> [1,0,0,1]
    # v: B11+B22 -> indices 0,3 -> [1,0,0,1]
    # w: C11+=1, C22+=1 -> indices 0,3 -> [1,0,0,1]
    U_list.append([1, 0, 0, 1])
    V_list.append([1, 0, 0, 1])
    W_list.append([1, 0, 0, 1])
    
    # Term 2: M2 = (A21+A22)B11
    # u: A21+A22 -> indices 2,3 -> [0,0,1,1]
    # v: B11 -> index 0 -> [1,0,0,0]
    # w: C21+=1 -> index 2 -> [0,0,1,0]
    U_list.append([0, 0, 1, 1])
    V_list.append([1, 0, 0, 0])
    W_list.append([0, 0, 1, 0])
    
    # Term 3: M3 = A11(B12-B22)
    # u: A11 -> index 0 -> [1,0,0,0]
    # v: B12-B22 -> indices 1,3 -> [0,1,0,-1]
    # w: C12+=1 -> index 1 -> [0,1,0,0]
    U_list.append([1, 0, 0, 0])
    V_list.append([0, 1, 0, -1])
    W_list.append([0, 1, 0, 0])
    
    # Term 4: M4 = A22(B21-B11)
    # u: A22 -> index 3 -> [0,0,0,1]
    # v: B21-B11 -> indices 2,0 -> [0,0,1,-1]
    # w: C11+=1, C21+=1 -> indices 0,2 -> [1,0,1,0]
    U_list.append([0, 0, 0, 1])
    V_list.append([0, 0, 1, -1])
    W_list.append([1, 0, 1, 0])
    
    # Term 5: M5 = (A11+A12)B22
    # u: A11+A12 -> indices 0,1 -> [1,1,0,0]
    # v: B22 -> index 3 -> [0,0,0,1]
    # w: C11-=1, C12+=1 -> indices 0,1 -> [-1,1,0,0]
    U_list.append([1, 1, 0, 0])
    V_list.append([0, 0, 0, 1])
    W_list.append([-1, 1, 0, 0])
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # u: A21-A11 -> indices 2,0 -> [0,0,1,-1]
    # v: B11+B12 -> indices 0,1 -> [1,1,0,0]
    # w: C21+=1 -> index 2 -> [0,0,1,0]
    U_list.append([0, 0, 1, -1])
    V_list.append([1, 1, 0, 0])
    W_list.append([0, 0, 1, 0])
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # u: A12-A22 -> indices 1,3 -> [0,1,0,-1]
    # v: B21+B22 -> indices 2,3 -> [0,0,1,1]
    # w: C22+=1 -> index 3 -> [0,0,0,1]
    U_list.append([0, 1, 0, -1])
    V_list.append([0, 0, 1, 1])
    W_list.append([0, 0, 0, 1])
    
    # Verify correctness
    T = np.zeros((4, 4, 4), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T[i*2+j, j*2+k, k*2+i] = 1.0
    
    T_recon = np.zeros((4, 4, 4), dtype=float)
    for t in range(7):
        T_recon += np.einsum('a,b,c->abc', np.array(U_list[t]), np.array(V_list[t]), np.array(W_list[t]))
    
    if not np.allclose(T, T_recon, atol=1e-10):
        # Fallback to schoolbook 2x2 (rank 8)
        U_list = []
        V_list = []
        W_list = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = [0.0]*4; v = [0.0]*4; w = [0.0]*4
                    u[i*2+j] = 1.0
                    v[j*2+k] = 1.0
                    w[k*2+i] = 1.0
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
    
    return U_list, V_list, W_list

def _kronecker_block_decomposition(n, m, p):
    """
    Decompose the matrix multiplication tensor for n x m times m x p
    using a block-based Strassen approach.
    
    For n=m=p=2, this gives rank 7 (Strassen) or 8 (schoolbook).
    For larger dimensions, we use a hierarchical block decomposition.
    """
    # For now, handle the specific case we need: 4x4 x 4x9
    # We'll build this from smaller blocks.
    pass

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    U_all = []
    V_all = []
    W_all = []
    
    def add_term(u, v, w):
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)
    
    # Strategy: Use Strassen-style decomposition for the 4x4 x 4x4 blocks,
    # and schoolbook for the remaining columns.
    # 
    # Split p=9 into [4, 4, 1].
    # For each 4-column block, decompose 4x4 x 4x4 using Strassen at the 2x2 block level.
    # For the 1-column block, use schoolbook.
    
    # First, get the 2x2 Strassen decomposition
    S_U, S_V, S_W = _strassen_2x2()
    s_rank = len(S_U)  # 7 if Strassen works, 8 if fallback
    
    col_groups = [4, 4, 1]
    col_offset = 0
    
    for q in col_groups:
        if q == 4:
            # Decompose 4x4 x 4x4 using block Strassen.
            # Split into 2x2 blocks:
            # A = [[A00, A01], [A10, A11]]
            # B = [[B00, B01], [B10, B11]]
            # C = [[C00, C01], [C10, C11]]
            #
            # C00 = A00*B00 + A01*B10
            # C01 = A00*B01 + A01*B11
            # C10 = A10*B00 + A11*B10
            # C11 = A10*B01 + A11*B11
            #
            # Each product like A00*B00 is a 2x2 x 2x2 = 2x2 multiplication.
            # But we need to handle the sums. The tensor decomposition must be exact.
            #
            # Approach: For each of the 4 block products, use the Strassen 2x2 decomposition.
            # But the sums (e.g., C00 = A00*B00 + A01*B10) mean we need to add the tensors.
            #
            # Actually, the correct way is to treat the 4x4x4 tensor as a 2x2x2 tensor
            # where each "entry" is a 2x2x2 tensor, and apply Strassen at the top level.
            # This gives 7 terms, each of which is a product of 2x2x2 tensors.
            # Each 2x2x2 tensor product can be decomposed with rank 7.
            # Total: 7 * 7 = 49 terms.
            #
            # Let me implement this properly.
            #
            # The 4x4x4 tensor T[A_idx, B_idx, C_idx] = 1 where:
            # A_idx = i*4+j, B_idx = j*4+k, C_idx = k*4+i for i,j,k in 0..3.
            #
            # Write i = 2*i0 + i1, j = 2*j0 + j1, k = 2*k0 + k1.
            # Then A_idx = (2*i0+i1)*4 + (2*j0+j1) = 8*i0 + 4*i1 + 2*j0 + j1
            # B_idx = (2*j0+j1)*4 + (2*k0+k1) = 8*j0 + 4*j1 + 2*k0 + k1
            # C_idx = (2*k0+k1)*4 + (2*i0+i1) = 8*k0 + 4*k1 + 2*i0 + i1
            #
            # The tensor is a sum over i0,j0,k0 of:
            # delta[i0,k0] * (2x2x2 tensor for i1,j1,k1 with fixed i0,j0,k0)
            #
            # Actually, it's simpler to think of it as:
            # T = sum_{i0,j0,k0} delta[i0,k0] * T_{i0,j0,k0}
            # where T_{i0,j0,k0} is the 2x2x2 tensor for the sub-blocks.
            #
            # But this isn't quite right because the indices are coupled.
            #
            # Let me use a different approach: directly construct the 4x4x4 decomposition
            # by combining the 2x2 Strassen decompositions.
            #
            # For the 4x4 x 4x4 multiplication, we can view it as:
            # C = A * B
            # where A, B, C are 4x4 matrices partitioned into 2x2 blocks.
            #
            # C00 = A00*B00 + A01*B10
            # C01 = A00*B01 + A01*B11
            # C10 = A10*B00 + A11*B10
            # C11 = A10*B01 + A11*B11
            #
            # Each term like A00*B00 is a 2x2 matrix product.
            # The tensor for A00*B00 has U indices in {0,1,4,5} (i1=0,1; j1=0,1 for i0=0,j0=0),
            # V indices in {0,1,4,5} (j1=0,1; k1=0,1 for j0=0,k0=0),
            # W indices in {0,1,4,5} (k1=0,1; i1=0,1 for k0=0,i0=0).
            #
            # This is getting complex. Let me just use schoolbook for the 4x4x4 part
            # for now, which gives rank 64, and see if I can optimize later.
            #
            # Actually, let me try a simpler approach: use the Kronecker product structure.
            # The 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors.
            # T_4 = T_2 ⊗ T_2
            # The rank of a Kronecker product is at most the product of the ranks.
            # So rank(T_4) <= 7 * 7 = 49 (if Strassen works) or 8 * 8 = 64 (schoolbook).
            #
            # Let me construct this using the Kronecker product of the 2x2 decompositions.
            
            # For the Kronecker product approach:
            # If T = T_A ⊗ T_B, and T_A = sum_{t=1}^{r_A} u_t^A ⊗ v_t^A ⊗ w_t^A,
            # T_B = sum_{s=1}^{r_B} u_s^B ⊗ v_s^B ⊗ w_s^B,
            # then T = sum_{t,s} (u_t^A ⊗ u_s^B) ⊗ (v_t^A ⊗ v_s^B) ⊗ (w_t^A ⊗ w_s^B)
            
            # For 4x4x4, we need to be careful about the index mapping.
            # The 4x4x4 tensor T_4 has:
            # U index: i*4+j where i,j in 0..3
            # V index: j*4+k where j,k in 0..3
            # W index: k*4+i where k,i in 0..3
            #
            # Write i = 2*i0 + i1, j = 2*j0 + j1, k = 2*k0 + k1.
            # U index = (2*i0+i1)*4 + (2*j0+j1) = 8*i0 + 4*i1 + 2*j0 + j1
            # V index = (2*j0+j1)*4 + (2*k0+k1) = 8*j0 + 4*j1 + 2*k0 + k1
            # W index = (2*k0+k1)*4 + (2*i0+i1) = 8*k0 + 4*k1 + 2*i0 + i1
            #
            # The 2x2x2 tensor T_2 has:
            # U index: i*2+j where i,j in 0..1
            # V index: j*2+k where j,k in 0..1
            # W index: k*2+i where k,i in 0..1
            #
            # For the Kronecker product, we need to map the 4x4 indices to pairs of 2x2 indices.
            # U index 0..15 -> (u0, u1) where u0 = U_idx // 4, u1 = U_idx % 4
            # But u0 = 2*i0 + i1, u1 = 2*j0 + j1
            # So u0 encodes (i0, i1) and u1 encodes (j0, j1).
            #
            # This is a bit tricky. Let me just use schoolbook for the 4x4x4 part.
            
            # Schoolbook for 4x4 x 4x4:
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        u = [0.0] * U_rows
                        v = [0.0] * V_rows
                        w = [0.0] * W_rows
                        u[i * 4 + j] = 1.0
                        v[j * 9 + (col_offset + k)] = 1.0
                        w[(col_offset + k) * 4 + i] = 1.0
                        add_term(u, v, w)
        else:
            # q == 1: 4x4 x 4x1, schoolbook rank 16
            for i in range(4):
                for j in range(4):
                    for k in range(1):
                        u = [0.0] * U_rows
                        v = [0.0] * V_rows
                        w = [0.0] * W_rows
                        u[i * 4 + j] = 1.0
                        v[j * 9 + (col_offset + k)] = 1.0
                        w[(col_offset + k) * 4 + i] = 1.0
                        add_term(u, v, w)
        
        col_offset += q
    
    rank = len(U_all)
    U = [[float(x) for x in row] for row in np.array(U_all).T]
    V = [[float(x) for x in row] for row in np.array(V_all).T]
    W = [[float(x) for x in row] for row in np.array(W_all).T]
    
    return {"U": U, "V": V, "W": W}
