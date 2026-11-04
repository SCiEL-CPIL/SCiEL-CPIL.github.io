import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Standard rank
    rank_std = n * m * p
    U_std = [[0] * rank_std for _ in range(n * m)]
    V_std = [[0] * rank_std for _ in range(m * p)]
    W_std = [[0] * rank_std for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_std[i * m + j][col] = 1
                V_std[j * p + k][col] = 1
                W_std[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank
    # Target tensor T
    # Indices: 
    # U: i*m + j (6)
    # V: j*p + k (9)
    # W: k*n + i (6)
    # Tensor component T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Note: The sum is over j for the multiplication.
    # C_{i', k'} = sum_j A_{i', j} B_{j, k'}
    # So T_{(i', j), (j, k'), (k', i')} = 1.
    # In our indexing:
    # a = i' * m + j
    # b = j * p + k'
    # c = k' * n + i'
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Explicit construction of a rank-17 decomposition using Strassen's algorithm for the 2x2x2 block
    # and direct multiplication for the remaining 2x2x1 and 2x1x3 blocks.
    # 
    # We view the 2x3 matrix A as [A_0 | A_1] where A_0, A_1 are 2x2.
    # We view the 3x3 matrix B as [B_0 B_1 B_2] where B_0, B_1, B_2 are 3x3? No, B is 3x3.
    # Let's partition A (2x3) into A0 (2x2) and A1 (2x1).
    # Let's partition B (3x3) into B0 (2x3) and B1 (1x3).
    # C = A0 * B0 + A1 * B1.
    # A0*B0 is 2x2 * 2x3 = 2x3. This can be done with Strassen for 2x2*2x2 part?
    # Actually, let's use the known result: rank(2x3 * 3x3) <= 17.
    # One construction:
    # Split A = [A11 A12 | A21 A22] (2x2 blocks? No, 2x3)
    # Let A = [[a11, a12, a13], [a21, a22, a23]]
    # Let B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    #
    # Consider the product C = A B.
    # C11 = a11*b11 + a12*b21 + a13*b31
    # C12 = a11*b12 + a12*b22 + a13*b32
    # C13 = a11*b13 + a12*b23 + a13*b33
    # C21 = a21*b11 + a22*b21 + a23*b31
    # C22 = a21*b12 + a22*b22 + a23*b32
    # C23 = a21*b13 + a22*b23 + a23*b33
    #
    # This is the tensor product of (2x1x3) and (1x3x3)? No.
    # It is the tensor [2,3,3].
    # 
    # Known decomposition of rank 17:
    # We can embed the 2x3x3 tensor into a 3x3x3 tensor?
    # Rank(3x3x3) <= 23 (Strassen).
    # 
    # Let's try a specific construction found in literature or derived.
    # Strassen's algorithm for 2x2x2 has rank 7.
    # We can decompose 2x3x3 as 2x2x2 + 2x2x1 + 2x1x2 + 2x1x1?
    # If we split the 3 dimension into 2+1.
    # A (2x3) -> A_left (2x2), A_right (2x1)
    # B (3x3) -> B_top (2x3), B_bottom (1x3)
    # C = A_left * B_top + A_right * B_bottom
    # A_left * B_top is 2x2 * 2x3.
    # Let B_top = [B_11, B_12, B_13] where B_1j are 2x1? No, B_top is 2x3.
    # Let's split the middle dimension of B_top (which is 2) ? No, B_top is 2x3.
    # The product A_left (2x2) * B_top (2x3) requires rank(2,2,3).
    # Rank(2,2,3) <= Rank(2,2,2) + Rank(2,2,1) = 7 + 4 = 11?
    # Actually Rank(2,2,3) is known to be 11? Or 10?
    # If Rank(2,2,3) = 11, then total rank = 11 + Rank(2,1,3).
    # Rank(2,1,3) is just matrix multiplication 2x1 * 1x3 = 2x3 matrix.
    # This is an outer product of a 2x1 vector and a 1x3 vector? No.
    # C = u * v^T where u is 2x1, v is 3x1? No, C is 2x3.
    # C_{i,k} = u_i * v_k. This is rank 1 in the tensor sense?
    # T_{i, j, k} = delta_{j,0} * 1? No.
    # The tensor for 2x1 * 1x3 has indices i (2), j (1), k (3).
    # T[i, 0, k] = 1.
    # This tensor has rank 1? No, it's the identity in a way?
    # U[i, 0] = 1, V[0, 0] = 1, W[k, 0] = 1?
    # Sum_t U[i,t] V[j,t] W[k,t].
    # If t=0: U[i,0]=1, V[0,0]=1, W[k,0]=1.
    # Then T[i,0,k] = 1.
    # Yes, Rank(2,1,3) = 1? Wait.
    # The multiplication map is (u, v) -> u v^T.
    # The tensor is sum_{i,k} e_i e_0 e_k.
    # This is not rank 1. It is sum of 6 rank 1 tensors?
    # No, e_i e_0 e_k is rank 1.
    # The sum of 6 rank 1 tensors is not necessarily rank 6.
    # But actually, the tensor for 1x1 matrix multiplication (scalar) is rank 1.
    # For 2x1 * 1x3, the result is a 2x3 matrix.
    # Each entry C_{ik} = A_{i0} B_{0k}.
    # This is a sum of products.
    # It can be written as:
    # C = (A_col) (B_row^T)
    # This is an outer product of a vector (2) and a vector (3).
    # In tensor decomposition, this corresponds to rank 1?
    # Let's check.
    # T_{i, j, k} = 1 if j=0 else 0.
    # U[i, 0] = 1, V[0, 0] = 1, W[k, 0] = 1.
    # Then sum_t U[i,t]V[j,t]W[k,t] = 1*1*1 = 1 for j=0.
    # So yes, Rank(2,1,3) = 1?
    # Wait, U has shape (2,1), V has shape (1,1), W has shape (3,1).
    # This works.
    # So Rank(2,1,3) = 1.
    #
    # Now we need Rank(2,2,3).
    # We can decompose 2x2 * 2x3 as 2x2 * 2x2 + 2x2 * 2x1.
    # Rank(2,2,2) = 7 (Strassen).
    # Rank(2,2,1) = 4 (standard).
    # Total = 7 + 4 = 11.
    # So Rank(2,2,3) <= 11.
    #
    # Total Rank = Rank(2,2,3) + Rank(2,1,3) = 11 + 1 = 12?
    # Wait, the decomposition C = A_left * B_top + A_right * B_bottom.
    # The tensors are disjoint in terms of indices?
    # A_left uses A_{i, 0:2}, B_top uses B_{0:2, k}.
    # A_right uses A_{i, 2}, B_bottom uses B_{2, k}.
    # The sum is over the middle index j.
    # j=0,1 for A_left * B_top.
    # j=2 for A_right * B_bottom.
    # The tensors are:
    # T1[i, j, k] = 1 if j in {0,1}. (2x2x3 tensor)
    # T2[i, j, k] = 1 if j == 2. (2x1x3 tensor)
    # Total tensor T = T1 + T2.
    # The rank of a sum is at most the sum of ranks.
    # So Rank(T) <= Rank(T1) + Rank(T2) = 11 + 1 = 12.
    #
    # Is Rank(2,2,3) really 11?
    # Let's verify Rank(2,2,1).
    # 2x2 * 2x1. Result 2x1.
    # C_i = sum_j A_{ij} B_{j0}.
    # T[i, j, k] with k=0.
    # This is standard 2x2 * 2x1 multiplication.
    # Rank is 4?
    # Yes, each of the 2 outputs requires 2 multiplications?
    # C_0 = A_00 B_00 + A_01 B_10
    # C_1 = A_10 B_00 + A_11 B_10
    # Can we do better?
    # Strassen doesn't apply directly to rectangular.
    # But 2x2 * 2x1 is equivalent to 2x2 * 1x2^T?
    # Rank(2,2,1) is known to be 4.
    #
    # So Rank(2,2,3) <= 7 + 4 = 11.
    # And Rank(2,1,3) = 1?
    # Let's re-verify Rank(2,1,3).
    # T[i, j, k] = delta_{j,0}.
    # U[i, 0] = 1, V[0, 0] = 1, W[k, 0] = 1.
    # This gives T[i, 0, k] = 1.
    # Yes, rank 1.
    #
    # So total rank <= 12.
    #
    # Can we do better?
    # Maybe Rank(2,2,3) is less than 11?
    # If Rank(2,2,3) = 10, total = 11.
    #
    # Let's construct the rank 12 solution first.
    #
    # Strassen for 2x2x2:
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B01 - B11)
    # m4 = A11 * (B10 - B00)
    # m5 = (A00 + A01) * B11
    # m6 = (A10 - A00) * (B00 + B01)
    # m7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # We need to map this to our indices.
    # T1 is 2x2x3.
    # A is 2x2 (indices i, j).
    # B is 2x3 (indices j, k).
    # C is 2x3 (indices i, k).
    #
    # Let's split B (2x3) into B_left (2x2) and B_right (2x1).
    # B_left has k=0,1. B_right has k=2.
    #
    # Part 1: A * B_left (2x2 * 2x2). Rank 7.
    # Part 2: A * B_right (2x2 * 2x1). Rank 4.
    #
    # Part 1:
    # A = [[A00, A01], [A10, A11]]
    # B_left = [[B00, B01], [B10, B11]] (Note: B indices are j, k. j in {0,1}, k in {0,1})
    #
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B01 - B11)
    # m4 = A11 * (B10 - B00)
    # m5 = (A00 + A01) * B11
    # m6 = (A10 - A00) * (B00 + B01)
    # m7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # Part 2:
    # A (2x2) * B_right (2x1).
    # B_right = [[B02], [B12]] (j in {0,1}, k=2)
    #
    # C02 = A00 * B02 + A01 * B12
    # C12 = A10 * B02 + A11 * B12
    #
    # Standard rank 4:
    # m8 = A00 * B02
    # m9 = A01 * B12
    # m10 = A10 * B02
    # m11 = A11 * B12
    #
    # C02 = m8 + m9
    # C12 = m10 + m11
    #
    # Part 3: A_right * B_bottom (2x1 * 1x3). Rank 1.
    # A_right = [[A02], [A12]] (i in {0,1}, j=2)
    # B_bottom = [B20, B21, B22] (j=2, k in {0,1,2})
    #
    # C0k = A02 * B2k
    # C1k = A12 * B2k
    #
    # m12 = A02 * B20
    # m13 = A02 * B21
    # m14 = A02 * B22
    # m15 = A12 * B20
    # m16 = A12 * B21
    # m17 = A12 * B22
    #
    # Wait, Rank(2,1,3) = 1?
    # If Rank(2,1,3) = 1, then we can do it with 1 term?
    # m12 = (A02 + A12) * (B20 + B21 + B22)? No.
    # The tensor is T[i, 2, k] = 1.
    # U[i, t] V[2, t] W[k, t].
    # If t=0: U[i,0]=1, V[2,0]=1, W[k,0]=1.
    # Then T[i,2,k] = 1.
    # This works for all i,k.
    # So yes, 1 term is enough for the whole block?
    # No, the values are A_{i2} * B_{2k}.
    # The term is U[i,0] * V[2,0] * W[k,0].
    # If U[i,0]=1, V[2,0]=1, W[k,0]=1, the product is 1.
    # But we need A_{i2} * B_{2k}.
    # The decomposition is sum_t U[i,t] V[j,t] W[k,t].
    # This sums to 1.
    # But the actual value is A_{i2} B_{2k}.
    # The tensor T_{i,j,k} = 1 if j=2.
    # The decomposition gives sum_t ... = 1.
    # This matches T_{i,2,k} = 1.
    # So yes, rank 1.
    #
    # So total rank = 7 (Strassen) + 4 (Standard 2x2*2x1) + 1 (Standard 2x1*1x3) = 12.
    #
    # Let's build the U, V, W matrices.
    #
    # Dimensions:
    # U: (2*3, 12) = (6, 12)
    # V: (3*3, 12) = (9, 12)
    # W: (3*2, 12) = (6, 12)
    #
    # Indices:
    # U[i*m + j, t]
    # V[j*p + k, t]
    # W[k*n + i, t]
    #
    # n=2, m=3, p=3.
    #
    # Term 0 (m1):
    # U: (A00 + A11) -> U[0,0]=1, U[3,0]=1
    # V: (B00 + B11) -> V[0,0]=1, V[4,0]=1 (B00 is j=0,k=0; B11 is j=1,k=1)
    # W: C00, C11 -> W[0,0]=1, W[1,0]=1 (C00 is k=0,i=0; C11 is k=1,i=1)
    #
    # Term 1 (m2):
    # U: (A10 + A11) -> U[3,1]=1, U[4,1]=1
    # V: B00 -> V[0,1]=1
    # W: C10 -> W[2,1]=1 (C10 is k=0,i=1)
    #
    # Term 2 (m3):
    # U: A00 -> U[0,2]=1
    # V: (B01 - B11) -> V[1,2]=1, V[4,2]=-1
    # W: C01 -> W[1,2]=1 (C01 is k=1,i=0)
    #
    # Term 3 (m4):
    # U: A11 -> U[4,3]=1
    # V: (B10 - B00) -> V[3,3]=1, V[0,3]=-1
    # W: C00, C10 -> W[0,3]=1, W[2,3]=1
    #
    # Term 4 (m5):
    # U: (A00 + A01) -> U[0,4]=1, U[1,4]=1
    # V: B11 -> V[4,4]=1
    # W: -C00, C01 -> W[0,4]=-1, W[1,4]=1
    #
    # Term 5 (m6):
    # U: (A10 - A00) -> U[3,5]=1, U[0,5]=-1
    # V: (B00 + B01) -> V[0,5]=1, V[1,5]=1
    # W: C11 -> W[1,5]=1
    #
    # Term 6 (m7):
    # U: (A01 - A11) -> U[1,6]=1, U[4,6]=-1
    # V: (B10 + B11) -> V[3,6]=1, V[4,6]=1
    # W: C00 -> W[0,6]=1
    #
    # Term 7 (m8):
    # U: A00 -> U[0,7]=1
    # V: B02 -> V[2,7]=1 (B02 is j=0,k=2)
    # W: C02 -> W[3,7]=1 (C02 is k=2,i=0)
    #
    # Term 8 (m9):
    # U: A01 -> U[1,8]=1
    # V: B12 -> V[5,8]=1 (B12 is j=1,k=2)
    # W: C02 -> W[3,8]=1
    #
    # Term 9 (m10):
    # U: A10 -> U[3,9]=1
    # V: B02 -> V[2,9]=1
    # W: C12 -> W[4,9]=1 (C12 is k=2,i=1)
    #
    # Term 10 (m11):
    # U: A11 -> U[4,10]=1
    # V: B12 -> V[5,10]=1
    # W: C12 -> W[4,10]=1
    #
    # Term 11 (m12...m17 combined? No, rank 1 for the block)
    # Wait, my previous analysis said Rank(2,1,3)=1.
    # But the terms are A02*B20, A02*B21, etc.
    # The tensor T[i,2,k]=1.
    # U[i,11]=1, V[2,11]=1, W[k,11]=1.
    # This adds 1 to all C[i,k] for i=0,1 and k=0,1,2?
    # No, W[k,11]=1 for all k.
    # So it adds 1 to C00, C01, C02, C10, C11, C12.
    # But we only want to add A02*B2k and A12*B2k.
    # The tensor T[i,2,k]=1 means that for every i,k, the coefficient is 1.
    # The decomposition sum_t U[i,t]V[2,t]W[k,t] = 1.
    # This is correct for the tensor.
    #
    # So Term 11:
    # U[0,11]=1, U[1,11]=1
    # V[2,11]=1
    # W[0,11]=1, W[1,11]=1, W[2,11]=1, W[3,11]=1, W[4,11]=1, W[5,11]=1
    #
    # Let's verify this sum.
    #
    # Initialize U, V, W as zeros.
    #
    rank_17 = 17 # We are aiming for 12, which is < 17.
    # Actually, let's just return the rank 12 solution if valid.
    #
    U_12 = np.zeros((6, 12))
    V_12 = np.zeros((9, 12))
    W_12 = np.zeros((6, 12))
    
    # Term 0
    U_12[0, 0] = 1; U_12[3, 0] = 1
    V_12[0, 0] = 1; V_12[4, 0] = 1
    W_12[0, 0] = 1; W_12[1, 0] = 1
    
    # Term 1
    U_12[3, 1] = 1; U_12[4, 1] = 1
    V_12[0, 1] = 1
    W_12[2, 1] = 1
    
    # Term 2
    U_12[0, 2] = 1
    V_12[1, 2] = 1; V_12[4, 2] = -1
    W_12[1, 2] = 1
    
    # Term 3
    U_12[4, 3] = 1
    V_12[3, 3] = 1; V_12[0, 3] = -1
    W_12[0, 3] = 1; W_12[2, 3] = 1
    
    # Term 4
    U_12[0, 4] = 1; U_12[1, 4] = 1
    V_12[4, 4] = 1
    W_12[0, 4] = -1; W_12[1, 4] = 1
    
    # Term 5
    U_12[3, 5] = 1; U_12[0, 5] = -1
    V_12[0, 5] = 1; V_12[1, 5] = 1
    W_12[1, 5] = 1
    
    # Term 6
    U_12[1, 6] = 1; U_12[4, 6] = -1
    V_12[3, 6] = 1; V_12[4, 6] = 1
    W_12[0, 6] = 1
    
    # Term 7
    U_12[0, 7] = 1
    V_12[2, 7] = 1
    W_12[3, 7] = 1
    
    # Term 8
    U_12[1, 8] = 1
    V_12[5, 8] = 1
    W_12[3, 8] = 1
    
    # Term 9
    U_12[3, 9] = 1
    V_12[2, 9] = 1
    W_12[4, 9] = 1
    
    # Term 10
    U_12[4, 10] = 1
    V_12[5, 10] = 1
    W_12[4, 10] = 1
    
    # Term 11
    U_12[0, 11] = 1; U_12[1, 11] = 1
    V_12[2, 11] = 1
    W_12[0, 11] = 1; W_12[1, 11] = 1; W_12[2, 11] = 1
    W_12[3, 11] = 1; W_12[4, 11] = 1; W_12[5, 11] = 1
    
    # Verify
    T_check = np.einsum('at,bt,ct->abc', U_12, V_12, W_12)
    if np.allclose(T_check, T, atol=1e-9):
        return {"U": U_12.tolist(), "V": V_12.tolist(), "W": W_12.tolist()}
    
    # If rank 12 failed (it shouldn't), return standard rank 18
    return {"U": U_std, "V": V_std, "W": W_std}
# EVOLVE-BLOCK-END
