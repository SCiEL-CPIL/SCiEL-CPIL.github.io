import numpy as np
import time
import sys

def check_decomposition_fast(U, V, W, n, m, p):
    """Verify the tensor decomposition identity using NumPy broadcasting."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_reshaped = U_arr.reshape(n, m, -1)
    V_reshaped = V_arr.reshape(m, p, -1)
    W_reshaped = W_arr.reshape(p, n, -1)
    
    S = np.einsum('abr,cdr,kir->abcdki', U_reshaped, V_reshaped, W_reshaped)
    
    a_idx = np.arange(n)
    b_idx = np.arange(m)
    c_idx = np.arange(m)
    d_idx = np.arange(p)
    k_idx = np.arange(p)
    i_idx = np.arange(n)
    
    delta_a_i = (a_idx[:, None, None, None, None, None] == i_idx[None, None, None, None, None, :]).astype(float)
    delta_b_c = (b_idx[None, :, None, None, None, None] == c_idx[None, None, :, None, None, None]).astype(float)
    delta_d_k = (d_idx[None, None, None, :, None, None] == k_idx[None, None, None, None, :, None]).astype(float)
    
    T = delta_a_i * delta_b_c * delta_d_k
    
    return np.allclose(S, T, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 6
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # We aim for rank 27.
    # Structure:
    # The multiplication is C = A * B where A is 2x3 and B is 3x6.
    # We can view B as [B1, B2, B3] where Bi are 3x2 blocks.
    # C = A1*B1 + A2*B2 + A3*B3.
    # Each Ai is 2x1 (column of A), Bi is 3x2.
    # Wait, A is 2x3. Columns are A[:,0], A[:,1], A[:,2].
    # B is 3x6. Columns are B[:,0]...B[:,5].
    # C[:,j] = A * B[:,j].
    
    # Let's use the "3 Strassen blocks" idea more carefully.
    # If we split B into three 3x2 blocks B0, B1, B2.
    # C0 = A * B0, C1 = A * B1, C2 = A * B2.
    # Each Ci is 2x2.
    # We know rank(2x2x2) = 7.
    # If we just did 3 independent 2x2x2 multiplications, rank would be 3*7 = 21.
    # However, the A matrix is shared.
    # In the tensor decomposition T = sum U_r V_r W_r, the "A" part is U, "B" part is V, "C" part is W.
    # If we treat them independently, we are not sharing the A factors.
    # But the tensor is T[a, b, c, d, k, i].
    # a,i in 0..1 (rows of A, cols of C)
    # b,c in 0..2 (cols of A, rows of B)
    # d,k in 0..5 (cols of B, cols of C)
    
    # The term is 1 if a==i, b==c, d==k.
    # This is exactly the identity tensor for the multiplication.
    
    # Let's try to construct a rank 27 solution.
    # We will use 7 terms for each of the 3 "column pairs" of B?
    # No, the coupling is on b,c (the inner dimension).
    
    # Alternative: Use the known rank of 2x3x3 which is 11?
    # Rank(2x3x3) is not 11. Rank(2x2x2)=7. Rank(2x3x3) >= 11?
    # Actually, the current solution uses 17 for 2x3x3.
    # 34 = 2 * 17.
    
    # Let's try a different decomposition.
    # We can write the 2x3x6 multiplication as a sum of 6 rank-1 terms for the "diagonal" parts?
    # No, that's rank 36.
    
    # Let's try to find a rank 27 solution by construction.
    # We use 3 blocks of Strassen (7 each) = 21.
    # We need 6 more terms to handle the "off-diagonal" couplings between the blocks?
    
    # Let's define the 3 Strassen blocks for C0, C1, C2.
    # Block 0 handles C[:, 0:2] = A * B[:, 0:2]
    # Block 1 handles C[:, 2:4] = A * B[:, 2:4]
    # Block 2 handles C[:, 4:6] = A * B[:, 4:6]
    
    # In a standard tensor decomposition, the terms are global.
    # If we use 3 disjoint sets of terms, each set r in Block_j only contributes to C[:, 2j:2j+2].
    # This means W vectors for Block_j have non-zeros only in indices k=2j, 2j+1.
    # U vectors for Block_j can be anything.
    # V vectors for Block_j can be anything.
    
    # If we do this, we are effectively doing 3 independent 2x3x2 multiplications.
    # What is the rank of 2x3x2 multiplication?
    # It is the rank of the tensor T[a, b, c, d, k, i] where d,k in {0,1} (local), b,c in {0,1,2}, a,i in {0,1}.
    # This is equivalent to multiplying a 2x3 matrix by a 3x2 matrix.
    # Let's call this rank R_232.
    # If R_232 = 7, then total rank = 21.
    # If R_232 = 8, total rank = 24.
    
    # Is rank(2x3x2) = 7?
    # 2x3x2 is a "slice" of 2x3x3.
    # Known: rank(2x2x2) = 7.
    # 2x3x2 can be seen as 2x2x2 + 2x1x2 + 1x2x2? No.
    # It is A(2x3) * B(3x2).
    # We can split A into A1(2x2) and A2(2x1).
    # C = A1*B1 + A2*B2.
    # This doesn't help directly with rank.
    
    # Let's check if rank(2x3x2) is 7.
    # If we use Strassen on the 2x2 part (A1, B1), we get 7 terms for C1 = A1*B1.
    # We still need to compute A2*B2 (2x1 * 1x2 -> 2x2).
    # A2 is 2x1 (vector), B2 is 1x2 (row vector).
    # C2 = A2 * B2 is an outer product.
    # C2[i, k] = A2[i] * B2[k].
    # This is a rank-1 tensor in the (i, k) space, but it depends on A2 and B2.
    # In the full tensor T[a,b,c,d,k,i], the term for b=2, c=2 is T[a, 2, 2, d, k, i] = 1 if a==i and d==k.
    # This is exactly the outer product tensor of the identity?
    # T[a, 2, 2, d, k, i] = delta_{a,i} delta_{d,k}.
    # This can be decomposed into 2 rank-1 terms:
    # Term 1: U[0, r]=1, U[1, r]=0; V[2, r]=1, V[5, r]=1 (d=0,k=0? No, d and k are indices of B/C).
    # Wait, V index is c*p + d. c=2, d in 0..1 (local).
    # W index is k*n + i. k in 0..1 (local), i in 0..1.
    
    # Let's analyze the "cross" term b=2, c=2.
    # We need to represent delta_{a,i} delta_{d,k} for a,i in 0,1 and d,k in 0,1.
    # This is the tensor product of two identity matrices (2x2).
    # Rank of I_2 \otimes I_2 is 4?
    # I_2 \otimes I_2 = sum_{i=0,1} sum_{k=0,1} (e_i \otimes e_k) \otimes (e_i \otimes e_k)?
    # In the tensor decomposition U V W:
    # We need sum_r U_r V_r W_r = T.
    # For the slice b=2, c=2:
    # Sum_r U[a, 2, r] V[2, d, r] W[k, i, r] = delta_{a,i} delta_{d,k}.
    # Note: V index here is c=2, d. So V[2*p + d, r].
    # W index is k, i.
    
    # Let's try to decompose delta_{a,i} delta_{d,k} into rank 4.
    # r=0: a=0, i=0, d=0, k=0.
    # U[0, 0]=1, U[1, 0]=0. V[2, 0]=1, V[2, 1]=0. W[0, 0]=1, W[0, 1]=0.
    # This gives T[0, 2, 2, 0, 0, 0] = 1.
    # r=1: a=0, i=0, d=1, k=1.
    # U[0, 1]=1, U[1, 1]=0. V[2, 1]=1, V[2, 0]=0. W[1, 0]=1, W[1, 1]=0.
    # This gives T[0, 2, 2, 1, 1, 0] = 1.
    # r=2: a=1, i=1, d=0, k=0.
    # U[0, 2]=0, U[1, 2]=1. V[2, 0]=1, V[2, 1]=0. W[0, 1]=1, W[0, 0]=0.
    # This gives T[1, 2, 2, 0, 0, 1] = 1.
    # r=3: a=1, i=1, d=1, k=1.
    # U[0, 3]=0, U[1, 3]=1. V[2, 1]=1, V[2, 0]=0. W[1, 1]=1, W[1, 0]=0.
    # This gives T[1, 2, 2, 1, 1, 1] = 1.
    
    # So the "diagonal" part b=c=2 requires rank 4.
    # What about the "off-diagonal" parts b!=c?
    # If b!=c, T=0.
    # In the Strassen part for b,c in {0,1}, we handle b,c in {0,1}.
    # The Strassen decomposition for 2x2x2 (A1, B1) handles b,c in {0,1}.
    # It produces correct values for C1 = A1 B1.
    # Does it produce 0 for b=2 or c=2?
    # In the Strassen terms, U and V are non-zero only for indices in the 2x2 sub-block.
    # So for b=2 or c=2, the Strassen terms contribute 0.
    # Thus, the Strassen terms (rank 7) handle the b,c in {0,1} case perfectly for the 2x2x2 sub-problem.
    # The remaining part is the b=2, c=2 case, which we showed requires rank 4.
    # Total rank for 2x3x2 = 7 + 4 = 11.
    
    # So for 2x3x6, if we split into three 2x3x2 blocks:
    # Block 0 (cols 0,1 of B): rank 11.
    # Block 1 (cols 2,3 of B): rank 11.
    # Block 2 (cols 4,5 of B): rank 11.
    # Total rank = 33.
    # This is better than 34!
    
    # Can we do better?
    # Maybe the rank of 2x3x2 is less than 11?
    # Lower bound for 2x3x2:
    # It contains 2x2x2 (rank 7) and the outer product part.
    # Actually, 2x3x2 is equivalent to 3x2x2? No.
    # Let's check if rank 10 is possible.
    # If we can share terms between the "Strassen" part and the "outer product" part?
    # The Strassen part uses b,c in {0,1}. The outer product uses b,c=2.
    # They are disjoint in the b,c indices.
    # So the tensor is a direct sum of T_{01,01} and T_{2,2}.
    # Rank(T1 + T2) = Rank(T1) + Rank(T2) if supports are disjoint in at least one mode.
    # Here, the support in (b,c) is disjoint: {0,1}x{0,1} vs {2}x{2}.
    # So Rank(2x3x2) = Rank(2x2x2) + Rank(2x1x2) = 7 + 4 = 11.
    # Wait, Rank(2x1x2) is the rank of the tensor delta_{a,i} delta_{d,k}.
    # This is I_2 \otimes I_2. Rank is 4.
    # So Rank(2x3x2) = 11.
    # Therefore Rank(2x3x6) = 3 * 11 = 33.
    
    # The current program has 34. We can improve to 33.
    # Let's construct the rank 33 solution.
    
    target_rank = 33
    U = np.zeros((dim_u, target_rank))
    V = np.zeros((dim_v, target_rank))
    W = np.zeros((dim_w, target_rank))
    
    # We will fill 3 blocks of 11.
    # Block j corresponds to B columns 2j, 2j+1.
    # Local indices for B/C: d, k in {0, 1}. Global: D = 2j+d, K = 2j+k.
    
    for j in range(3):
        offset = j * 11
        # 7 Strassen terms for b,c in {0,1}
        # Standard Strassen for 2x2x2:
        # U, V, W are 2x2 matrices (flattened).
        # Here U is 2x2 (a,b in 0,1), V is 2x2 (c,d in 0,1 local), W is 2x2 (k,i in 0,1 local).
        
        # Strassen Terms:
        # 1: U=I, V=I, W=I
        # 2: U=[1,0;0,1], V=[0,1;1,0], W=[1,1;0,0]? No, standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Let's use the explicit vectors from the current program's U_half, V_half, W_half for the first 7 terms.
        # U_half[0:6, 0:7] etc.
        
        U_strassen = np.array([
            [1, 1, 1, 0, 1, -1, 1],
            [0, 0, 0, 1, 1, 0, -1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 1, -1, -1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0]
        ])
        # Note: U_half in current code is 6x17. The first 7 columns are the Strassen part for 2x2x2.
        # Indices:
        # 0: a=0,b=0
        # 1: a=0,b=1
        # 2: a=1,b=0
        # 3: a=1,b=1
        # 4: a=0,b=0 (wait, current code uses 6 rows? n*m=6. 2*3=6.
        # Rows 0,1 are a=0. Rows 2,3 are a=1?
        # Current code: U_half[0,0]=1 (a=0,b=0), U_half[4,0]=1 (a=1,b=1? No, 4 is a=1,b=1 if 0,1 are a=0 and 2,3 are a=1?
        # Let's check current code indices.
        # n=2, m=3.
        # Index = a*m + b.
        # 0: a=0,b=0
        # 1: a=0,b=1
        # 2: a=0,b=2
        # 3: a=1,b=0
        # 4: a=1,b=1
        # 5: a=1,b=2
        
        # Current code U_half:
        # Col 0: U[0,0]=1, U[4,0]=1. (a=0,b=0) and (a=1,b=1). This is A11+A22. Correct.
        # Col 1: U[3,1]=1, U[4,1]=1. (a=1,b=0) and (a=1,b=1). This is A21+A22. Correct.
        # Col 2: U[0,2]=1. A11. Correct.
        # Col 3: U[4,3]=1. A22. Correct.
        # Col 4: U[0,4]=1, U[1,4]=1. (a=0,b=0) and (a=0,b=1). A11+A12. Correct.
        # Col 5: U[3,5]=1, U[0,5]=-1. (a=1,b=0) and (a=0,b=0). A21-A11. Correct.
        # Col 6: U[1,6]=1, U[4,6]=-1. (a=0,b=1) and (a=1,b=1). A12-A22. Correct.
        
        # So U_strassen for b in 0,1 is:
        # Row 0 (a=0,b=0): 1, 0, 1, 0, 1, -1, 1
        # Row 1 (a=0,b=1): 0, 0, 0, 0, 1, 0, 1
        # Row 2 (a=0,b=2): 0, 0, 0, 0, 0, 0, 0
        # Row 3 (a=1,b=0): 0, 1, 0, 0, 0, 1, 0
        # Row 4 (a=1,b=1): 1, 1, 0, 1, 0, 0, -1
        # Row 5 (a=1,b=2): 0, 0, 0, 0, 0, 0, 0
        
        U_block = np.zeros((6, 11))
        U_block[0:2, 0:7] = [[1, 0, 1, 0, 1, -1, 1],
                             [0, 0, 0, 0, 1, 0, 1]]
        U_block[3:5, 0:7] = [[0, 1, 0, 0, 0, 1, 0],
                             [1, 1, 0, 1, 0, 0, -1]]
        
        # V_strassen for c in 0,1, d in 0,1 (local)
        # Current code V_half:
        # Col 0: V[0,0]=1, V[4,0]=1. (c=0,d=0) and (c=1,d=1). B11+B22.
        # Col 1: V[0,1]=1. B11.
        # Col 2: V[1,2]=1, V[4,2]=-1. (c=0,d=1) and (c=1,d=1). B12-B22.
        # Col 3: V[3,3]=1, V[0,3]=-1. (c=1,d=0) and (c=0,d=0). B21-B11.
        # Col 4: V[4,4]=1. B22.
        # Col 5: V[0,5]=1, V[1,5]=1. (c=0,d=0) and (c=0,d=1). B11+B12.
        # Col 6: V[3,6]=1, V[4,6]=1. (c=1,d=0) and (c=1,d=1). B21+B22.
        
        # V indices: c*p + d. Here p=2 (local).
        # c=0, d=0 -> 0
        # c=0, d=1 -> 1
        # c=1, d=0 -> 2
        # c=1, d=1 -> 3
        
        V_block = np.zeros((6, 11)) # m*p = 3*2 = 6. c in 0,1,2. d in 0,1.
        # c=0, d=0 (idx 0): 1, 1, 0, -1, 0, 1, 0
        # c=0, d=1 (idx 1): 0, 0, 1, 0, 0, 1, 0
        # c=1, d=0 (idx 2): 0, 0, 0, 1, 0, 0, 1
        # c=1, d=1 (idx 3): 1, 0, -1, 0, 1, 0, 1
        # c=2, d=0 (idx 4): 0
        # c=2, d=1 (idx 5): 0
        
        V_block[0, 0:7] = [1, 1, 0, -1, 0, 1, 0]
        V_block[1, 0:7] = [0, 0, 1, 0, 0, 1, 0]
        V_block[2, 0:7] = [0, 0, 0, 1, 0, 0, 1]
        V_block[3, 0:7] = [1, 0, -1, 0, 1, 0, 1]
        
        # W_strassen for k in 0,1 (local), i in 0,1
        # Current code W_half:
        # Col 0: W[0,0]=1, W[3,0]=1. (k=0,i=0) and (k=1,i=0)?
        # W index: k*n + i. n=2.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # Col 0: W[0,0]=1, W[3,0]=1. (k=0,i=0) and (k=1,i=1). C11+C22.
        # Col 1: W[1,1]=1. (k=0,i=1). C12.
        # Col 2: W[2,2]=1, W[3,2]=1. (k=1,i=0) and (k=1,i=1). C21+C22.
        # Col 3: W[0,3]=1, W[1,3]=1. (k=0,i=0) and (k=0,i=1). C11+C12.
        # Col 4: W[0,4]=-1, W[2,4]=1. (k=0,i=0) and (k=1,i=0). -C11+C21.
        # Col 5: W[3,5]=1. (k=1,i=1). C22.
        # Col 6: W[0,6]=1, W[3,6]=1. (k=0,i=0) and (k=1,i=1). C11+C22.
        
        W_block = np.zeros((4, 11))
        # k=0, i=0 (idx 0): 1, 0, 0, 1, -1, 0, 1
        # k=0, i=1 (idx 1): 0, 1, 0, 1, 0, 0, 0
        # k=1, i=0 (idx 2): 0, 0, 1, 0, 1, 0, 0
        # k=1, i=1 (idx 3): 1, 0, 1, 0, 0, 1, 1
        
        W_block[0, 0:7] = [1, 0, 0, 1, -1, 0, 1]
        W_block[1, 0:7] = [0, 1, 0, 1, 0, 0, 0]
        W_block[2, 0:7] = [0, 0, 1, 0, 1, 0, 0]
        W_block[3, 0:7] = [1, 0, 1, 0, 0, 1, 1]
        
        # Now the 4 terms for b=2, c=2.
        # We need to represent delta_{a,i} delta_{d,k}.
        # r=7: a=0,i=0; d=0,k=0.
        # r=8: a=0,i=0; d=1,k=1.
        # r=9: a=1,i=1; d=0,k=0.
        # r=10: a=1,i=1; d=1,k=1.
        
        # U terms:
        # r=7: a=0. U[0,7]=1.
        # r=8: a=0. U[0,8]=1.
        # r=9: a=1. U[3,9]=1. (a=1,b=0? No, b=2. U[5,9]=1).
        # Wait, U index is a*m+b. b=2.
        # a=0, b=2 -> idx 2.
        # a=1, b=2 -> idx 5.
        
        U_block[2, 7] = 1
        U_block[2, 8] = 1
        U_block[5, 9] = 1
        U_block[5, 10] = 1
        
        # V terms:
        # c=2.
        # r=7: d=0. V[2*2+0, 7]=1 -> V[4,7]=1.
        # r=8: d=1. V[2*2+1, 8]=1 -> V[5,8]=1.
        # r=9: d=0. V[4,9]=1.
        # r=10: d=1. V[5,10]=1.
        
        V_block[4, 7] = 1
        V_block[5, 8] = 1
        V_block[4, 9] = 1
        V_block[5, 10] = 1
        
        # W terms:
        # k,i local.
        # r=7: k=0, i=0. W[0,7]=1.
        # r=8: k=1, i=0. W[2,8]=1.
        # r=9: k=0, i=1. W[1,9]=1.
        # r=10: k=1, i=1. W[3,10]=1.
        
        W_block[0, 7] = 1
        W_block[2, 8] = 1
        W_block[1, 9] = 1
        W_block[3, 10] = 1
        
        # Map to global indices
        for r in range(11):
            global_r = offset + r
            # U
            for a in range(2):
                for b in range(3):
                    U[a*3+b, global_r] = U_block[a*3+b, r]
            # V
            for c in range(3):
                for d in range(2):
                    global_d = 2*j + d
                    V[c*6+global_d, global_r] = V_block[c*2+d, r]
            # W
            for k in range(2):
                for i in range(2):
                    global_k = 2*j + k
                    W[global_k*2+i, global_r] = W_block[k*2+i, r]
    
    # Verify
    if check_decomposition_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 34 if construction fails (should not)
    U34, V34, W34 = get_block_solution()
    return {"U": U34.tolist(), "V": V34.tolist(), "W": W34.tolist()}

def get_block_solution():
    n, m, p = 2, 3, 6
    rank_half = 17
    rank_total = 34
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    U_half = np.zeros((6, rank_half))
    V_half = np.zeros((9, rank_half))
    W_half = np.zeros((6, rank_half))
    
    # Strassen on A(0,1) x B(0,1) -> C(0,1)
    U_half[0, 0] = 1; U_half[4, 0] = 1
    V_half[0, 0] = 1; V_half[4, 0] = 1
    W_half[0, 0] = 1; W_half[3, 0] = 1

    U_half[3, 1] = 1; U_half[4, 1] = 1
    V_half[0, 1] = 1
    W_half[1, 1] = 1

    U_half[0, 2] = 1
    V_half[1, 2] = 1; V_half[4, 2] = -1
    W_half[2, 2] = 1; W_half[3, 2] = 1

    U_half[4, 3] = 1
    V_half[3, 3] = 1; V_half[0, 3] = -1
    W_half[0, 3] = 1; W_half[1, 3] = 1

    U_half[0, 4] = 1; U_half[1, 4] = 1
    V_half[4, 4] = 1
    W_half[0, 4] = -1; W_half[2, 4] = 1

    U_half[3, 5] = 1; U_half[0, 5] = -1
    V_half[0, 5] = 1; V_half[1, 5] = 1
    W_half[3, 5] = 1

    U_half[1, 6] = 1; U_half[4, 6] = -1
    V_half[3, 6] = 1; V_half[4, 6] = 1
    W_half[0, 6] = 1; W_half[3, 6] = 1

    # Column 2 of B: A(0,1,2) x B(0,1,2) col 2 -> C col 2
    U_half[0, 7] = 1
    V_half[2, 7] = 1
    W_half[4, 7] = 1

    U_half[1, 8] = 1
    V_half[5, 8] = 1
    W_half[4, 8] = 1

    U_half[3, 9] = 1
    V_half[2, 9] = 1
    W_half[5, 9] = 1

    U_half[4, 10] = 1
    V_half[5, 10] = 1
    W_half[5, 10] = 1

    # A col 2 x B(0,1,2) -> C(0,1,2)
    U_half[2, 11] = 1
    V_half[6, 11] = 1
    W_half[0, 11] = 1

    U_half[2, 12] = 1
    V_half[7, 12] = 1
    W_half[2, 12] = 1

    U_half[2, 13] = 1
    V_half[8, 13] = 1
    W_half[4, 13] = 1

    U_half[5, 14] = 1
    V_half[6, 14] = 1
    W_half[1, 14] = 1

    U_half[5, 15] = 1
    V_half[7, 15] = 1
    W_half[3, 15] = 1

    U_half[5, 16] = 1
    V_half[8, 16] = 1
    W_half[5, 16] = 1

    # Combine two 2x3x3 blocks
    U[:, :17] = U_half
    U[:, 17:] = U_half

    for j in range(3):
        for k in range(3):
            V[j * 6 + k, :17] = V_half[j * 3 + k, :17]
            V[j * 6 + 3 + k, 17:] = V_half[j * 3 + k, :17]

    for k in range(3):
        for i in range(2):
            W[k * 2 + i, :17] = W_half[k * 2 + i, :17]
            W[(k + 3) * 2 + i, 17:] = W_half[k * 2 + i, :17]

    return U, V, W

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

