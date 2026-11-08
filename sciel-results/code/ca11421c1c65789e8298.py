"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    
    # We want to find a rank-50 decomposition.
    # Structure: two independent 2x2x7 blocks.
    # Each 2x2x7 block is built from a 2x2x2 Strassen part and a 2x2x5 schoolbook part.
    
    # Strassen 2x2x2 decomposition (rank 7)
    # A = [[a11, a12], [a21, a22]] (indices 0,1,2,3 in row-major: a11=0, a12=1, a21=2, a22=3)
    # B = [[b11, b12], [b21, b22]] (indices 0,1,2,3)
    # C = [[c11, c12], [c21, c22]] (indices 0,1,2,3)
    
    # U, V, W are 4x4 matrices for this block (if padded) or lists of vectors.
    # Let's build the U, V, W columns for the 7 Strassen terms.
    
    # Term 1: M1 = (a11 + a22) * (b11 + b22)
    # u1 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # w1 = [1, 0, 0, 1] -> c11 += 1, c12 += 0, c21 += 0, c22 += 1 ? 
    # Wait, C = sum w_t * (u_t . A) * (v_t . B)? No.
    # C[i,k] = sum_t U[i,t] V[k,t] W[i,t]? No.
    # Tensor T = sum_t u_t \otimes v_t \otimes w_t.
    # C[i,k] = sum_t U[i,t] V[k,t] W[i,k,t]? No.
    # The decomposition is: C[i,k] = sum_t (sum_j U[i,j,t] A[i,j]) * (sum_l V[k,l,t] B[k,l]) * W_coeff?
    # No, the standard rank decomposition of the multiplication tensor T_{i,j,k} is:
    # T_{i,j,k} = sum_t U_{i,t} V_{j,t} W_{k,t}
    # Where U, V, W are matrices of size (dim * dim) x rank?
    # Let's check the interface:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # And the output is C = A * B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has indices (i, j, k).
    # T[i, j, k] = delta_{i,j,k} ? No.
    # The multiplication tensor T is defined such that C = sum_t U_t \otimes V_t \otimes W_t.
    # Specifically, C[i,k] = sum_j A[i,j] B[j,k].
    # This can be written as:
    # C = sum_t (U_t \cdot A) \otimes (V_t \cdot B) \cdot W_t?
    # Let's look at the standard form.
    # T = sum_{t=1}^r u^{(t)} \otimes v^{(t)} \otimes w^{(t)}
    # where u^{(t)} \in R^{n \times m}, v^{(t)} \in R^{m \times p}, w^{(t)} \in R^{p \times n}?
    # No, the indices of T are (i, j, k).
    # T[i, j, k] is 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    # Actually, T is the tensor such that <T, A \otimes B> = C (flattened).
    # So T[i, j, k] = 1 if we are multiplying A[i,j] and B[j,k] to get C[i,k].
    # Wait, A[i,j] * B[j,k] contributes to C[i,k].
    # So T[i, j, k] is 1.
    # Then C[i,k] = sum_{j} T[i, j, k] A[i,j] B[j,k].
    # If T = sum_t u_t \otimes v_t \otimes w_t, then
    # T[i, j, k] = sum_t U[i,j,t] V[j,k,t] W[i,k,t]?
    # No, u_t is a vector in the A-space (size nm), v_t in B-space (size mp), w_t in C-space (size pn).
    # But the tensor T is 3-indexed (i, j, k).
    # The decomposition is T[i, j, k] = sum_t U[i, t] V[j, t] W[k, t]?
    # Let's check dimensions.
    # U is (nm, rank). Index i in U corresponds to (row i, col ?) of A?
    # The prompt says: U shape (n*m, rank).
    # So U[i*m+j, t].
    # V shape (m*p, rank). V[j*p+k, t].
    # W shape (p*n, rank). W[k*n+i, t].
    # The tensor entry T[i, j, k] corresponds to the product A[i, j] * B[j, k] contributing to C[i, k].
    # In the flattening:
    # A is flattened to a vector of length nm. Index i*m+j.
    # B is flattened to a vector of length mp. Index j*p+k.
    # C is flattened to a vector of length pn. Index k*n+i.
    # The bilinear form is sum_{i,j,k} T[i, j, k] A_flat[i*m+j] B_flat[j*p+k] = C_flat[k*n+i].
    # So T[i, j, k] must satisfy:
    # C_flat[k*n+i] = sum_{j} A_flat[i*m+j] B_flat[j*p+k].
    # This implies that the term A_flat[i*m+j] * B_flat[j*p+k] adds 1 to C_flat[k*n+i].
    # So the tensor T has non-zero entries at indices corresponding to (i*m+j, j*p+k, k*n+i).
    # The value is 1.
    
    # Now, the rank decomposition T = sum_t U_t \otimes V_t \otimes W_t means:
    # T[idx_A, idx_B, idx_C] = sum_t U[idx_A, t] * V[idx_B, t] * W[idx_C, t].
    
    # Let's verify this against Strassen.
    # For 2x2x2:
    # A = [[a11, a12], [a21, a22]] -> indices 0,1,2,3.
    # B = [[b11, b12], [b21, b22]] -> indices 0,1,2,3.
    # C = [[c11, c12], [c21, c22]] -> indices 0,1,2,3.
    # c11 = a11*b11 + a12*b21.
    # c12 = a11*b12 + a12*b22.
    # c21 = a21*b11 + a22*b21.
    # c22 = a21*b12 + a22*b22.
    
    # Strassen computes 7 products.
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Let's construct U, V, W for one Strassen term t.
    # U_t is a vector of length 4. V_t is a vector of length 4. W_t is a vector of length 4.
    # T = sum_t U_t \otimes V_t \otimes W_t.
    
    # Term 1: M1
    # u1: a11 and a22 have coeff 1. Others 0.
    # u1 = [1, 0, 0, 1]
    # v1: b11 and b22 have coeff 1. Others 0.
    # v1 = [1, 0, 0, 1]
    # w1: c11 has +1, c22 has +1.
    # w1 = [1, 0, 0, 1]
    
    # Let's check if this contributes correctly to T.
    # T[0,0,0] (a11, b11, c11) should get contribution 1*1*1 = 1. Correct.
    # T[0,2,0] (a11, b21, c11) should get contribution 1*0*1 = 0. Correct (M1 doesn't use b21 directly, but the combination does).
    # Wait, the tensor T is fixed. T[i,j,k] is 1 if A[i,j]*B[j,k] contributes to C[i,k].
    # For i=0, j=0, k=0: A[0,0]*B[0,0] contributes to C[0,0]. T[0,0,0] = 1.
    # For i=0, j=1, k=0: A[0,1]*B[1,0] contributes to C[0,0]. T[1,2,0] = 1. (Indices: i*m+j=1, j*p+k=2, k*n+i=0).
    
    # Let's check the sum for T[0,0,0] (a11, b11, c11).
    # It should be 1.
    # Contributions from Strassen terms:
    # t=1: u[0]=1, v[0]=1, w[0]=1. Prod=1.
    # t=2: u[0]=0.
    # t=3: u[0]=1, v[0]=0.
    # t=4: u[0]=0.
    # t=5: u[0]=1, v[0]=0.
    # t=6: u[0]=-1, v[0]=1, w[0]=0.
    # t=7: u[0]=0.
    # Sum = 1. Correct.
    
    # Let's check T[1,2,0] (a12, b21, c11).
    # i=0, j=1, k=0.
    # A index 1, B index 2, C index 0.
    # t=1: u[1]=0.
    # t=2: u[1]=0.
    # t=3: u[1]=0.
    # t=4: u[1]=0.
    # t=5: u[1]=1, v[2]=1, w[0]=0. (c11 coeff is 0 for M5? c11 = M1+M4-M5+M7. So w5[0] = -1).
    # Wait, w5 = [ -1, 1, 0, 0 ]. (c11=-1, c12=1).
    # t=5: u[1]=1, v[2]=1 (b22? No, v5 corresponds to b22. v5 = [0,0,0,1]).
    # So t=5: u[1]=1, v[2]=0. Prod=0.
    # t=6: u[1]=0.
    # t=7: u[1]=1, v[2]=1 (b21? No, v7 corresponds to b21+b22. v7 = [0,0,1,1]).
    # w7 = [1, 0, 0, -1]. (c11=1, c22=-1).
    # t=7: u[1]=1, v[2]=1, w[0]=1. Prod=1.
    # Sum = 1. Correct.
    
    # Okay, the logic holds.
    
    # Let's define the Strassen U, V, W vectors.
    # u1 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # w1 = [1, 0, 0, 1]
    
    # u2 = [0, 0, 1, 1] (a21, a22)
    # v2 = [1, 0, 0, 0] (b11)
    # w2 = [0, 0, 1, 0] (c21)
    
    # u3 = [1, 0, 0, 0] (a11)
    # v3 = [0, 1, 0, -1] (b12, b22)
    # w3 = [0, 1, 0, 0] (c12)
    
    # u4 = [0, 0, 0, 1] (a22)
    # v4 = [0, 0, 1, -1] (b21, b11)
    # w4 = [1, 0, 1, 0] (c11, c21)
    
    # u5 = [1, 1, 0, 0] (a11, a12)
    # v5 = [0, 0, 0, 1] (b22)
    # w5 = [-1, 1, 0, 0] (c11, c12)
    
    # u6 = [-1, 0, 1, 0] (a21, a11)
    # v6 = [1, 1, 0, 0] (b11, b12)
    # w6 = [0, 0, 0, 1] (c22)
    
    # u7 = [0, 1, 0, -1] (a12, a22)
    # v7 = [0, 0, 1, 1] (b21, b22)
    # w7 = [1, 0, 0, -1] (c11, c22)
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, -1]
    ]
    
    # Now we need to build the full 2x4x7 decomposition.
    # This consists of two 2x2x7 blocks.
    # Block 0: A rows 0-1, B cols 0-1, C cols 0-1.
    # Block 1: A rows 0-1, B cols 2-3, C cols 2-3.
    # Wait, B is 4x7. So B has 4 rows and 7 columns.
    # A is 2x4. So A has 2 rows and 4 columns.
    # C is 2x7.
    
    # Let's re-map indices.
    # A[i, j] where i in 0..1, j in 0..3.
    # B[j, k] where j in 0..3, k in 0..6.
    # C[i, k] where i in 0..1, k in 0..6.
    
    # The schoolbook decomposition would have rank 2*4*7 = 56.
    # We want to reduce this.
    # We can view B as two 2x7 matrices? No, B is 4x7.
    # We can view A as two 2x2 matrices?
    # A = [A_left, A_right] where A_left is 2x2 (cols 0,1) and A_right is 2x2 (cols 2,3).
    # C = A * B = A_left * B_top + A_right * B_bottom.
    # Where B_top is the top 2 rows of B (2x7) and B_bottom is the bottom 2 rows of B (2x7).
    
    # So C = A_left * B_top + A_right * B_bottom.
    # Each term is a 2x2 matrix times a 2x7 matrix.
    # The rank of 2x2x7 is at most 2*2*7 = 28 (schoolbook).
    # But we can do better using Strassen for the 2x2 part?
    # No, Strassen reduces the rank of the 2x2x2 multiplication.
    # For 2x2x7, we can decompose the 7 columns of B into 2+2+1+1+1?
    # Or just use Strassen for the first 2 columns and schoolbook for the rest?
    # Rank(2x2x7) = Rank(2x2x2) + Rank(2x2x5)?
    # If we split the 7 columns of B into two sets: {0,1} and {2,3,4,5,6}.
    # Then C = A * B_01 + A * B_23456.
    # The first part is 2x2x2 (rank 7).
    # The second part is 2x2x5 (rank 20 schoolbook).
    # Total rank for one 2x2x7 block = 7 + 20 = 27.
    # Since we have two such blocks (for A_left and A_right), total rank = 27 + 27 = 54.
    # Can we do better?
    # Maybe split B into 2+2+2+1?
    # Rank = 7 + 7 + 7 + 4 = 25.
    # Total rank = 25 + 25 = 50.
    
    # Let's implement the 2x2x25 decomposition for one block.
    # Block structure:
    # A is 2x2. B is 2x7. C is 2x7.
    # B columns 0,1 -> Strassen (rank 7).
    # B columns 2,3 -> Strassen (rank 7).
    # B columns 4,5 -> Strassen (rank 7).
    # B column 6 -> Schoolbook (rank 4).
    # Total rank 25.
    
    # We need to map the local indices (0..1 for A rows, 0..1 for B rows, 0..6 for B/C cols)
    # to the global indices.
    
    # Global A indices: i*m + j. i in 0..1, j in 0..3.
    # For A_left (j in 0,1): local j' = j. Global index = i*4 + j'.
    # For A_right (j in 2,3): local j' = j-2. Global index = i*4 + (j'+2).
    
    # Global B indices: j*p + k. j in 0..3, k in 0..6.
    # For B_top (j in 0,1): local j' = j. Global index = j'*7 + k.
    # For B_bottom (j in 2,3): local j' = j-2. Global index = (j'+2)*7 + k.
    
    # Global C indices: k*n + i. k in 0..6, i in 0..1.
    # Local k' = k. Global index = k'*2 + i.
    
    # Let's build the 25 vectors for one 2x2x7 block.
    
    def build_227_block():
        U_list = []
        V_list = []
        W_list = []
        
        # 1. Strassen for cols 0,1
        # Local A indices: 0,1,2,3.
        # Local B indices: 0,1,2,3 (for cols 0,1, rows 0,1).
        # Wait, B is 2x2 for this part. Indices: j'*2 + k'. j' in 0,1, k' in 0,1.
        # So B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3.
        
        # We need to map local Strassen V vectors (length 4) to global V vectors (length 14? No, 2*7=14).
        # The local V vector has non-zeros at indices 0,1,2,3.
        # These correspond to global B indices for j' in 0,1 and k in 0,1.
        # Global B index = j'*7 + k.
        
        for t in range(7):
            u = strassen_U[t]
            v = strassen_V[t]
            w = strassen_W[t]
            
            # Pad u to length 4 (already 4)
            # Pad v to length 14.
            v_padded = [0]*14
            for idx, val in enumerate(v):
                # idx is 0..3. j'=idx//2, k'=idx%2.
                # Global index = j'*7 + k'.
                j_prime = idx // 2
                k_prime = idx % 2
                v_padded[j_prime*7 + k_prime] = val
            
            # Pad w to length 14? No, W is for C.
            # C is 2x7. Indices k*n+i. k in 0..6, i in 0..1.
            # The Strassen W vector has length 4 (for 2x2 C).
            # Indices: k'*2 + i'. k' in 0,1.
            # So w[0] -> k=0, i=0. w[1] -> k=0, i=1. w[2] -> k=1, i=0. w[3] -> k=1, i=1.
            w_padded = [0]*14
            for idx, val in enumerate(w):
                k_prime = idx // 2
                i_prime = idx % 2
                w_padded[k_prime*2 + i_prime] = val
                
            U_list.append(u)
            V_list.append(v_padded)
            W_list.append(w_padded)
            
        # 2. Strassen for cols 2,3
        for t in range(7):
            u = strassen_U[t]
            v = strassen_V[t]
            w = strassen_W[t]
            
            v_padded = [0]*14
            for idx, val in enumerate(v):
                j_prime = idx // 2
                k_prime = idx % 2
                # k is 2,3. k_global = 2 + k_prime.
                v_padded[j_prime*7 + (2 + k_prime)] = val
                
            w_padded = [0]*14
            for idx, val in enumerate(w):
                k_prime = idx // 2
                i_prime = idx % 2
                # k is 2,3. k_global = 2 + k_prime.
                w_padded[(2 + k_prime)*2 + i_prime] = val
                
            U_list.append(u)
            V_list.append(v_padded)
            W_list.append(w_padded)
            
        # 3. Strassen for cols 4,5
        for t in range(7):
            u = strassen_U[t]
            v = strassen_V[t]
            w = strassen_W[t]
            
            v_padded = [0]*14
            for idx, val in enumerate(v):
                j_prime = idx // 2
                k_prime = idx % 2
                k_global = 4 + k_prime
                v_padded[j_prime*7 + k_global] = val
                
            w_padded = [0]*14
            for idx, val in enumerate(w):
                k_prime = idx // 2
                i_prime = idx % 2
                k_global = 4 + k_prime
                w_padded[k_global*2 + i_prime] = val
                
            U_list.append(u)
            V_list.append(v_padded)
            W_list.append(w_padded)
            
        # 4. Schoolbook for col 6
        # A is 2x1, B is 2x1. C is 2x1.
        # 4 products.
        # a11*b11, a12*b21, a21*b11, a22*b21.
        # Local A indices: 0,1,2,3.
        # Local B indices: 0,1 (for col 6, rows 0,1).
        # Local C indices: 0,1 (for col 6, rows 0,1).
        
        # Product 1: a11 * b11 -> c11
        u = [1, 0, 0, 0]
        v = [0]*14
        v[0*7 + 6] = 1
        w = [0]*14
        w[6*2 + 0] = 1
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Product 2: a12 * b21 -> c11
        u = [0, 1, 0, 0]
        v = [0]*14
        v[1*7 + 6] = 1
        w = [0]*14
        w[6*2 + 0] = 1
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Product 3: a21 * b11 -> c21
        u = [0, 0, 1, 0]
        v = [0]*14
        v[0*7 + 6] = 1
        w = [0]*14
        w[6*2 + 1] = 1
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Product 4: a22 * b21 -> c21
        u = [0, 0, 0, 1]
        v = [0]*14
        v[1*7 + 6] = 1
        w = [0]*14
        w[6*2 + 1] = 1
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        return U_list, V_list, W_list
        
    U_227, V_227, W_227 = build_227_block()
    
    # Now combine two blocks.
    # Block 0: A_left, B_top.
    # Block 1: A_right, B_bottom.
    
    rank = 50
    U = [[0]*rank for _ in range(8)] # 2*4
    V = [[0]*rank for _ in range(28)] # 4*7
    W = [[0]*rank for _ in range(14)] # 7*2
    
    # Block 0
    for t in range(25):
        u = U_227[t]
        v = V_227[t]
        w = W_227[t]
        
        # Map U to global A.
        # Local u indices 0..3 correspond to i in 0,1 and j' in 0,1.
        # Global A index = i*4 + j'.
        for idx, val in enumerate(u):
            if val != 0:
                i = idx // 2
                j_prime = idx % 2
                U[i*4 + j_prime][t] = val
                
        # Map V to global B.
        # Local v indices 0..13 correspond to j' in 0,1 and k in 0..6.
        # Global B index = j'*7 + k. (B_top)
        for idx, val in enumerate(v):
            if val != 0:
                V[idx][t] = val
                
        # Map W to global C.
        # Local w indices 0..13 correspond to k in 0..6 and i in 0,1.
        # Global C index = k*2 + i.
        for idx, val in enumerate(w):
            if val != 0:
                W[idx][t] = val
                
    # Block 1
    for t in range(25):
        u = U_227[t]
        v = V_227[t]
        w = W_227[t]
        
        # Map U to global A.
        # Local u indices 0..3 correspond to i in 0,1 and j' in 0,1.
        # Global A index = i*4 + (j'+2).
        for idx, val in enumerate(u):
            if val != 0:
                i = idx // 2
                j_prime = idx % 2
                U[i*4 + j_prime + 2][25 + t] = val
                
        # Map V to global B.
        # Local v indices 0..13 correspond to j' in 0,1 and k in 0..6.
        # Global B index = (j'+2)*7 + k. (B_bottom)
        for idx, val in enumerate(v):
            if val != 0:
                j_prime = idx // 7
                k = idx % 7
                V[(j_prime+2)*7 + k][25 + t] = val
                
        # Map W to global C.
        # Local w indices 0..13 correspond to k in 0..6 and i in 0,1.
        # Global C index = k*2 + i.
        for idx, val in enumerate(w):
            if val != 0:
                W[idx][25 + t] = val
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
