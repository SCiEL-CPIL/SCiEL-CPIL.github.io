import numpy as np
import time

def construct():
    n = 4
    m = 4
    p = 6
    
    # We will build U, V, W as lists of columns (rank-1 tensors)
    # then stack them.
    # U: (16, rank)
    # V: (24, rank)
    # W: (24, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Part 1: 4x4x4 multiplication using Strassen blocks
    # Strassen 2x2x2 decomposition
    # A' = [[a00, a01], [a10, a11]]
    # B' = [[b00, b01], [b10, b11]]
    # C' = [[c00, c01], [c10, c11]]
    
    # The 7 Strassen terms for 2x2x2
    # Each term is (U_vec, V_vec, W_vec)
    # U_vec is coeffs for A' elements (flattened 0:00, 1:01, 2:10, 3:11)
    # V_vec is coeffs for B' elements
    # W_vec is coeffs for C' elements
    
    strassen_terms = [
        # 1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        ( [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1] ),
        # 2: (a10+a11)b00 -> c10 += 1, c11 += 1
        ( [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1] ),
        # 3: a00(b10-b11) -> c01 += 1, c11 += 1
        ( [1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 1] ),
        # 4: a11(b01-b00) -> c00 += 1, c10 += 1
        ( [0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, 0] ),
        # 5: (a00+a01)b11 -> c00 += -1, c01 += 1
        ( [1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0] ),
        # 6: (a10-a00)(b00+b01) -> c11 += 1
        ( [1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1] ),
        # 7: (a01-a11)(b10+b11) -> c00 += 1, c11 += -1
        ( [0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1] ),
    ]
    
    # We need to map the 2x2 block indices to global 4x4 indices.
    # Block (I, J) of A corresponds to rows 2I..2I+1, cols 2J..2J+1.
    # Global index in U for A[i,j] is i*m + j = i*4 + j.
    
    # For each of the 7 "super" Strassen multiplications M_s
    # M_s = A_s * B_s
    # A_s is a 2x2 matrix of blocks.
    # B_s is a 2x2 matrix of blocks.
    
    # The coefficients for A_s and B_s are from the 2x2 Strassen formulas applied to blocks.
    # A_s[I,J] = sum of A_blocks[I', J'] with coeffs.
    
    # Let's define the block-level Strassen coefficients.
    # A_s is formed by:
    # A_s[0,0] = A[0,0] + A[1,1]  (for M1)
    # etc.
    
    # Actually, it's easier to iterate over the 7 block-level terms.
    # For each block-level term s (1..7):
    #   We have linear combos of A-blocks and B-blocks.
    #   Let A_block_coeffs[I, J] be the coefficient of A[I,J] in A_s.
    #   Let B_block_coeffs[J, K] be the coefficient of B[J,K] in B_s.
    
    # The product M_s is a 2x2 matrix C_s where:
    # C_s[I, K] = sum_{J} A_s[I, J] * B_s[J, K]
    # But A_s and B_s are themselves linear combos of blocks.
    # So M_s is a linear combo of block products A[I,J] * B[J,K].
    # And each block product A[I,J] * B[J,K] is a 2x2x2 tensor.
    # We decompose each such block product using the 7 inner Strassen terms.
    
    # So, for each block-level term s:
    #   For each I, J, K:
    #      coeff = A_block_coeffs[I, J] * B_block_coeffs[J, K]
    #      If coeff != 0:
    #          We need to add the tensor for A[I,J] * B[J,K] scaled by coeff.
    #          But wait, the block product A[I,J] * B[J,K] contributes to C[I,K].
    #          The tensor for A[I,J] * B[J,K] is:
    #          sum_{i', j', k'} A[2I+i', 2J+j'] * B[2J+j', 2K+k'] * C[2I+i', 2K+k']
    #          This is exactly the 2x2x2 tensor shifted.
    #          We can decompose this shifted 2x2x2 tensor using the 7 inner Strassen terms.
    
    # So the total rank is 7 (block terms) * 4*4*4 (block positions? No) 
    # Wait. The block-level Strassen uses 7 terms.
    # Each term s involves specific blocks.
    # For a fixed s, A_s is a specific linear combo of 4 A-blocks.
    # B_s is a specific linear combo of 4 B-blocks.
    # The product A_s * B_s is a 2x2 matrix of block products.
    # So for a fixed s, we have 4 block products (for each I,K in C_s).
    # But the block products are not independent; they share the same A_s and B_s.
    # However, the tensor decomposition is linear.
    # T = sum_s (A_s * B_s) * ReconstructionCoeffs
    # A_s * B_s is a 2x2 matrix of tensors (each 2x2x2).
    # So for each s, we have 4 sub-tensors (corresponding to C_s[0,0], C_s[0,1], C_s[1,0], C_s[1,1]).
    # Each sub-tensor is a 2x2x2 tensor.
    # We decompose each 2x2x2 tensor into 7 rank-1 tensors.
    # So for each s, we have 4 * 7 = 28 rank-1 tensors.
    # Total for 7 s's: 7 * 28 = 196?
    # No, this is not optimal.
    
    # The standard Strassen for 4x4x4 uses 7 multiplications of 2x2x2.
    # Each multiplication of 2x2x2 has rank 7.
    # So total rank is 7 * 7 = 49.
    # How?
    # The "multiplication" is the operation.
    # The tensor is sum of 7 terms.
    # Each term is a "block multiplication".
    # A block multiplication is a 2x2x2 tensor.
    # The rank of a 2x2x2 tensor is 7.
    # So the rank of the sum is at most 7 * 7 = 49.
    
    # So we need to generate 49 rank-1 tensors.
    
    # Let's define the block-level coefficients.
    # For s in 1..7:
    #   A_s_coeffs[I, J]
    #   B_s_coeffs[J, K]
    #   C_s_coeffs[I, K] (from reconstruction)
    
    # Actually, the reconstruction gives C as sum of M_s * C_s_coeffs.
    # M_s = A_s * B_s.
    # A_s is a 2x2 matrix. B_s is a 2x2 matrix.
    # M_s[I, K] = sum_J A_s[I, J] B_s[J, K].
    # This M_s[I, K] is a 2x2 matrix (block).
    # So the term in the tensor is:
    # sum_{I, K} C_s_coeffs[I, K] * M_s[I, K]
    # M_s[I, K] is the product of block A_s[I, :] and B_s[:, K].
    # A_s[I, :] is a row of blocks. B_s[:, K] is a column of blocks.
    # This is a sum of block products.
    # A_s[I, J] * B_s[J, K].
    # A_s[I, J] is a linear combo of A-blocks.
    # B_s[J, K] is a linear combo of B-blocks.
    
    # So for each s, and each I, J, K:
    #   coeff = C_s_coeffs[I, K] * A_s_coeffs[I, J] * B_s_coeffs[J, K]
    #   If coeff != 0:
    #       We add the tensor for A-block[I,J] * B-block[J,K] scaled by coeff.
    #       This tensor is a 2x2x2 tensor (shifted).
    #       We decompose this 2x2x2 tensor into 7 rank-1 tensors.
    #       So we add 7 rank-1 tensors for this (s, I, J, K).
    
    # How many (s, I, J, K) tuples?
    # s: 7
    # I, J, K: 2, 2, 2 -> 8
    # Total 56 tuples.
    # Each adds 7 rank-1 tensors.
    # Total 56 * 7 = 392?
    # This is much higher than 49.
    
    # The error is in the decomposition logic.
    # The rank of the sum of tensors is not the sum of ranks if they can be combined.
    # But here, the tensors are distinct.
    # However, the standard result R(4,4,4) <= 49 implies a smarter decomposition.
    # The 49 comes from: 7 multiplications of 2x2x2.
    # Each 2x2x2 multiplication is a *single* tensor of rank 7.
    # So the total tensor is a sum of 7 tensors, each of rank 7.
    # Total rank 49.
    
    # So we should treat each "block multiplication" as a single 2x2x2 tensor.
    # The block multiplication M_s is defined by A_s and B_s.
    # A_s is a 2x2 matrix of *scalars* (coeffs for blocks).
    # No, A_s is a 2x2 matrix where each entry is a *block*?
    # No, in Strassen, A and B are 2x2 matrices of *scalars* (or blocks).
    # If we view the 4x4 matrix as a 2x2 matrix of 2x2 blocks,
    # then Strassen applies to the *blocks*.
    # The "multiplication" is of 2x2 block matrices.
    # The rank of the tensor for multiplying 2x2 block matrices (where blocks are scalars) is 7.
    # But the blocks are 2x2 matrices.
    # So the tensor is a "tensor of tensors".
    # The rank of the total tensor is the rank of the outer tensor times the rank of the inner tensor?
    # Yes, if the inner tensor is the same for all outer indices.
    # Here, the inner tensor is the standard 2x2x2 multiplication tensor.
    # The outer tensor is the Strassen 2x2x2 tensor.
    # So the total rank is Rank(Outer) * Rank(Inner) = 7 * 7 = 49.
    
    # So we need to generate 49 rank-1 tensors.
    # Each rank-1 tensor is formed by:
    #   Outer term t (1..7)
    #   Inner term u (1..7)
    #   The U vector is the product of the U vector of the outer term (mapped to blocks) and the U vector of the inner term (mapped to elements within blocks).
    
    # Let's implement this.
    
    # Outer Strassen terms (for blocks)
    # A_outer[I, J] coeffs
    # B_outer[J, K] coeffs
    # C_outer[I, K] coeffs
    
    # Inner Strassen terms (for elements within blocks)
    # A_inner[i', j'] coeffs
    # B_inner[j', k'] coeffs
    # C_inner[i', k'] coeffs
    
    # Global U index: (2I + i', 2J + j') -> (2I + i')*4 + (2J + j')
    # Global V index: (2J + j', 2K + k') -> (2J + j')*6 + (2K + k')
    # Global W index: (2K + k', 2I + i') -> (2K + k')*4 + (2I + i')
    
    # We iterate over outer term t and inner term u.
    # For each pair (t, u), we form a rank-1 tensor.
    
    # Outer term t:
    #   U_outer[I, J]
    #   V_outer[J, K]
    #   W_outer[I, K]
    
    # Inner term u:
    #   U_inner[i', j']
    #   V_inner[j', k']
    #   W_inner[i', k']
    
    # The combined U vector has non-zero entries at:
    #   (2I + i', 2J + j') with value U_outer[I, J] * U_inner[i', j']
    #   for all I, J, i', j' where coeffs are non-zero.
    
    # We can build the U, V, W vectors for each (t, u) pair.
    
    U_outer = []
    V_outer = []
    W_outer = []
    
    # 1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U_outer.append( [[1, 0], [0, 1]] ) # I,J
    V_outer.append( [[1, 0], [1, 0]] ) # J,K
    W_outer.append( [[1, 0], [0, 1]] ) # I,K
    
    # 2: (A10+A11)B00 -> C10+=1, C11+=1
    U_outer.append( [[0, 0], [1, 1]] )
    V_outer.append( [[1, 0], [0, 0]] )
    W_outer.append( [[0, 0], [1, 1]] )
    
    # 3: A00(B10-B11) -> C01+=1, C11+=1
    U_outer.append( [[1, 0], [0, 0]] )
    V_outer.append( [[0, -1], [1, 0]] )
    W_outer.append( [[0, 1], [0, 1]] )
    
    # 4: A11(B01-B00) -> C00+=1, C10+=1
    U_outer.append( [[0, 0], [0, 1]] )
    V_outer.append( [[1, -1], [0, 0]] )
    W_outer.append( [[1, 0], [1, 0]] )
    
    # 5: (A00+A01)B11 -> C00+=-1, C01+=1
    U_outer.append( [[1, 1], [0, 0]] )
    V_outer.append( [[0, 0], [0, 1]] )
    W_outer.append( [[-1, 1], [0, 0]] )
    
    # 6: (A10-A00)(B00+B01) -> C11+=1
    U_outer.append( [[1, 0], [-1, 0]] )
    V_outer.append( [[1, 1], [0, 0]] )
    W_outer.append( [[0, 0], [0, 1]] )
    
    # 7: (A01-A11)(B10+B11) -> C00+=1, C11+=-1
    U_outer.append( [[0, 1], [0, -1]] )
    V_outer.append( [[0, 0], [1, 1]] )
    W_outer.append( [[1, 0], [0, -1]] )
    
    # Inner Strassen terms (standard 2x2x2)
    # Same as strassen_terms above, but flattened.
    
    for t in range(7):
        for u in range(7):
            U_vec = np.zeros(16)
            V_vec = np.zeros(24)
            W_vec = np.zeros(24)
            
            # Unpack outer
            Uo = U_outer[t]
            Vo = V_outer[t]
            Wo = W_outer[t]
            
            # Unpack inner
            Ui = strassen_terms[u][0]
            Vi = strassen_terms[u][1]
            Wi = strassen_terms[u][2]
            
            # Combine
            for I in range(2):
                for J in range(2):
                    if Uo[I][J] != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                if Ui[i_prime*2 + j_prime] != 0:
                                    i_global = 2*I + i_prime
                                    j_global = 2*J + j_prime
                                    val = Uo[I][J] * Ui[i_prime*2 + j_prime]
                                    U_vec[i_global*4 + j_global] = val
                                    
            for J in range(2):
                for K in range(2):
                    if Vo[J][K] != 0:
                        for j_prime in range(2):
                            for k_prime in range(2):
                                if Vi[j_prime*2 + k_prime] != 0:
                                    j_global = 2*J + j_prime
                                    k_global = 2*K + k_prime
                                    val = Vo[J][K] * Vi[j_prime*2 + k_prime]
                                    V_vec[j_global*6 + k_global] = val
                                    
            for I in range(2):
                for K in range(2):
                    if Wo[I][K] != 0:
                        for i_prime in range(2):
                            for k_prime in range(2):
                                if Wi[i_prime*2 + k_prime] != 0:
                                    i_global = 2*I + i_prime
                                    k_global = 2*K + k_prime
                                    val = Wo[I][K] * Wi[i_prime*2 + k_prime]
                                    W_vec[k_global*4 + i_global] = val
                                    
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)
            
    # Part 2: 4x4x2 schoolbook
    # k in 4, 5
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U_vec = np.zeros(16)
                V_vec = np.zeros(24)
                W_vec = np.zeros(24)
                
                U_vec[i*4 + j] = 1
                V_vec[j*6 + k] = 1
                W_vec[k*4 + i] = 1
                
                U_cols.append(U_vec)
                V_cols.append(V_vec)
                W_cols.append(W_vec)
                
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {'U': U, 'V': V, 'W': W}

# Verify
if __name__ == '__main__':
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Check decomposition
    # T = sum U[:, r] * V[:, r] * W[:, r]
    # We can check a random instance.
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 6)
    C = A @ B
    
    U_flat = U.T.reshape(rank, 16)
    V_flat = V.T.reshape(rank, 24)
    W_flat = W.T.reshape(rank, 24)
    
    # Reconstruct C
    C_recon = np.zeros((4, 6))
    for r in range(rank):
        u = U_flat[r].reshape(4, 4)
        v = V_flat[r].reshape(4, 6)
        w = W_flat[r].reshape(4, 6)
        # The tensor is sum u[i,j] v[j,k] w[k,i]?
        # No, the decomposition is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r U[i,j, r] V[j,k, r] W[k,i, r] ??
        # The task says W[k*n+i, r].
        # So W index is k*4+i.
        # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # So C[i,k] = sum_j sum_r U[i*4+j, r] V[j*6+k, r] W[k*4+i, r].
        
        # Let's verify this sum.
        
    C_check = np.zeros((4, 6))
    for r in range(rank):
        u_col = U[:, r].reshape(4, 4)
        v_col = V[:, r].reshape(4, 6)
        w_col = W[:, r].reshape(6, 4) # k*4+i -> k is row, i is col?
        # W index k*4+i. If we reshape (24, 1) to (6, 4), index k*4+i is [k, i].
        # So w_col[k, i].
        
        # C[i, k] = sum_j u_col[i, j] * v_col[j, k] * w_col[k, i]
        
        for i in range(4):
            for k in range(6):
                s = 0
                for j in range(4):
                    s += u_col[i, j] * v_col[j, k] * w_col[k, i]
                C_check[i, k] += s
                
    print("Max error:", np.max(np.abs(C - C_check)))
