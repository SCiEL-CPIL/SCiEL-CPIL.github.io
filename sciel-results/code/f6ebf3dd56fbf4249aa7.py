import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 140

    # Helper to verify a decomposition
    def verify(U, V, W):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        # Check tensor identity: sum_c U[i,j,c] V[j,k,c] W[k,i,c] == delta(i,j,k) 
        # Wait, the tensor for matrix mult is T[i,j,k] = 1 if we are computing C[i,k] = sum_j A[i,j]B[j,k]
        # The tensor T is in C^{n x m} x C^{m x p} x C^{p x n}
        # T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
        # Actually, the tensor T_{ijk} = 1 if we are looking at the term A_{ij}B_{jk} contributing to C_{ik}.
        # The decomposition is T = sum_c u_c v_c w_c
        # where u_c is a vector in C^{n*m}, v_c in C^{m*p}, w_c in C^{p*n}.
        # The index for W in the task is W[k*n + i, r]. This corresponds to the third mode index being k*n+i?
        # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
        # Usually, the third mode is indexed by (output_row, output_col) or similar.
        # For C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor T has components T_{i,j,k}.
        # The factorization is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}? No.
        # Standard tensor decomposition for matmul:
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} is not quite right because the indices must match the tensor modes.
        # Mode 1: (i,j) -> index i*m + j
        # Mode 2: (j,k) -> index j*p + k
        # Mode 3: (k,i) -> index k*n + i (This matches the W index given: W[k*n+i, r])
        # So, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
        # Wait, is it 1? Yes, the tensor for matrix multiplication has 1s at the "valid" positions (i,j,k) corresponding to the terms in the sum.
        # Specifically, T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # Because every term A_{ij}B_{jk} appears exactly once in the sum for C_{ik}.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Strategy:
    # We can decompose the 2x4 x 4x7 multiplication into blocks.
    # A is 2x4. We can view it as two 2x2 blocks horizontally: A = [A1, A2]
    # B is 4x7. We can view it as two 2x7 blocks vertically? No, B is 4x7.
    # Let's split the inner dimension m=4 into two parts of size 2.
    # A = [A1 | A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x7.
    # Then C = A1 B1 + A2 B2.
    # This is a sum of two 2x2 x 2x7 multiplications? No.
    # A1 is 2x2, B1 is 2x7. Product is 2x7.
    # A2 is 2x2, B2 is 2x7. Product is 2x7.
    # So the problem reduces to two instances of multiplying a 2x2 matrix by a 2x7 matrix.
    # Let's analyze the rank of 2x2 x 2x7.
    # We can further split the 2x7 into seven 2x1 column vectors?
    # Or split 2x7 into 2x2 + 2x2 + 2x1?
    # 
    # Let's look at the rank of 2x2 x 2x7.
    # We can use Strassen for the 2x2 part?
    # Strassen computes 2x2 * 2x2 with rank 7.
    # But here B is 2x7.
    # We can view B as [B1 | B2 | B3 | B4 | B5 | B6 | B7] where each Bk is 2x1.
    # Then C = sum_{k=1}^7 A * Bk.
    # This doesn't help directly because A is the same.
    # 
    # Alternative:
    # Decompose 2x2 x 2x7 into 7 instances of 2x2 x 2x1?
    # Rank of 2x2 x 2x1:
    # C = A B. A is 2x2, B is 2x1.
    # C1 = A11 B1 + A21 B2
    # C2 = A12 B1 + A22 B2
    # This is a bilinear map.
    # We can use the standard 4-term decomposition:
    # 1. A11 B1
    # 2. A21 B2
    # 3. A12 B1
    # 4. A22 B2
    # Rank 4.
    # Is there a lower rank?
    # For 1x2 x 2x1 (scalar = v^T w), rank is 2.
    # For 2x2 x 2x1, the tensor is T_{i,j,k} where i is row of C (1,2), j is col of A (1,2), k is col of B (1,2).
    # Actually, let's just use the schoolbook for the sub-blocks if we can't find better.
    # 
    # Let's try to construct a decomposition for 2x2 x 2x7 using Strassen-like ideas.
    # We can compute A1 * B1 and A2 * B2.
    # A1 * B1 is 2x2 * 2x7.
    # We can split B1 into B1a (2x2) and B1b (2x5)?
    # This seems complicated.
    # 
    # Let's try a simpler structural decomposition.
    # We know Strassen 2x2 x 2x2 has rank 7.
    # We can compute A1 * B1 where B1 is 2x7 by splitting B1 into 3 blocks: 2x2, 2x2, 2x3.
    # This is getting messy.
    # 
    # Let's try a direct construction for 2x4 x 4x7 using known ranks.
    # Rank of 2x4 x 4x7 is at most rank(2x2 x 2x7) + rank(2x2 x 2x7).
    # If we can find a low rank for 2x2 x 2x7, we are good.
    # 
    # Let's compute rank of 2x2 x 2x7.
    # We can view this as 7 instances of 2x2 x 2x1? No, A is shared.
    # 
    # Let's use the fact that we can use Strassen for the 2x2 parts.
    # Let's split the 4x7 matrix B into 7 columns of size 4x1.
    # Then C[:, k] = A * B[:, k].
    # A is 2x4. B[:, k] is 4x1.
    # This is 2x4 x 4x1 multiplication.
    # We can split A into two 2x2 blocks horizontally.
    # A = [A1 | A2]. B[:, k] = [b1; b2] where b1, b2 are 2x1.
    # C[:, k] = A1 b1 + A2 b2.
    # This is a sum of two 2x2 x 2x1 products.
    # Rank of 2x2 x 2x1 is 4 (schoolbook).
    # So each column of C requires rank 8?
    # Total rank 7 * 8 = 56. This is schoolbook.
    # 
    # Can we do better for 2x2 x 2x1?
    # C1 = A11 B1 + A21 B2
    # C2 = A12 B1 + A A22 B2
    # This is equivalent to multiplying a 2x2 matrix by a 2x1 vector.
    # This is a bilinear map from (A, b) to c.
    # The tensor T has dimensions 4 (for A) x 2 (for b) x 2 (for c).
    # T_{i,j,k} where i is index in A (1..4), j is index in b (1..2), k is index in c (1..2).
    # T_{1,1,1}=1, T_{2,2,1}=1, T_{3,1,2}=1, T_{4,2,2}=1.
    # This tensor is the Kronecker product of two 1x1x1 tensors? No.
    # It's just 4 independent terms.
    # Is the rank 4?
    # If we try rank 3:
    # We need 3 vectors u, v, w such that sum u_i v_j w_k = T_{ijk}.
    # This is equivalent to finding a rank-3 decomposition of a 4x2x2 tensor with 4 non-zeros.
    # It is known that the rank of the "matrix multiplication" tensor <1,2,1> is 4?
    # Actually, <1,2,1> is just a vector dot product with a matrix?
    # No, it's C = A B where A is 2x2 and B is 2x1.
    # This is not standard matmul.
    # 
    # Let's check the rank of <2,2,1>.
    # The tensor T is in C^{4} x C^{2} x C^{2}.
    # It has 4 non-zero entries.
    # If the entries were in general position, rank would be 4.
    # Here they are at (1,1,1), (2,2,1), (3,1,2), (4,2,2).
    # These are disjoint in the first index.
    # So the rank is exactly 4.
    # 
    # So, if we treat each column of B independently, we get rank 4 per column of A-block?
    # Wait, A is 2x4. We split into A1 (2x2) and A2 (2x2).
    # For each column k of B (size 4x1), we have b1 (2x1) and b2 (2x1).
    # C_k = A1 b1 + A2 b2.
    # This requires rank 4 for A1 b1 and rank 4 for A2 b2.
    # Total rank per column k is 8.
    # 7 columns -> 56.
    # 
    # Is there a way to share factors across columns?
    # The factors U, V, W are global.
    # U is 2x4 x rank.
    # V is 4x7 x rank.
    # W is 7x2 x rank.
    # 
    # Let's try to construct a rank-50 solution by finding a valid decomposition for a subset of the tensor and extending?
    # No, it must be exact.
    # 
    # Let's try to use the Strassen algorithm for 2x2 x 2x2 and extend it.
    # We can compute A1 * B1 (where B1 is 2x2 part of B) using Strassen (rank 7).
    # We have 3 such 2x2 blocks in B if we group columns?
    # B is 4x7.
    # Split B into B1 (4x2), B2 (4x2), B3 (4x2), B4 (4x1).
    # A1 * B1 (2x2 * 2x2) -> Rank 7 (Strassen)
    # A2 * B2 (2x2 * 2x2) -> Rank 7
    # A3 * B3 (2x2 * 2x2) -> Rank 7
    # A4 * B4 (2x2 * 2x1) -> Rank 4
    # Total rank = 7 + 7 + 7 + 4 = 25?
    # Wait, A is 2x4. A1, A2 are 2x2.
    # B is 4x7.
    # B1 is top 2 rows, first 2 cols? No.
    # A = [A1 | A2].
    # B = [B1; B2] where B1 is 2x7, B2 is 2x7.
    # C = A1 B1 + A2 B2.
    # A1 is 2x2. B1 is 2x7.
    # We can split B1 into B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1).
    # Then A1 B1 = A1 B1a + A1 B1b + A1 B1c + A1 B1d.
    # A1 B1a is 2x2 * 2x2. Rank 7.
    # A1 B1b is 2x2 * 2x2. Rank 7.
    # A1 B1c is 2x2 * 2x2. Rank 7.
    # A1 B1d is 2x2 * 2x1. Rank 4.
    # Total for A1 B1 is 25.
    # Similarly for A2 B2 is 25.
    # Total rank 50.
    # 
    # This is a valid decomposition with rank 50!
    # We can implement this.
    
    # Helper to get Strassen 2x2 x 2x2 decomposition
    def get_strassen_222():
        # Returns U, V, W for 2x2 * 2x2
        # U is 4x7, V is 4x7, W is 4x7
        # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen formulas for C = AB
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-d)(f+g)
        # M7 = (b-d)(e+g)
        
        # Term 1: (a+d)(e+h)
        # a is A[0,0] (idx 0), d is A[1,1] (idx 3)
        # e is B[0,0] (idx 0), h is B[1,1] (idx 3)
        # Contribution to C:
        # C[0,0] += M1
        # C[1,1] += M1
        # C[0,1] -= M1
        # C[1,0] -= M1
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
        
        # Term 2: (c+d)e
        # c is A[1,0] (idx 2), d is A[1,1] (idx 3)
        # e is B[0,0] (idx 0)
        # Contribution:
        # C[1,0] += M2
        # C[1,1] += M2
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # Term 3: a(f-h)
        # a is A[0,0] (idx 0)
        # f is B[0,1] (idx 1), h is B[1,1] (idx 3)
        # Contribution:
        # C[0,1] += M3
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # Term 4: d(g-e)
        # d is A[1,1] (idx 3)
        # g is B[1,0] (idx 2), e is B[0,0] (idx 0)
        # Contribution:
        # C[1,1] += M4
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # Term 5: (a+b)h
        # a is A[0,0] (idx 0), b is A[0,1] (idx 1)
        # h is B[1,1] (idx 3)
        # Contribution:
        # C[0,1] += M5
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # Term 6: (c-d)(f+g)
        # c is A[1,0] (idx 2), d is A[1,1] (idx 3)
        # f is B[0,1] (idx 1), g is B[1,0] (idx 2)
        # Contribution:
        # C[1,0] += M6
        
        U[2, 5] = 1; U[3, 5] = -1
        V[1, 5] = 1; V[2, 5] = 1
        W[2, 5] = 1
        
        # Term 7: (b-d)(e+g)
        # b is A[0,1] (idx 1), d is A[1,1] (idx 3)
        # e is B[0,0] (idx 0), g is B[1,0] (idx 2)
        # Contribution:
        # C[0,0] += M7
        
        U[1, 6] = 1; U[3, 6] = -1
        V[0, 6] = 1; V[2, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Helper to get Schoolbook 2x2 x 2x1 decomposition
    def get_school_221():
        # Returns U, V, W for 2x2 * 2x1
        # U is 4x4, V is 2x4, W is 2x4
        # A is 2x2, B is 2x1, C is 2x1
        # C[0] = A[0,0]B[0] + A[0,1]B[1]
        # C[1] = A[1,0]B[0] + A[1,1]B[1]
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # Term 1: A[0,0]B[0] -> C[0]
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 2: A[0,1]B[1] -> C[0]
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # Term 3: A[1,0]B[0] -> C[1]
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # Term 4: A[1,1]B[1] -> C[1]
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Combine decompositions
    def combine_tensors(decomps):
        # decomps is a list of (U, V, W, i_offset, j_offset, k_offset)
        # We need to place them into the global U, V, W
        # Global U shape: (n*m, rank) = (8, rank)
        # Global V shape: (m*p, rank) = (28, rank)
        # Global W shape: (p*n, rank) = (14, rank)
        
        total_rank = sum(d[0].shape[1] for d in decomps)
        U_global = np.zeros((8, total_rank))
        V_global = np.zeros((28, total_rank))
        W_global = np.zeros((14, total_rank))
        
        rank_offset = 0
        for U_d, V_d, W_d, i_off, j_off, k_off in decomps:
            r_d = U_d.shape[1]
            # Map local indices to global
            # Local U index: i_local * m_local + j_local
            # Global U index: (i_off + i_local) * m + (j_off + j_local) ?
            # Wait, the local decomposition is for a sub-problem.
            # The sub-problem is A_sub * B_sub = C_sub.
            # A_sub is a submatrix of A.
            # B_sub is a submatrix of B.
            # C_sub is a submatrix of C.
            
            # In our case:
            # A is 2x4. Split into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
            # B is 4x7. Split into B1 (rows 0-1, all cols) and B2 (rows 2-3, all cols).
            # Then C = A1 B1 + A2 B2.
            # A1 B1 is 2x2 * 2x7.
            # We split B1 into B1a (2x2, cols 0-1), B1b (2x2, cols 2-3), B1c (2x2, cols 4-5), B1d (2x1, col 6).
            # So we have 4 sub-problems for A1:
            # 1. A1 * B1a (2x2 * 2x2)
            # 2. A1 * B1b (2x2 * 2x2)
            # 3. A1 * B1c (2x2 * 2x2)
            # 4. A1 * B1d (2x2 * 2x1)
            # And 4 sub-problems for A2:
            # 5. A2 * B2a (2x2 * 2x2)
            # 6. A2 * B2b (2x2 * 2x2)
            # 7. A2 * B2c (2x2 * 2x2)
            # 8. A2 * B2d (2x2 * 2x1)
            
            # For each sub-problem, we have local U, V, W.
            # We need to map the local indices to the global indices.
            
            # Local A index: i_l in [0,2), j_l in [0,2) (for 2x2) or [0,1) (for 2x1)
            # Global A index: i_g = i_off + i_l, j_g = j_off + j_l
            # Local B index: j_l in [0,2) (rows), k_l in [0,2) (cols)
            # Global B index: j_g = j_off + j_l, k_g = k_off + k_l
            # Local C index: k_l in [0,2) (rows), i_l in [0,2) (cols)
            # Global C index: k_g = k_off + k_l, i_g = i_off + i_l
            
            # Note: The W index in global is k*n + i.
            # Local W index is k_l * n_local + i_l.
            
            for r in range(r_d):
                for i_l in range(U_d.shape[0] // 2): # This is wrong, U_d shape is (n_local*m_local, rank)
                    pass
            
            # Let's do it element by element
            n_l, m_l, p_l = U_d.shape[0]//U_d.shape[1] if False else (2, 2, 2) # Hardcoded for now, need to handle 2x1
            
            # Actually, let's just iterate over the non-zero elements of U_d, V_d, W_d
            # Or better, construct the mapping matrices.
            
            # For U_d:
            # U_d[u_idx, r]
            # u_idx = i_l * m_l + j_l
            # Global u_idx_g = (i_off + i_l) * m + (j_off + j_l)
            for u_idx in range(U_d.shape[0]):
                i_l = u_idx // m_l
                j_l = u_idx % m_l
                u_idx_g = (i_off + i_l) * m + (j_off + j_l)
                U_global[u_idx_g, rank_offset:rank_offset+r_d] += U_d[u_idx, :]
                
            # For V_d:
            # V_d[v_idx, r]
            # v_idx = j_l * p_l + k_l
            # Global v_idx_g = (j_off + j_l) * p + (k_off + k_l)
            for v_idx in range(V_d.shape[0]):
                j_l = v_idx // p_l
                k_l = v_idx % p_l
                v_idx_g = (j_off + j_l) * p + (k_off + k_l)
                V_global[v_idx_g, rank_offset:rank_offset+r_d] += V_d[v_idx, :]
                
            # For W_d:
            # W_d[w_idx, r]
            # w_idx = k_l * n_l + i_l
            # Global w_idx_g = (k_off + k_l) * n + (i_off + i_l)
            for w_idx in range(W_d.shape[0]):
                k_l = w_idx // n_l
                i_l = w_idx % n_l
                w_idx_g = (k_off + k_l) * n + (i_off + i_l)
                W_global[w_idx_g, rank_offset:rank_offset+r_d] += W_d[w_idx, :]
                
            rank_offset += r_d
            
        return U_global, V_global, W_global

    # Construct the specific decompositions
    U_s, V_s, W_s = get_strassen_222()
    U_sc, V_sc, W_sc = get_school_221()
    
    decomps = []
    
    # A1 (rows 0-1, cols 0-1) * B1a (rows 0-1, cols 0-1)
    decomps.append((U_s, V_s, W_s, 0, 0, 0)) # i_off=0, j_off=0, k_off=0
    # A1 * B1b (rows 0-1, cols 2-3)
    decomps.append((U_s, V_s, W_s, 0, 0, 2)) # k_off=2
    # A1 * B1c (rows 0-1, cols 4-5)
    decomps.append((U_s, V_s, W_s, 0, 0, 4)) # k_off=4
    # A1 * B1d (rows 0-1, cols 6-6)
    # Need to adjust p_l for B1d. It is 2x1.
    # The Strassen U, V, W are for 2x2.
    # We need a 2x2 x 2x1 version?
    # No, B1d is 2x1.
    # We can use the schoolbook 2x2 x 2x1.
    # But we need to map the indices correctly.
    # In get_school_221, n_l=2, m_l=2, p_l=1.
    decomps.append((U_sc, V_sc, W_sc, 0, 0, 6)) # k_off=6
    
    # A2 (rows 0-1, cols 2-3) * B2a (rows 2-3, cols 0-1)
    # A2 is at i_off=0, j_off=2.
    # B2a is at j_off=2, k_off=0.
    # C is at i_off=0, k_off=0.
    decomps.append((U_s, V_s, W_s, 0, 2, 0))
    # A2 * B2b (rows 2-3, cols 2-3)
    decomps.append((U_s, V_s, W_s, 0, 2, 2))
    # A2 * B2c (rows 2-3, cols 4-5)
    decomps.append((U_s, V_s, W_s, 0, 2, 4))
    # A2 * B2d (rows 2-3, cols 6-6)
    decomps.append((U_sc, V_sc, W_sc, 0, 2, 6))
    
    U_final, V_final, W_final = combine_tensors(decomps)
    
    # Verify
    if verify(U_final, V_final, W_final):
        return {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
