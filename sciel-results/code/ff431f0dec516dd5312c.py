"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock limit
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    # We want U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is the standard triple
    # i.e., the decomposition of the matrix multiplication tensor.
    
    # Strategy: Use Kronecker product of Strassen (2x2x2, rank 7) with schoolbook (2x4x4, rank 8)
    # This gives a 4x8x8 multiplication of rank 56.
    # But we need 4x4x8.
    # 
    # Alternative: Split the 8 columns into two groups of 4.
    # C[:, 0:4] = A * B[:, 0:4]
    # C[:, 4:8] = A * B[:, 4:8]
    # Each is a 4x4 * 4x4 multiplication.
    # If we use rank-49 Strassen-based decomposition for each, total rank = 98.
    #
    # Let's construct a rank-49 decomposition for 4x4 * 4x4.
    # We can do this by Kronecker product of Strassen (2x2x2, rank 7) with schoolbook (2x2x2, rank 8)?
    # No, that would be 4x4 * 4x4 with rank 7*8 = 56? Let's check.
    #
    # Actually, the tensor for 4x4 * 4x4 can be seen as:
    # C[i,j] = sum_{a,b} A[i,a] * B[a,j]
    # If we write A and B as 2x2 blocks of 2x2 blocks, and C similarly:
    # C_{pq} = sum_{rs} A_{pr} * B_{rs} (block multiplication)
    # Each block multiplication is a 2x2 * 2x2 multiplication.
    # We can use Strassen for each block multiplication? No, that's not how it works.
    #
    # Better approach: Use the Kronecker product structure.
    # The 4x4*4x4 multiplication tensor is the Kronecker product of two 2x2*2x2 tensors.
    # So if T2 is the 2x2*2x2 tensor (rank 8 schoolbook, or rank 7 Strassen),
    # then T4 = T2 ⊗ T2.
    # The rank of T4 is at most rank(T2)^2.
    # If T2 has rank 7 (Strassen), then T4 has rank at most 49.
    #
    # Let's verify: T2 has indices (i,j,k) for i,j,k in {0,1}.
    # T2[i,j,k] = delta_{i,k} * delta_{j,?} ... no.
    # The matrix multiplication tensor for 2x2 * 2x2:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So the tensor T[i,j,k] = 1 if we're computing C[i,j] from A[i,k] and B[k,j].
    # In the Kronecker product T4 = T2 ⊗ T2:
    # T4[i1,i2, j1,j2, k1,k2] = T2[i1,j1,k1] * T2[i2,j2,k2]
    # Wait, the indices need to be consistent.
    #
    # Let's define it more carefully.
    # For n x m * m x p, the tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 (this is the standard tensor where C[i,k] = sum_j A[i,j] * B[j,k])
    # Wait, the problem says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank), and W[k*n+i, r] is the coefficient for C[i,k] from the r-th rank-1 term.
    # U has shape (n*m, rank), U[i*m+j, r] is the coefficient for A[i,j].
    # V has shape (m*p, rank), V[j*p+k, r] is the coefficient for B[j,k].
    #
    # So the tensor is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k?} no.
    # Actually, the matrix multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the "tensor" is T[i,j,k] = 1 for all i,j,k? No.
    # The rank-1 decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where (u_r)_{i,j} = U[i*m+j, r], (v_r)_{j,k} = V[j*p+k, r], (w_r)_{k,i} = W[k*n+i, r]
    # And the condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if we're "selecting" this (i,j,k)
    # Actually, the condition is that for each (i,k), C[i,k] = sum_j A[i,j] B[j,k]
    # In the decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) * (sum_? ... )
    # No, the standard tensor decomposition for matrix multiplication is:
    # C = A * B means C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C = sum_r (U_r^T A) ⊙ (V_r^T B) where ⊙ is some operation.
    # 
    # Let me re-read the problem statement carefully.
    # "U, V, W factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]" - so W is indexed by (k,i) where k is the column of B / column of C, i is the row of A / row of C.
    # 
    # The decomposition should satisfy:
    # For all i, j, k: the "tensor entry" T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] should be 1 if this corresponds to a valid term in C[i,k] = sum_j A[i,j] B[j,k].
    # 
    # Actually, I think the condition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In the decomposition, each rank-1 term contributes:
    # (sum_j U[i*m+j, r] A[i,j]) * (sum_k' V[j'*p+k', r] B[j',k']) * ... no.
    #
    # Let me think about this differently. The standard way to express matrix multiplication as a tensor:
    # M_{i,j,k} = 1 if we multiply A[i,j] and B[j,k] to contribute to C[i,k].
    # So M[i,j,k] = 1 for all valid i,j,k.
    # The rank of this tensor is the minimum number of rank-1 tensors needed to sum to M.
    # A rank-1 tensor is u ⊗ v ⊗ w where u has indices (i,j), v has indices (j,k), w has indices (k,i).
    # So M[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    # 
    # For the schoolbook decomposition:
    # For each (i,j,k): U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    # And all other entries are 0. This gives rank = n*m*p.
    # 
    # Now, for the Kronecker product approach:
    # If M2 is the 2x2*2x2 multiplication tensor (indices i,j,k in {0,1}),
    # then M4 = M2 ⊗ M2 for 4x4*4x4.
    # M4[i1,i2, j1,j2, k1,k2] = M2[i1,j1,k1] * M2[i2,j2,k2]
    # Wait, the indices don't quite work out this way.
    #
    # Let's use a different approach. For 4x4 * 4x4:
    # A is 4x4, B is 4x4, C is 4x4.
    # We can write A as a 2x2 block matrix of 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Similarly B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each of these is a sum of two 2x2*2x2 multiplications.
    # If we use Strassen for each pair, we can reduce the rank.
    #
    # Actually, the cleanest approach is:
    # For 4x4 * 4x4, use the Kronecker product of two Strassen decompositions.
    # Strassen for 2x2*2x2 has rank 7.
    # So 4x4*4x4 has rank at most 7*7 = 49.
    #
    # Let me implement this.
    # First, define the Strassen decomposition for 2x2*2x2.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a(e+h)-b(g+h)+(a+b)h, a(f+h)-d(f+h)+(a+d)f],
    #      [c(e+h)-(c+d)h+dh, c(g+h)-(c+d)g+dg]]
    # Strassen's 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)h
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)e
    # p6 = (c+d)(f-h)
    # p7 = (a-b)(g+h)
    # C00 = p1 - p2 + p5 + p7
    # C01 = p3 + p4
    # C10 = p2 + p6
    # C11 = p1 - p4 + p6 - p7
    #
    # So for each product p_r, we have:
    # p1: A factors: a, d; B factors: e, h
    # p2: A factors: c, d; B factors: h
    # p3: A factors: a; B factors: f, -h
    # p4: A factors: d; B factors: g, -e
    # p5: A factors: a, b; B factors: e
    # p6: A factors: c, d; B factors: f, -h
    # p7: A factors: a, -b; B factors: g, h
    #
    # Now for the Kronecker product:
    # 4x4 * 4x4 can be seen as (2x2 blocks of 2x2) * (2x2 blocks of 2x2)
    # Let A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # Similarly for B and C.
    # The block multiplication is: C_ab = sum_cd A_ac * B_cd where a,b,c,d in {0,1}
    # This is exactly a 2x2 * 2x2 multiplication where the "elements" are 2x2 matrices.
    # 
    # If we use Strassen for the block level (rank 7), and for each block-level product,
    # we need to multiply two 2x2 matrices, which itself can be done with Strassen (rank 7).
    # So total rank = 7 * 7 = 49.
    #
    # Let's implement this.
    #
    # For the block-level Strassen, we have 7 products.
    # For each block-level product, say p_r_block = A_block_alpha * B_block_beta,
    # this is a product of two 2x2 matrices.
    # We decompose this using Strassen for 2x2*2x2, giving 7 sub-products.
    # So total rank = 7 * 7 = 49.
    
    # Let's define the Strassen factors for 2x2*2x2.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*2+j, r] is the coefficient for A[i,j]
    # V[j*2+k, r] is the coefficient for B[j,k]
    # W[k*2+i, r] is the coefficient for C[i,k]
    
    # Strassen products for 2x2*2x2:
    # r=0: p1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C00+=1, C11+=1
    # r=1: p2 = (c+d)h: U: c=1,d=1; V: h=1; W: C00-=1, C10+=1
    # r=2: p3 = a(f-h): U: a=1; V: f=1, h=-1; W: C01+=1
    # r=3: p4 = d(g-e): U: d=1; V: g=1, e=-1; W: C01+=1, C11-=1
    # r=4: p5 = (a+b)e: U: a=1,b=1; V: e=1; W: C00+=1
    # r=5: p6 = (c+d)(f-h): U: c=1,d=1; V: f=1, h=-1; W: C10+=1, C11+=1
    # r=6: p7 = (a-b)(g+h): U: a=1,b=-1; V: g=1, h=1; W: C00+=1, C11-=1
    
    strassen_U = [
        [1, 0, 0, 0, 1, 0, 1],  # a
        [0, 0, 0, 0, 1, 0, -1], # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 1, 0]   # d
    ]
    strassen_V = [
        [1, 0, 1, -1, 1, 1, 1], # e
        [0, 0, 0, 0, 0, 0, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 1, -1, 0, 0, -1, 1]  # h
    ]
    strassen_W = [
        [1, -1, 0, 0, 1, 0, 1],  # C00
        [0, 0, 1, 1, 0, 0, 0],   # C01
        [0, 1, 0, 0, 0, 1, 0],   # C10
        [1, 0, 0, -1, 0, 1, -1]  # C11
    ]
    
    # Now, for 4x4 * 4x4:
    # We have block indices (a,b) for A blocks, (c,d) for B blocks, (a,d) for C blocks.
    # Each block is 2x2.
    # The block multiplication is: C_ab = sum_cd A_ac * B_cd
    # Using Strassen at the block level:
    # Let's define the block-level Strassen.
    # A_blocks: A00, A01, A10, A11
    # B_blocks: B00, B01, B10, B11
    # C_blocks: C00, C01, C10, C11
    #
    # Block-level Strassen products:
    # q1 = (A00 + A11)(B00 + B11)
    # q2 = (A10 + A11) * B11
    # q3 = A00 * (B01 - B11)
    # q4 = A11 * (B10 - B00)
    # q5 = (A00 + A01) * B00
    # q6 = (A10 + A11) * (B01 - B11)
    # q7 = (A00 - A01) * (B10 + B11)
    #
    # C00 = q1 - q2 + q5 + q7
    # C01 = q3 + q4
    # C10 = q2 + q6
    # C11 = q1 - q4 + q6 - q7
    #
    # Each q_r is a product of two 2x2 matrices (or sums thereof).
    # q1 = (A00 + A11)(B00 + B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This is a sum of 4 products of 2x2 matrices.
    # Each product of 2x2 matrices can be decomposed using Strassen (rank 7).
    # So q1 has rank 4*7 = 28? That doesn't seem right.
    #
    # Wait, I think I'm overcomplicating this.
    # The Kronecker product of two tensors:
    # If T2 is the 2x2*2x2 multiplication tensor (rank 7 with Strassen),
    # then T4 = T2 ⊗ T2 is the 4x4*4x4 multiplication tensor.
    # The rank of T4 is at most rank(T2) * rank(T2) = 49.
    #
    # The Kronecker product of two rank-1 decompositions:
    # T2 = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
    # T2' = sum_{s=1}^{7} u'_s ⊗ v'_s ⊗ w'_s
    # T4 = T2 ⊗ T2' = sum_{r,s} (u_r ⊗ u'_s) ⊗ (v_r ⊗ v'_s) ⊗ (w_r ⊗ w'_s)
    # So T4 has rank at most 49.
    
    # Let's construct the 4x4*4x4 decomposition using the Kronecker product.
    # U4[i*4+j, (r,s)] = strassen_U[i, r] * strassen_U[j, s]
    # V4[j*4+k, (r,s)] = strassen_V[j, r] * strassen_V[k, s]
    # W4[k*4+i, (r,s)] = strassen_W[k, r] * strassen_W[i, s]
    # Wait, the indices need to be consistent.
    #
    # For 2x2*2x2:
    # U2[i*2+j, r] where i,j in {0,1}
    # V2[j*2+k, r] where j,k in {0,1}
    # W2[k*2+i, r] where k,i in {0,1}
    #
    # For 4x4*4x4:
    # U4[i*4+j, (r,s)] where i,j in {0,1,2,3}
    # V4[j*4+k, (r,s)] where j,k in {0,1,2,3}
    # W4[k*4+i, (r,s)] where k,i in {0,1,2,3}
    #
    # The Kronecker product:
    # U4[i*4+j, (r,s)] = U2[i%2, r] * U2[i//2, s] ? No.
    #
    # Let's think of the 4x4 matrix as a 2x2 block matrix of 2x2 blocks.
    # A[i,j] where i = i1*2 + i0, j = j1*2 + j0
    # A is indexed by (i1,i0,j1,j0)
    # The block A_{i1,j1} is the 2x2 matrix with entries A[i1*2+i0, j1*2+j0]
    #
    # In the Kronecker product T4 = T2 ⊗ T2:
    # T4[i,j,k] = T2[i1,j1,k1] * T2[i0,j0,k0]
    # where i = i1*2 + i0, j = j1*2 + j0, k = k1*2 + k0
    #
    # So for the rank-1 decomposition:
    # U4[i*4+j, (r,s)] = U2[i1*2+j1, r] * U2[i0*2+j0, s]
    # V4[j*4+k, (r,s)] = V2[j1*2+k1, r] * V2[j0*2+k0, s]
    # W4[k*4+i, (r,s)] = W2[k1*2+i1, r] * W2[k0*2+i0, s]
    
    # Let's verify with the schoolbook case first.
    # For schoolbook 2x2*2x2:
    # U2[i*2+j, (i,j,k)] = 1 if the column index corresponds to (i,j,k)
    # This is getting complicated. Let's just implement the Kronecker product directly.
    
    # Define strassen_U, strassen_V, strassen_W as numpy arrays.
    sU = np.array(strassen_U, dtype=float)  # shape (4, 7)
    sV = np.array(strassen_V, dtype=float)  # shape (4, 7)
    sW = np.array(strassen_W, dtype=float)  # shape (4, 7)
    
    # Construct 4x4*4x4 decomposition.
    rank44 = 7 * 7
    U44 = np.zeros((16, rank44))
    V44 = np.zeros((16, rank44))
    W44 = np.zeros((16, rank44))
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            U44[i * 4 + j, col] = sU[i1 * 2 + j1, r] * sU[i0 * 2 + j0, s]
                            for k1 in range(2):
                                for k0 in range(2):
                                    k = k1 * 2 + k0
                                    V44[j * 4 + k, col] = sV[j1 * 2 + k1, r] * sV[j0 * 2 + k0, s]
                                    W44[k * 4 + i, col] = sW[k1 * 2 + i1, r] * sW[k0 * 2 + i0, s]
    
    # Now, for 4x4 * 4x8:
    # We can split B into two 4x4 matrices: B1 = B[:, 0:4], B2 = B[:, 4:8]
    # C[:, 0:4] = A * B1
    # C[:, 4:8] = A * B2
    # Each is a 4x4 * 4x4 multiplication.
    # So we can use the same U44 for both, and V44 for B1 and B2 separately.
    
    # For C[:, 0:4]:
    # U = U44
    # V[j*4+k, r] for k in 0:4
    # W[k*4+i, r] for k in 0:4
    
    # For C[:, 4:8]:
    # U = U44
    # V[j*4+(k+4), r] for k in 0:4
    # W[(k+4)*4+i, r] for k in 0:4
    
    # Total rank = 49 + 49 = 98.
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((16, 32))  # m*p = 4*8 = 32
    W = np.zeros((32, rank))  # p*n = 8*4 = 32
    
    # First half: C[:, 0:4]
    U[:, 0:49] = U44
    V[:, 0:4] = V44
    W[0:16, 0:49] = W44
    
    # Second half: C[:, 4:8]
    U[:, 49:98] = U44
    V[:, 4:8] = V44
    W[16:32, 49:98] = W44
    
    # Convert to lists.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
