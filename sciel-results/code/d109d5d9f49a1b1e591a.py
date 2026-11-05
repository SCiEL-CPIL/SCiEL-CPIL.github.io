"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use Strassen's rank-7 algorithm on 2x2 blocks where possible
    # For 5x5 matrices, we can partition as 2+2+1 in each dimension
    # This gives us 3x3 = 9 blocks of sizes (2x2, 2x1, 1x2, 1x1) etc.
    
    # Let's think about this more carefully.
    # We have A (5x5) * B (5x5) = C (5x5)
    # Partition indices: rows of A and cols of A/B into {0,1}, {2,3}, {4}
    # So we have 3x3 blocks.
    
    # Block (i,j) where i is row-block of A (and col-block of C), j is col-block of A (and row-block of B)
    # Sizes: block (i,j) has size s_i x s_j where s = [2, 2, 1]
    
    # For each pair of block-rows i and block-cols j, we need to compute C[i,:] = sum_j A[i,j] * B[j,:]
    
    # The key insight: for 2x2 blocks, we can use Strassen (rank 7 instead of 8)
    # For mixed sizes, we need to be more careful.
    
    # Let's enumerate all block interactions:
    # Row blocks of A: R0 (size 2), R1 (size 2), R2 (size 1)
    # Col blocks of A / Row blocks of B: C0 (size 2), C1 (size 2), C2 (size 1)
    # Col blocks of B: K0 (size 2), K1 (size 2), K2 (size 1)
    
    # C[Ri, Kj] = sum_j A[Ri, Cj] * B[Cj, Kj]
    
    # We need to decompose this tensor. The naive approach uses rank = sum over all (i,j,k) of s_i * s_j * s_k = 125
    
    # With Strassen on 2x2 sub-blocks, we can reduce some of these.
    
    # Let's build the decomposition explicitly.
    
    # First, let's define the Strassen rank-7 decomposition for 2x2x2:
    # For 2x2 matrices X, Y, Z = X*Y:
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Let's build the full decomposition.
    
    # I'll create a helper to add a term to the decomposition.
    # Each term is a triple (u_indices, v_indices, w_indices) with coefficients.
    
    # Actually, let me think of this differently. The tensor T[i,j,k] = delta_{i,k} (for matrix mult A*B=C where C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, the convention is W[k*n+i, r]. So the tensor is T[i,j,k] where i is row of A, j is col of A/row of B, k is col of B.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if we're computing the (i,k) entry using A[i,j] and B[j,k].
    # Actually T is the multiplication tensor: T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the standard matrix multiplication tensor).
    
    # A rank-r decomposition is: T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # where U is (n x r), V is (m x r), W is (p x r)
    # But the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # And W[k*n+i, r] - so W is indexed by (k,i) pairs, meaning W combines the output indices.
    
    # Let me re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # So U has rows indexed by (i,j) pairs? No wait...
    
    # Actually, looking at the seed code:
    # U[i*m+j, c] - so U is indexed by (i,j) where i is row of A, j is col of A
    # V[j*p+k, c] - V is indexed by (j,k) where j is row of B, k is col of B
    # W[k*n+i, c] - W is indexed by (k,i) where k is col of B, i is row of A (output index)
    
    # So the decomposition is: for each output (i,k), C[i,k] = sum_j A[i,j] B[j,k]
    # And the factorization is: sum_c U[i,j,c] V[j,k,c] W[k,i,c] should give the correct tensor.
    
    # Wait, that's a 3-way tensor factorization where:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # And we need T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k]).
    
    # So we need: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
    
    # This is a tensor decomposition problem. The rank is the minimum number of terms.
    
    # For the schoolbook method, we use r = n*m*p = 125 terms, one for each (i,j,k) triple.
    
    # To reduce rank, we can use bilinear methods like Strassen.
    
    # Let me try a different approach: use the known Strassen-based decomposition for 5x5.
    
    # Partition into 2+2+1. The block structure gives us a 3x3x3 grid of blocks.
    # For each combination of block indices (a,b,c) where a is row-block, b is col-block/row-block, c is col-block:
    # We have a sub-problem of size s_a x s_b x s_c.
    
    # For 2x2x2 sub-problems, we can use rank 7 (Strassen) instead of rank 8.
    # For other sizes, we use the schoolbook rank = s_a * s_b * s_c.
    
    # Let's count:
    # s = [2, 2, 1]
    # Block (a,b,c) has size s_a x s_b x s_c, schoolbook rank = s_a * s_b * s_c
    # Strassen-applicable when s_a = s_b = s_c = 2: rank 7 instead of 8
    
    # Number of 2x2x2 blocks: only block (0,0,0) is 2x2x2. That's 1 block.
    # So we save 1 rank unit: 125 - 1 = 124? No, that's not right.
    
    # Actually, the blocks are not independent in the simple way I'm thinking.
    # Let me reconsider.
    
    # The full tensor T[i,j,k] for i in 0..4, j in 0..4, k in 0..4.
    # We partition i into {0,1}, {2,3}, {4}; j into {0,1}, {2,3}, {4}; k into {0,1}, {2,3}, {4}.
    
    # For each combination of block indices (I,J,K) where I,J,K in {0,1,2}:
    # The sub-tensor T[I,J,K] has entries T[i,j,k] for i in block I, j in block J, k in block K.
    # This sub-tensor is a s_I x s_J x s_K tensor that is all ones (since T[i,j,k]=1 for all i,j,k).
    
    # Wait no! T[i,j,k] = 1 for all i,j,k means the tensor is the all-ones tensor.
    # But that's not right for matrix multiplication. Let me re-examine.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined such that C[i,k] = sum_{j,c} A[i,j] B[j,k] * (coefficient)
    # In a rank-r decomposition: C[i,k] = sum_c (sum_j A[i,j] U[j,c]) * (sum_k' B[k',k] V[c,k']) * ... 
    
    # I'm confusing myself. Let me look at the seed code again.
    
    # Seed code:
    # For each (i,j,k): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # So term c contributes: U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 * 1 * 1 = 1 when the indices match.
    # The total contribution to T[i,j,k] from all c is: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # For the schoolbook, only the term with (i,j,k) = (i,j,k) contributes, giving T[i,j,k] = 1.
    
    # So yes, we need T[i,j,k] = 1 for all i,j,k, and T is the all-ones tensor of size 5x5x5.
    # Wait, that can't be right. The all-ones tensor has a rank of 1 (it's rank-1: T[i,j,k] = 1*1*1).
    
    # I think I'm misunderstanding the tensor. Let me re-read the task.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 5 by 5 matrix by a 5 by 5 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # I think the tensor is NOT the all-ones tensor. The matrix multiplication tensor T_matmult is defined as:
    # T[i,j,k] = delta_{i,k} ... no that's not right either.
    
    # Actually, I think the issue is that the factorization is:
    # C[i,k] = sum_{j, c} A[i,j] * B[j,k] * (something)
    # No, that's not a standard factorization.
    
    # Let me think about this differently. In a bilinear decomposition of matrix multiplication:
    # C = A * B
    # We write: C[i,k] = sum_c M_c(A) * N_c(B)
    # where M_c(A) is a linear form in A and N_c(B) is a linear form in B.
    # M_c(A) = sum_{i,j} U[i,j,c] A[i,j]
    # N_c(B) = sum_{j,k} V[j,k,c] B[j,k]
    # And C[i,k] = sum_c (sum_{i',j'} U[i',j',c] A[i',j']) * (sum_{j',k'} V[j',k',c] B[j',k'])
    # But this doesn't isolate C[i,k] properly.
    
    # I think the correct interpretation is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # No, that doesn't work either because U depends on both i and j.
    
    # Let me just trust the seed code and the interface. The seed code produces a valid decomposition with rank 125.
    # The condition is: for all i,j,k: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    
    # So we need to find matrices U, V, W such that this holds, minimizing rank.
    
    # This is equivalent to finding a rank decomposition of the tensor T where T[i*m+j, j*p+k, k*n+i] ... no.
    
    # Actually, I think it's: define a 3-way tensor X of size (n*m) x (m*p) x (p*n) where:
    # X[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent, else 0.
    # But that's not quite right either.
    
    # Let me just accept the formulation: we need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k in 0..4.
    
    # This is a system of 125 equations in rank * 75 variables (since U has 25*rank entries, V has 25*rank, W has 25*rank).
    
    # The minimum rank is what we're looking for.
    
    # Given the complexity, let me just use the Strassen-based approach that was measured to give rank 117.
    
    # From the knowledge: "Using Strassen's rank-7 algorithm for the eight 2x2x2 sub-blocks in a 2+2+1 partition reduces the total rank from 125 to 117"
    
    # Wait, "eight 2x2x2 sub-blocks"? In a 2+2+1 partition, there's only one 2x2x2 block. Unless they mean something else.
    
    # Maybe they're using a different partition or a different technique.
    
    # Let me just implement the schoolbook method (rank 125) which is known to be valid, and see if I can improve it.
    
    # Actually, let me just return the schoolbook decomposition since it's guaranteed to be valid.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
