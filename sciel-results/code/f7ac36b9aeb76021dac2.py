"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Strategy: Use the 2x2 block partition of the 4x4 multiplication tensor.
The 4x4 matrix multiplication can be viewed as multiplying two 2x2 block matrices
where each block is 2x2. This gives a 2x2x2 tensor of 2x2 matrices.

The key insight: The rank of 2x2x2 matrix multiplication is 7 (Strassen).
If we treat the 4x4 multiplication as a 2x2 block multiplication where each
"element" is a 2x2 matrix, we can use Strassen's algorithm at the block level.

However, this only works if the block interactions are independent. For a general
4x4 matrix multiplication, the blocks are NOT independent - there are cross terms.

A better approach: The 4x4 matrix multiplication tensor has a known rank.
The best known upper bound for 4x4 matrix multiplication rank is 49 (from 
applying Strassen recursively or via block methods).

Actually, let me reconsider. The standard result:
- 2x2 matrix multiplication: rank 7 (Strassen)
- 4x4 matrix multiplication: rank <= 49 (by applying Strassen to 2x2 blocks of 2x2)

But we need to be careful about the indexing convention. The output W follows
W[k*n+i, r] where n=m=p=4.

Let me construct the rank-49 decomposition using the block Strassen approach.

For 4x4 multiplication, partition into 2x2 blocks:
A = [[A11, A12], [A21, A22]] where each Aij is 2x2
B = [[B11, B12], [B21, B22]] where each Bij is 2x2

C = A*B = [[C11, C12], [C21, C22]] where each Cij is 2x2

Using Strassen at the block level:
M1 = (A11 + A22)(B11 + B22)
M2 = (A21 + A22)B11
M3 = A11(B12 - B22)
M4 = A22(B21 - B11)
M5 = (A11 + A12)B22
M6 = (A21 - A11)(B11 + B12)
M7 = (A12 - A22)(B21 + B22)

C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6

Each Mi is a product of two 2x2 matrices, which itself has rank 7.
So total rank = 7 * 7 = 49.

Now I need to map this to the U, V, W matrices with the correct indexing.

U has shape (n*m, rank) = (16, 49)
V has shape (m*p, rank) = (16, 49)  
W has shape (p*n, rank) = (16, 49)

The tensor identity is:
sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = (A*B)[i,k]

Wait, let me re-read the convention. The task says:
"U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"

So for the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
This gives sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = (A*B)[i,k]

Actually the standard tensor decomposition for matrix multiplication T_{i,j,k} means:
(A*B)[i,k] = sum_j A[i,j]*B[j,k]

So the tensor is T[i,j,k] = delta where the contraction is over j.

In the factorization: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Wait, that doesn't look right dimensionally. Let me re-read.

"U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"

So U is indexed by (i*m+j) which ranges over 0..15 for n=m=4
V is indexed by (j*p+k) which ranges over 0..15 for m=p=4
W is indexed by (k*n+i) which ranges over 0..15 for p=n=4

The product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal (A*B)[i,k]

For this to work as a tensor decomposition of matrix multiplication, we need:
(A*B)[i,k] = sum_j A[i,j] B[j,k]

So the factorization must satisfy:
sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} in some sense...

No wait. The tensor for matrix multiplication is:
T[i,j,k] such that (A*B)[i,k] = sum_j A[i,j] T[i,j,k] B[j,k]... 

Actually no. The standard way: matrix multiplication is a bilinear map.
C = A*B means C[i,k] = sum_j A[i,j] * B[j,k]

As a trilinear form or tensor: the multiplication tensor M has entries M[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].

More precisely, the tensor T of matrix multiplication satisfies:
T[i,j,k] = 1 (it's the identity in the sense that C[i,k] = sum_j A[i,j] B[j,k])

The rank decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]... 

Hmm, I think the indexing in the problem is:
U has rows indexed by (i,j) pairs -> row i*m+j
V has rows indexed by (j,k) pairs -> row j*p+k  
W has rows indexed by (k,i) pairs -> row k*n+i

And the identity is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta where...

Actually I think the correct interpretation is that this decomposes the matrix multiplication tensor where:
(A*B)[i,k] = sum_j sum_r A[i,j] * B[j,k] * (something)

No. Let me think again. In tensor decomposition of matrix multiplication:
The multiplication tensor T is a 3-tensor with T[i,j,k] representing the coefficient.
For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (it's just the structure).

The rank-r decomposition means:
T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[i,k,r]... 

But the indices don't match. Let me look at the schoolbook example more carefully.

Schoolbook: for each (i,j,k), we have one term with U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.

So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

For schoolbook, this equals 1 for all i,j,k (since each (i,j,k) has exactly one c where all three are 1).

And indeed (A*B)[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_c A[i,j] * B[j,k] * (U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])

Wait that's not right either because we'd be summing over c too.

I think the correct interpretation is:
The matrix multiplication tensor is defined such that:
(A*B)[i,k] = sum_j A[i,j] * B[j,k]

And the tensor T[i,j,k] = 1 (all ones) represents this.

The rank decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

This is a rank-1 decomposition of the all-ones tensor, which has rank 1... but that's not right because the schoolbook uses rank 64.

I'm confused. Let me just verify with the schoolbook: it has rank 64 = 4*4*4, and for each (i,j,k) triple, there's exactly one column c where U, V, W all have 1. So sum_r ... = 1 for each (i,j,k). That checks out.

So I need to find U, V, W of rank r such that for all i in [0,4), j in [0,4), k in [0,4):
sum_{r=0}^{r-1} U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1

Let me implement the rank-49 Strassen-based decomposition.

Actually, I realize I should just implement a known correct decomposition. Let me use the recursive Strassen approach properly.

For 4x4, I'll use the fact that it can be decomposed using 7 Strassen multiplications at the 2x2 block level, each of which is a 2x2x2 multiplication of rank 7.

Let me code this up carefully.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS
    
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # 1. Define the 2x2 Strassen decomposition
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # S1 = (P00+P11)(Q00+Q11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # S2 = (P10+P11)Q00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1
    
    # S3 = P00(Q01-Q11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1
    
    # S4 = P11(Q10-Q00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # S5 = (P00+P01)Q11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # S6 = (P10-P00)(Q00+Q01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # S7 = (P01-P11)(Q10+Q11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1
    
    # 2. Define the 4x4 Block Strassen decomposition
    # The 4x4 matrices are partitioned into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # We perform 7 block multiplications M_t = P_t * Q_t
    # Each M_t is a 2x2 matrix. We decompose M_t using the 2x2 Strassen decomposition above.
    
    # block_terms[t] = (P_blocks, Q_blocks, R_blocks)
    # P_blocks: list of (block_index, coeff) where block_index is 0..3 for (0,0), (0,1), (1,0), (1,1)
    # block_index mapping: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    
    block_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # M2 = (A10+A11)B00 -> C10(+1)
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1) ]),
        # M3 = A00(B01-B11) -> C01(+1), C11(+1)
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        # M4 = A11(B10-B00) -> C00(+1), C10(+1)
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),
        # M5 = (A00+A01)B11 -> C00(-1), C01(+1)
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
        # M6 = (A10-A00)(B00+B01) -> C11(+1)
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
        # M7 = (A01-A11)(B10+B11) -> C00(+1), C11(-1)
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (3, -1) ]),
    ]
    
    # Helper to get the 4 rows of U/V/W corresponding to a 2x2 block
    # Block index b: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # For U (A matrix): rows are i*4+j.
    # Block (br, bc) covers i in [2*br, 2*br+2), j in [2*bc, 2*bc+2)
    # Row indices:
    # (2*br)*4 + (2*bc), (2*br)*4 + (2*bc+1)
    # (2*br+1)*4 + (2*bc), (2*br+1)*4 + (2*bc+1)
    
    def get_block_rows_U(b):
        br = b // 2
        bc = b % 2
        return [
            (2*br)*4 + (2*bc),
            (2*br)*4 + (2*bc+1),
            (2*br+1)*4 + (2*bc),
            (2*br+1)*4 + (2*bc+1)
        ]
    
    def get_block_rows_V(b):
        # V is indexed by j*4+k.
        # Block (br, bc) of B corresponds to j in [2*br, 2*br+2), k in [2*bc, 2*bc+2)
        # Note: In B, the "row" index is j and "col" is k.
        # So block (br, bc) in B means j is in block row br, k is in block col bc.
        br = b // 2
        bc = b % 2
        return [
            (2*br)*4 + (2*bc),
            (2*br)*4 + (2*bc+1),
            (2*br+1)*4 + (2*bc),
            (2*br+1)*4 + (2*bc+1)
        ]
    
    def get_block_rows_W(b):
        # W is indexed by k*4+i.
        # Block (br, bc) of C corresponds to i in [2*br, 2*br+2), k in [2*bc, 2*bc+2)
        # So block (br, bc) in C means k is in block col bc, i is in block row br.
        # Wait, W index is k*n + i.
        # C[i, k]. Block (br, bc) means i in [2*br, ...], k in [2*bc, ...]
        # So k is the "row" index in the W flat array? No, W[k*4+i].
        # So for a fixed block (br, bc) of C:
        # i ranges 2*br, 2*br+1
        # k ranges 2*bc, 2*bc+1
        # The rows in W are k*4 + i.
        br = b // 2
        bc = b % 2
        return [
            (2*bc)*4 + (2*br),
            (2*bc)*4 + (2*br+1),
            (2*bc+1)*4 + (2*br),
            (2*bc+1)*4 + (2*br+1)
        ]
    
    # Construct U, V, W
    for t in range(7):
        P_blocks, Q_blocks, R_blocks = block_terms[t]
        
        # Create coefficient vectors for P, Q, R blocks
        # p_coeff[b] = sum of coeffs for block b in P_blocks
        p_coeff = [0, 0, 0, 0]
        for b, c in P_blocks:
            p_coeff[b] += c
        q_coeff = [0, 0, 0, 0]
        for b, c in Q_blocks:
            q_coeff[b] += c
        r_coeff = [0, 0, 0, 0]
        for b, c in R_blocks:
            r_coeff[b] += c
            
        for s in range(7):
            col = t * 7 + s
            
            # For each of the 4 elements in the 2x2 P matrix (indices 0..3 in U2)
            # U2[pe, s] is the coefficient for P element pe.
            # P element pe corresponds to block (pe//2, pe%2) in the 2x2 block structure.
            # However, U2 is for a 2x2 matrix P.
            # The "P" in the block Strassen is a 2x2 matrix where each element is a 2x2 block of A.
            # So P[0,0] = sum of A blocks with coeff from P_blocks where block is (0,0).
            # P[0,1] = sum of A blocks with coeff from P_blocks where block is (0,1).
            # etc.
            
            # U2[pe, s] multiplies P[pe//2, pe%2].
            # P[pe//2, pe%2] is the linear combination of A blocks at that position.
            # Specifically, P[pe//2, pe%2] = p_coeff[pe] * (the block A[pe]) ?
            # No. P is a 2x2 matrix of blocks.
            # P[0,0] is the block A[0,0] scaled by p_coeff[0].
            # But wait, P is formed by linear combination of A's blocks.
            # If P_blocks = [(0, 1), (3, 1)], then P[0,0] = A[0,0] and P[1,1] = A[1,1].
            # So P[pe//2, pe%2] is the block A[pe] scaled by p_coeff[pe].
            
            # The 2x2 Strassen decomposition says:
            # sum_s U2[pe, s] * P[pe] * ... 
            # Actually, the term is U2[pe, s] * V2[qe, s] * W2[re, s] * P[pe] * Q[qe] * R[re]
            # But P[pe] is not a scalar, it's a 2x2 block of A.
            # So we need to distribute U2[pe, s] * p_coeff[pe] to the 4 rows of A corresponding to block pe.
            
            # Let's iterate over the 4 "positions" in the 2x2 P matrix.
            for pe in range(4):
                coeff_p = U2[pe, s]
                if coeff_p == 0:
                    continue
                
                # The block index in A is pe (0,1,2,3)
                # The coefficient for this block in P is p_coeff[pe]
                final_coeff = coeff_p * p_coeff[pe]
                if final_coeff == 0:
                    continue
                
                rows = get_block_rows_U(pe)
                for r in rows:
                    U[r, col] += final_coeff
            
            for qe in range(4):
                coeff_q = V2[qe, s]
                if coeff_q == 0:
                    continue
                
                final_coeff = coeff_q * q_coeff[qe]
                if final_coeff == 0:
                    continue
                
                rows = get_block_rows_V(qe)
                for r in rows:
                    V[r, col] += final_coeff
            
            for re in range(4):
                coeff_r = W2[re, s]
                if coeff_r == 0:
                    continue
                
                final_coeff = coeff_r * r_coeff[re]
                if final_coeff == 0:
                    continue
                
                rows = get_block_rows_W(re)
                for r in rows:
                    W[r, col] += final_coeff

    # Verify
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U[i*4+j, :] * V[j*4+k, :] * W[k*4+i, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    
    if max_err < 1e-10:
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                        mat[i][j] = round(mat[i][j])
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank_fb = 64
        U_fb = [[0] * rank_fb for _ in range(16)]
        V_fb = [[0] * rank_fb for _ in range(16)]
        W_fb = [[0] * rank_fb for _ in range(16)]
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j][column] = 1
                    V_fb[j * 4 + k][column] = 1
                    W_fb[k * 4 + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
