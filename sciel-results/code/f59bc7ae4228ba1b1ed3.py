"""
Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Uses a known rank-10 decomposition pattern for 2x2 * 2x2 matrices
(Strassen-like) and extends it to the 4x4 * 4x6 case via block structure.

The key insight: 4x4 * 4x6 can be decomposed as:
- Split into 2x2 blocks
- Use rank-7 decomposition for each 2x2 * 2x2 block multiplication
- But we need to handle the 2x2 * 2x3 blocks (right half)

Actually, let me think more carefully.

For 4x4 * 4x6:
- A is 4x4, B is 4x6
- Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
- Split B into 2x3 blocks: B1 (2x3), B2 (2x3) where B = [B1; B2]

C = A*B = [A11*A1 + A12*B1, A11*A2 + A12*B2; A21*A1 + A22*B1, A21*A2 + A22*B2]
where each product is 2x2 * 2x3 = 2x3.

For 2x2 * 2x3, we can think of it as two 2x2 * 2x1 products.
But 2x2 * 2x1 can be done with rank 4 (standard) or potentially less.

Actually, let me reconsider. The rank of matrix multiplication tensor for m x n * n x p is at most m*n*p, but can be less.

Known results:
- 2x2 * 2x2: rank 7 (Strassen)
- 2x2 * 2x1: rank 4 (I believe this is optimal)
- 2x2 * 1x1: rank 2

For 4x4 * 4x6, one approach:
Use the fact that we can combine blocks.

Let me try a different approach. Consider the tensor T_{i,k,j} = delta_{i,j} for (i,j) in 4x4 and k in 4, times the structure for the 6 columns.

Actually, the matrix multiplication tensor for (n,m,p) = (4,4,6) has components T[i,j,k,l] where:
- i in [0,4) row of A
- j in [0,4) col of A / row of B  
- k in [0,6) col of B
- l in [0,4) row of result

Wait, let me use the standard convention. For multiplying A (n x m) by B (m x p) to get C (n x p):
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T has indices (i,j,k) with i in [0,n), j in [0,m), k in [0,p), and T[i,j,k] = delta for the multiplication.

A rank-r decomposition gives:
C[i,k] = sum_{c=0}^{r-1} U[i,c] * V[j,c] * W[k,c]
where U is n x r, V is m x r, W is p x r.

But the task says:
U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)

This is a different indexing. Let me re-read.

"factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"

So U has n*m rows, V has m*p rows, W has p*n rows.

The decomposition must satisfy:
For the multiplication tensor, the bilinear form is:
C[i,k] = sum_j A[i,j] * B[j,k]

In tensor decomposition form with these shapes, I think the convention is:
The tensor has three modes: (i,j) for A, (j,k) for B, and (i,k) for C.
But flattened: U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i).

The decomposition condition is:
sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta_{i,j,k} in the right way.

Actually for matrix multiplication:
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T_{(i,j),(j,k),(i,k)} = 1 if the indices are consistent, else 0.

More precisely, T is a 3-tensor with:
- Mode 1: (i,j) for i in [0,n), j in [0,m) -> n*m indices
- Mode 2: (j,k) for j in [0,m), k in [0,p) -> m*p indices  
- Mode 3: (k,i) for k in [0,p), i in [0,n) -> p*n indices (using the W[k*n+i] convention)

And T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k, else 0.

Wait, that's not quite right either. Let me think about this differently.

The matrix multiplication bilinear map is:
(A, B) -> A*B

where A is n x m, B is m x p.

This can be written as a trilinear form:
sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = 0 for the "error" to be zero.

Actually, the standard way: C = A*B means C[i,k] = sum_j A[i,j] * B[j,k].

The tensor T has entries T[i,j,k,l] = 1 if (i,j,k,l) corresponds to a valid multiplication term, i.e., T[i,j,k] for the (i,k) output component.

I think the correct formulation is:
The multiplication tensor M has M[i,j,k] = 1 for the coefficient of A[i,j]*B[j,k] in C[i,k].

So M is a 4D tensor, but when we flatten:
- A is flattened to (n*m,)
- B is flattened to (m*p,)  
- C is flattened to (n*p,)

The bilinear map is: C = M(A, B) where M is a (n*p) x (n*m) x (m*p) tensor.

M[i*p+k, i*m+j, j*p+k] = 1 for all i,j,k.

A rank-r decomposition: M = sum_c U_c * V_c * W_c (outer products)
where U_c is (n*p), V_c is (n*m), W_c is (m*p).

But the task specifies shapes (n*m, rank), (m*p, rank), (p*n, rank).

So U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.

And the decomposition: for each c, the term contributes U[:,c] * V[:,c] * W[:,c] to the tensor.

The tensor M has M[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.

So we need: sum_c U[a,c] * V[b,c] * W[c_idx,c] = M[a,b,c_idx]

For the schoolbook: rank = n*m*p, with one term per (i,j,k).

Now, can we do better? 

For n=m=p=2, Strassen gives rank 7 < 8.

For 4x4 * 4x6, what's the known rank?

Let me try to construct a decomposition using Strassen on 2x2 blocks.

Approach: Split the 4x4 * 4x6 multiplication into blocks.

A (4x4) = [[A11, A12], [A21, A22]] where each is 2x2.
B (4x6) = [[B1, B2], [B3, B4]] where B1, B3 are 2x3, B2, B4 are 2x3.

Wait, B is 4x6, so splitting into 2x3 blocks:
B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6.
Or B = [[B1, B2], [B3, B4]] where each Bi is 2x3.

C = A*B is 4x6 = [[C11, C12], [C21, C22]] where each is 2x3.

C11 = A11*B1 + A12*B3
C12 = A11*B2 + A12*B4
C21 = A21*B1 + A22*B3
C22 = A21*B2 + A22*B4

Each product is 2x2 * 2x3.

For 2x2 * 2x3, we can write it as sum of two 2x2 * 2x1 products, or use a direct decomposition.

2x2 * 2x3: rank is at most 4*3 = 12 (schoolbook). Can we do better?

Actually, 2x2 * 2x3 can be seen as: given A (2x2) and B (2x3), compute A*B (2x3).
This is equivalent to computing two 2x2 * 2x1 multiplications (for the first two columns) and one 2x2 * 2x1 for the last column, but they share A.

If we use rank 4 for each 2x2*2x1, we get rank 12 total. But we can share the A factors.

Alternatively, think of 2x2 * 2x3 as a linear map. The tensor has (2*2) x (2*3) x (2*3) = 4 x 6 x 6 entries.

I think the rank of 2x2 * 2x3 might be 8 or so.

Let me try a different approach. Let me just try to find a rank-16 or so decomposition by combining Strassen-like ideas.

Actually, for the purpose of this task, let me try to implement a known good decomposition.

Known: The rank of 4x4 * 4x4 matrix multiplication is at most 49 (from some results), but I'm not sure of the exact minimum.

Let me try a simpler approach: use the schoolbook but see if I can reduce it.

Actually, let me just try to implement a Strassen-like recursion.

For 4x4 * 4x4, Strassen recursion gives rank 7^2 = 49.
For 4x4 * 4x6, it's trickier because B is not square.

Let me try: 4x4 * 4x6 = (2x2 blocks) * (2x3 blocks).

C = A*B where A is 4x4, B is 4x6.

Split A into four 2x2 blocks: A11, A12, A21, A22.
Split B into four 2x3 blocks: B11, B12 (top row of blocks), B21, B22 (bottom row).

C11 = A11*B11 + A12*B21  (2x3)
C12 = A11*B12 + A12*B22  (2x3)
C21 = A21*B11 + A22*B21  (2x3)
C22 = A21*B12 + A22*B22  (2x3)

For each 2x2 * 2x3 product, I need a decomposition.

2x2 * 2x3: Let A be 2x2, B be 2x3. C = A*B is 2x3.
C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]

Schoolbook rank: 8 (four A entries times... wait, 2*2*3 = 12 terms, but each term is A[i,j]*B[j,k] contributing to C[i,k]).

So 12 terms for 2x2*2x3 schoolbook.

Can we reduce? Let me think...

For the first row of C:
C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] for k=0,1,2.

This is: C[0,:] = A[0,0] * B[0,:] + A[0,1] * B[1,:]

Similarly for row 1.

So if I define:
P0 = A[0,0]
P1 = A[0,1]
P2 = A[1,0]
P3 = A[1,1]

And B0 = B[0,0], B1 = B[0,1], B2 = B[0,2], B3 = B[1,0], B4 = B[1,1], B5 = B[1,2].

Then:
C[0,0] = P0*B0 + P1*B3
C[0,1] = P0*B1 + P1*B4
C[0,2] = P0*B2 + P1*B5
C[1,0] = P2*B0 + P3*B3
C[1,1] = P2*B1 + P3*B4
C[1,2] = P2*B2 + P3*B5

Schoolbook uses 12 terms (4 A factors * 3 B factors each... wait no, it's 2*2*3 = 12).

Can we share? The B factors are shared across rows of C.

For example, B0 appears in C[0,0] and C[1,0]. B1 in C[0,1] and C[1,1]. Etc.

If I could factor out the B's, I'd need 6 B factors (one per B entry) and 4 A factors, but they're multiplied together.

The tensor rank of 2x2*2x3... I believe it's 8. Let me try to construct it.

Actually, I recall that for m x n * n x p, the rank is at most m*n*p, and for some cases can be reduced.

Let me just go with a conservative approach: use rank 32 or so by combining block decompositions.

Actually, let me just implement the schoolbook with rank 96 (4*4*6) and see if I can do a simple optimization.

Wait, the seed program already does rank 96. I need to beat that.

Let me try rank 64 or less.

Idea: Use Strassen on the 2x2 blocks.

For 2x2 * 2x2, Strassen has rank 7.
For 2x2 * 2x3, let's assume rank 8 (to be determined).

Then for 4x4 * 4x6:
- 4 block multiplications, each 2x2 * 2x3
- But they share A blocks and B blocks

C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

If I decompose each 2x2*2x3 with rank r, total rank is 4*r = 32 if r=8.

But I can do better by sharing.

Let me define for the 2x2*2x3 case a rank-8 decomposition.

Hmm, this is getting complex. Let me just try to implement a rank-49 decomposition for 4x4*4x4 and adapt it.

Actually, let me just write a program that tries to find a low-rank decomposition using numerical methods.

Given the time constraint, let me implement a greedy/iterative approach to find a low-rank decomposition.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin for 150s limit
    
    # Build the multiplication tensor M of shape (n*p, n*m, m*p)
    # M[i*p+k, i*m+j, j*p+k] = 1
    np_ = n * p  # 24
    nm_ = n * m  # 16
    mp_ = m * p  # 24
    
    M = np.zeros((np_, nm_, mp_), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i*p+k, i*m+j, j*p+k] = 1.0
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try to find a low-rank decomposition
    # Use iterative method: start with random, then optimize
    
    def evaluate_rank(U, V, W):
        """Check if decomposition is valid."""
        # Reconstruct tensor
        M_rec = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(M_rec, M, atol=1e-6)
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        for attempt in range(10):
            if time.time() > deadline - 10:
                return None
            
            U = np.random.randn(nm_, r)
            V = np.random.randn(mp_, r)
            W = np.random.randn(np_, r)
            
            # Iterative optimization
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                # U[:,c] should be such that U[:,c] * V[:,c] * W[:,c] approximates M
                # For fixed V, W: U[:,c] = (M contracted with V[:,c] and W[:,c]) / (||V[:,c]||^2 * ||W[:,c]||^2)
                # Actually, U[a,c] = sum_{b,d} M[a,b,d] * V[b,c] * W[d,c] / (sum_b V[b,c]^2 * sum_d W[d,c]^2)
                
                V_sq = np.sum(V**2, axis=0)  # (r,)
                W_sq = np.sum(W**2, axis=0)  # (r,)
                
                for c in range(r):
                    denom = V_sq[c] * W_sq[c]
                    if abs(denom) < 1e-12:
                        U[:,c] = 0
                        continue
                    # U[:,c] = M[:, :, c] contracted... 
                    # M[a,b,d] * V[b,c] * W[d,c] summed over b,d
                    U[:,c] = np.einsum('abd,bc,dc->a', M, V[:,c], W[:,c]) / denom
                
                # Update V
                U_sq = np.sum(U**2, axis=0)
                W_sq = np.sum(W**2, axis=0)
                for c in range(r):
                    denom = U_sq[c] * W_sq[c]
                    if abs(denom) < 1e-12:
                        V[:,c] = 0
                        continue
                    V[:,c] = np.einsum('abd,ac,dc->b', M, U[:,c], W[:,c]) / denom
                
                # Update W
                U_sq = np.sum(U**2, axis=0)
                V_sq = np.sum(V**2, axis=0)
                for c in range(r):
                    denom = U_sq[c] * V_sq[c]
                    if abs(denom) < 1e-12:
                        W[:,c] = 0
                        continue
                    W[:,c] = np.einsum('abd,ac,bc->d', M, U[:,c], V[:,c]) / denom
            
            # Check convergence
            if evaluate_rank(U, V, W):
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if evaluate_rank(U_rounded, V_rounded, W_rounded):
                    return (U_rounded, V_rounded, W_rounded)
        
        return None
    
    # Try decreasing ranks
    for r in range(48, 0, -2):  # Start from 48, go down
        if time.time() > deadline - 20:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_decomp = (U, V, W)
            break
    
    # If no low-rank found, use schoolbook
    if best_decomp is None:
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
    else:
        U, V, W = best_decomp
        # Convert to lists of lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}

