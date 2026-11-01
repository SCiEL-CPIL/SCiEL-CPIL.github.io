"""
Matrix multiplication 2x2 by 2x7 tensor decomposition.

We use the fact that for multiplying an (n x m) matrix A by an (m x p) matrix B,
the result C[i,k] = sum_j A[i,j] * B[j,k].

The bilinear map can be decomposed as:
C[i,k] = sum_r U[i*m+j, r] ... wait, let me think about this more carefully.

Actually, the standard schoolbook decomposition has rank = n*m*p, where each term
corresponds to a specific (i,j,k) triple.

But we can do better. Let me think about the structure.

The matrix multiplication tensor T is defined such that:
C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]

The tensor T has entries T[i,j,k] = 1 for all valid (i,j,k).

Wait, that's not right either. Let me re-examine the trace convention.

The output factor W follows W[k*n+i, r]. So W has dimensions p*n (rows) by rank (columns).
U has dimensions n*m (rows) by rank (columns).
V has dimensions m*p (rows) by rank (columns).

The decomposition is: for each r, we have a rank-1 term. The product is:
C = sum_r U[:,r] * V[:,r] * W[:,r]^T ... no, that's not quite right.

Let me think of it differently. The tensor T[i,j,k] represents the coefficient in:
C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]

So T is a 3-index tensor with T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).

This is indeed the all-ones tensor of shape (n, m, p).

The rank of the all-ones tensor of shape (n,m,p) is 1, since it equals u_i * v_j * w_k 
where u, v, w are all-ones vectors.

But wait - that can't be right because the known border rank of matrix multiplication is much higher than 1.

Let me re-read the problem. The matrix multiplication tensor is NOT the all-ones tensor in the way I was thinking.

Actually, I think I was confused. The matrix multiplication bilinear map is:
C[i,k] = sum_j A[i,j] * B[j,k]

This is a specific bilinear map. The tensor associated with this bilinear map has entries:
T[i*m+j, j*p+k, k*n+i] = 1

Wait, no. Let me be more careful.

The bilinear map maps (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k].

We can write C as a vector of length n*p. The bilinear map is:
C = M(A) * B where M(A) is a block matrix... 

Actually, let me think of it as a tensor. The bilinear map f: R^{n*m} x R^{m*p} -> R^{n*p}

f(A, B)[k*n+i] = sum_j A[i*m+j] * B[j*p+k]

So the tensor T has components T[a, b, c] where:
a = i*m + j (index into A)
b = j*p + k (index into B)  
c = k*n + i (index into C)

T[a, b, c] = 1 if the indices are consistent (same j for A and B, same i for A and C, same k for B and C)

More precisely, T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k, and 0 otherwise.

Now, is this the same as the all-ones tensor? Let me check for small case n=m=p=1:
T[0, 0, 0] = 1. Yes, rank 1.

For n=1, m=2, p=1:
A is 1x2, B is 2x1, C is 1x1.
C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
T[0*2+0, 0*1+0, 0*1+0] = T[0,0,0] = 1
T[0*2+1, 1*1+0, 0*1+0] = T[1,1,0] = 1
All other entries are 0.

So T has two non-zero entries: T[0,0,0]=1 and T[1,1,0]=1.
This can be written as sum of two rank-1 tensors: e_0*e_0*e_0 + e_1*e_1*e_0.
Rank = 2 = n*m*p = 1*2*1.

For the general case, the matrix multiplication tensor has rank = n*m*p in the standard decomposition, but it can potentially be lower.

Actually, I recall now that the rank of the matrix multiplication tensor M_{n,m,p} is at most n*m*p (schoolbook) and the question is whether it can be lower.

For small cases, the exact ranks are known. For M_{2,2,2}, the rank is 7 (Laderman's algorithm).

Let me just implement the schoolbook decomposition which gives rank = n*m*p = 2*2*7 = 28, and see if I can find something better.

Actually, let me think about this differently. The bilinear map is:
C[i,k] = sum_j A[i,j] * B[j,k]

We can decompose this as a sum of rank-1 bilinear maps. A rank-1 bilinear map has the form:
C[i,k] = alpha_i * beta_k * (sum over some pattern)

Hmm, let me just go with the schoolbook approach for now, which is guaranteed to work.

For (n, m, p) = (2, 2, 7):
rank = 28
U is 4x28, V is 14x28, W is 14x28
"""

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
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
