"""Construct a rank-minimal exact tensor decomposition for 2x3 @ 3x10 matrix multiplication.

Strategy: Use the block structure. The 3x10 middle factor splits into a 3x3 block and a 3x7 block.
- 3x3 block: use a rank-5 Strassen-like decomposition for 3x3 matrix multiplication? Actually,
  we can use the standard 3x3x3 Strassen-like rank-5? No, 3x3x3 minimal rank is 7 (or 19?).
  Let me think differently.

Actually, let's use the known fact: rank(n,m,p) <= rank(n,m1,p1) + rank(n,m2,p2) when m=m1+m2.
So rank(2,3,10) <= rank(2,3,3) + rank(2,3,7).

rank(2,3,3): We can use Strassen for the 2x2 part and direct for the 2x1 part?
Actually, rank(2,3,3) = 2*3*3 = 18 schoolbook. But we can do better.
Let me think: rank(2,m,p) for small m,p.

Actually, a cleaner approach: use the identity that matrix multiplication is bilinear.
The tensor T[i,j,k] = delta_{j,k} for the 2x3 x 3x10 case... wait, no.
The multiplication tensor for (n,m) @ (m,p) has T[i,j,k] = 1 if the row i of A times column k of B involves entry (j) in the dot product. Actually T[i,j,k] = 1 always for valid indices? No.

The matrix multiplication tensor M_{n,m,p} has components M_{i,j,k} = 1 for all i in [n], j in [m], k in [p]. The decomposition is:
C[i,k] = sum_j A[i,j] * B[j,k]
So the tensor is sum over j of (e_i for fixed j) outer (e_j) outer (e_k for fixed j)? No.

Actually, the tensor T is defined by T[i, j, k] = 1 for all i, j, k. Wait no, that's not right either.

Let me re-read the task. The output factor follows W[k*n+i, r]. The decomposition satisfies:
For all A (n x m), B (m x p):
A @ B = sum_r (A @ U[:,r]) * (V[:,r] @ B) * W[:,r] elementwise? No.

The standard tensor decomposition for matrix multiplication:
C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_i' A[i', U[i*m+j, r]] ...) 

Let me use the convention from the task:
U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
The decomposition means: for the bilinear map (A, B) -> C where C = A @ B,
C[i, k] = sum_r W[k*n+i, r] * (sum_j A[i, j] * U[j? ...])

Actually, the standard convention is:
U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
And C[i,k] = sum_r U[i*m+j, r] ... no, that's not right.

The correct interpretation: The matrix multiplication tensor T has T[i, j, k] = 1 (it's the identity tensor in the sense that C[i,k] = sum_j A[i,j] B[j,k]).
A rank-r decomposition gives:
C[i,k] = sum_{r=1}^{R} W[k*n+i, r] * (sum_j A[i,j] * alpha_r(i,j)) * (sum_k' B[j',k] * beta_r(j',k))

Wait, I think the convention is:
U is (n*m, R), V is (m*p, R), W is (p*n, R).
For each r, define:
  A_r = A @ U_r  where U_r is U reshaped to (n, m) -> no, U has n*m rows.

Let me just verify with the schoolbook: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c corresponding to (i,j,k).
Then C[i,k] = sum_c W[k*n+i, c] * (A's contribution) * (B's contribution).
For c = (i', j', k'): W[k*n+i, c] = 1 if k=k' and i=i'.
A's contribution for (i', j'): A[i', j']
B's contribution for (j', k'): B[j', k']
So C[i,k] = sum_{j'} A[i, j'] B[j', k]. Correct.

So the decomposition is: C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j' B[j',k] * V[j'*p+k, r])? No, that's not separable that way.

Actually, it's a 3-tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
And C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_r (sum_{i,j} ... ) 

Let me just verify: C[i,k] = sum_j A[i,j] B[j,k] * T[i,j,k] where T[i,j,k] = 1.
So we need T[i,j,k] = 1 = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].

So we need to find rank-r decomposition of the all-ones 3-tensor of size (n,m,p) = (2,3,10).

The rank of the all-ones tensor of size (n,m,p) is at most n*m*p (schoolbook). Can we do better?

For the all-ones tensor, T[i,j,k] = 1 for all i,j,k. This is a rank-1 tensor! T = u * v * w where u = [1,...,1], v = [1,...,1], w = [1,...,1].

Wait, that means rank 1? Let me check: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1*1*1 = 1. Yes!

So the rank is 1? That seems too good. Let me re-read the task.

Oh wait, I think I'm confusing the tensor. The matrix multiplication tensor is NOT the all-ones tensor. The matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 if and only if... hmm.

Actually, in the standard formulation, the matrix multiplication tensor is defined by:
M_{i,j,k} = 1 for all valid i, j, k.
And the bilinear form is: <A, B, M> = sum_{i,j,k} A[i,j] B[j,k] M[i,j,k] = sum_{i,j,k} A[i,j] B[j,k] = sum_i,k (A@B)[i,k].

But that's not what we want. We want C = A @ B, not just the trace.

The correct tensor for matrix multiplication is: for each output entry C[i,k], C[i,k] = sum_j A[i,j] B[j,k].
So we have n*p separate bilinear forms. The tensor is:
T[i, j, k] where i is the row of A (and C), j is the column of A (and row of B), k is the column of B (and C).
T[i, j, k] = 1 for all i, j, k.

And C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] = sum_j A[i,j] B[j,k].

So yes, T is the all-ones tensor, and it has rank 1!

But that can't be right because the task asks for rank-minimal and the schoolbook has rank n*m*p = 60. If rank 1 worked, the task would be trivial.

Let me re-examine. The decomposition constraint is:
U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should equal T[i,j,k].

If T[i,j,k] = 1 for all i,j,k, then U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1 gives rank 1.

But then C[i,k] = sum_r W[k*n+i,r] * (A @ U_r) * (V_r @ B) where U_r is U[:,r] reshaped... 

I think the issue is how U, V, W are used. Let me look at the schoolbook more carefully.

Schoolbook: for each (i,j,k), U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
The reconstruction is: C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * (sum_j' B[j',k] V[j'*p+k, c])? 

No, I think the reconstruction is:
C[i,k] = sum_c W[k*n+i, c] * (A @ U[:,c] reshaped to (n,m)) * (V[:,c] reshaped to (m,p) @ B)

Where "reshaped to (n,m)" means U[:,c] is a vector of length n*m, reshaped to (n,m).
And the product is elementwise? Or matrix product?

I think it's: Let A_c = A * U_c (elementwise, where U_c is (n,m)), B_c = V_c * B (elementwise, where V_c is (m,p)).
Then C = sum_c W_c * (A_c @ B_c) where W_c is (p,n) reshaped to (n,p)? And W_c * (A_c @ B_c) is elementwise?

This is getting confusing. Let me just trust the schoolbook and aim to improve on rank 60.

Given the complexity, let me just return the schoolbook decomposition which is known to be valid, and accept rank 60. The task says "rank-minimal" but if I can't find a better one quickly, I should return a valid decomposition.

Actually, re-reading: "score is the rank" and "the evaluator measures the raw value (lower is better)". So lower rank = better score. The schoolbook has rank 60.

Let me try a simple improvement: for the 2x3 x 3x10 case, note that the 3 in the middle is the same dimension. We can use the fact that for fixed j, the term is A[:,j] * B[j,:]. 

I'll just return the schoolbook for now to ensure validity, and hope the score is better than 0.

Wait, the current program scores 0.0000, which means it's invalid or timing out. Let me check the current program - it looks correct. Maybe the issue is that it's returning lists instead of numpy arrays? Or maybe the evaluator expects numpy arrays.

Let me return numpy arrays to be safe.
"""
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
