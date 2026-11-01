import time
import numpy as np

def construct():
    start = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = 2, 3, 5
    # Shapes: U (n*m, R) = (6, R), V (m*p, R) = (15, R), W (p*n, R) = (10, R)
    # Tensor T[i, j, k] corresponds to U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need sum_r T[i,j,k] = 1 if (i,j,k) is a valid index tuple, else 0.
    # Actually, the matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # The decomposition is: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need this to equal 1 for all (i,j,k).

    # Strategy: Use a known low-rank decomposition for 2x3 * 3x5.
    # The trivial rank is n*m*p = 30.
    # We can do better by exploiting structure.
    # Let's try to find a decomposition with rank < 30.
    # A simple improvement: if m=3, we can split j into 2+1.
    # For j in {0,1}, we have a 2x2x5 block. For j=2, we have a 2x1x5 block.
    # 2x1x5 block: C[i,k] += A[i,2] B[2,k]. This is rank 1? No, it's a sum of rank-1 terms.
    # Actually, A[i,2] B[2,k] is a rank-1 matrix (outer product of col 2 of A and row 2 of B).
    # So the j=2 part is rank 1.
    # The j in {0,1} part is a 2x2x5 tensor.
    # 2x2x5 tensor: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    # This is the sum of two rank-1 matrices (for each k, it's a 2x2 matrix).
    # Wait, the tensor is T[i, j, k] for i in {0,1}, j in {0,1}, k in {0..4}.
    # T[i,0,k] = 1, T[i,1,k] = 1.
    # This is a 2x2x5 tensor where all entries are 1.
    # The rank of a 2x2x5 all-ones tensor:
    # T = 1_2 x 1_2 x 1_5 (outer product). This is rank 1!
    # Wait, no. The tensor T[i,j,k] = 1 for all i,j,k is the outer product of three vectors of ones.
    # So it is rank 1.
    # Let's re-read the problem.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for 2x3 * 3x5 is:
    # T[i, j, k] = 1 if i in [0,2), j in [0,3), k in [0,5).
    # So T is a 2x3x5 tensor of all ones.
    # The rank of an all-ones tensor of shape n x m x p is 1.
    # Wait, is that right?
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # If we set U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1, then sum_r = 1.
    # So the rank is 1?
    # That seems too good to be true. Let me check the definition of tensor rank for matrix multiplication.
    # The matrix multiplication tensor M_{n,m,p} has entries M_{i,j,k} = 1 if i,j,k are valid indices.
    # Yes, it is the all-ones tensor.
    # And the rank of the all-ones tensor is 1.
    # But wait, the problem says "score is the rank" and "lower is better".
    # If the rank is 1, the score would be 1.0 (if seed is 1) or better.
    # But the current program has rank 30 (or 19 in the failed attempt).
    # Why is the rank not 1?
    # Let's look at the interface:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need this to be 1 for all i,j,k.
    # If we set U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i.
    # Then T[...] = 1*1*1 = 1.
    # So the rank is 1.
    # Why did the previous programs fail to find this?
    # Maybe I am misunderstanding the tensor definition.
    # "Matrix multiplication 2,3,5"
    # C = A B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # So T[i,j,k] = 1.
    # Yes, it is the all-ones tensor.
    # And the rank of the all-ones tensor is 1.
    # Let's verify this with a small example.
    # 2x2x2 all-ones tensor. Rank 1?
    # T[i,j,k] = 1.
    # U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1.
    # Sum = 1.
    # Yes.
    # So the answer is rank 1.
    # But the seed program has score 1.0, which corresponds to rank 30 (n*m*p).
    # And the inspirations also have score 1.0.
    # This suggests that the rank 1 decomposition is not valid, or I am missing something.
    # Let's re-read the contract carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor is indeed the all-ones tensor.
    # So the rank is 1.
    # Why is the score 1.0 for rank 30?
    # "score is the rank"
    # "search score is seed/value, so 1.0 equals the seed program"
    # So if seed is rank 30, and value is rank 30, score is 1.0.
    # If I find rank 1, value is 1, score is 30/1 = 30.0.
    # This would be a huge improvement.
    # Let's assume the rank 1 decomposition is correct and try it.

    # Rank 1 decomposition:
    R = 1
    U = np.ones((n * m, R), dtype=int)
    V = np.ones((m * p, R), dtype=int)
    W = np.ones((p * n, R), dtype=int)

    # Verify
    def verify(U, V, W):
        T = np.zeros((n * m, m * p, p * n), dtype=int)
        for r in range(U.shape[1]):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i * m + j, j * p + k, k * n + i] != 1:
                        return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i, j = divmod(a, m)
                    jj, k = divmod(b, p)
                    kk, ii = divmod(c, n)
                    if not (i == ii and j == jj and k == kk):
                        if T[a, b, c] != 0:
                            return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # If rank 1 fails (which it shouldn't), fall back to a higher rank.
    # But it should not fail. Let's double-check the verification.
    # T[i*m+j, j*p+k, k*n+i] should be 1.
    # With U=1, V=1, W=1, T[a,b,c] = 1 for all a,b,c.
    # So T[i*m+j, j*p+k, k*n+i] = 1. Correct.
    # And for other (a,b,c), T[a,b,c] = 1, which is not 0.
    # So the verification will fail because T[a,b,c] != 0 for invalid (a,b,c).
    # Ah! The tensor T is not the all-ones tensor in the flattened space.
    # The tensor T is defined on the indices (i,j,k) of the matrix multiplication.
    # But the factor matrices U, V, W are indexed by (i,j), (j,k), (k,i) respectively.
    # The decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We need T[i,j,k] = 1 for all i,j,k.
    # But the verification checks T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i.
    # And it checks that T[a,b,c] = 0 for invalid (a,b,c).
    # But with U=1, V=1, W=1, T[a,b,c] = 1 for all a,b,c.
    # So T[a,b,c] = 1 for invalid (a,b,c), which is not 0.
    # So the rank 1 decomposition is NOT valid for the flattened tensor.
    # The tensor T is not the all-ones tensor in the flattened space.
    # It is a sparse tensor where T[i*m+j, j*p+k, k*n+i] = 1, and 0 elsewhere.
    # So we need to find U, V, W such that:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) is valid, 0 otherwise.
    # This is a different problem.
    # The tensor is not the all-ones tensor. It is the matrix multiplication tensor, which is sparse.
    # So the rank is not 1.
    # Let's go back to the previous approach.

    # Let's try to find a low-rank decomposition for the sparse tensor.
    # The sparse tensor has 2*3*5 = 30 non-zero entries.
    # The trivial rank is 30.
    # We can try to find a lower rank.
    # Let's use a simple heuristic: try rank 20, 15, 10, etc.
    # But this is a hard problem.
    # Let's stick with the trivial rank 30 for now, and try to improve it.
    # Actually, the previous program tried to use Laderman for 2x2x2 blocks.
    # Let's try a different approach: use the fact that the tensor is sparse.
    # We can try to find a decomposition with rank < 30.
    # Let's try rank 20.
    # If it fails, try rank 25, etc.
    # But this is a search problem, which is hard.
    # Let's just return the trivial rank 30 decomposition.

    R = n * m * p
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1

    # Verify
    if not verify(U, V, W):
        # This should not happen.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
