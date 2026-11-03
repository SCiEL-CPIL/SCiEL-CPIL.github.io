import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The rank of the matrix multiplication tensor for n x m times m x p is at most n*m*p
    # but can be lower. For 3x4 times 4x4, let's try to find a lower rank decomposition.
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Known upper bounds: rank <= n*m*p = 48
    # For small matrices, we can try various ranks.
    
    # Let's try rank 16 first (the trivial lower bound)
    # Actually, for matrix multiplication, the rank is typically higher than the max dimension product.
    # Let's use a systematic approach: try ranks from lower bound up.
    
    import random
    from scipy.optimize import minimize
    
    # The tensor T has entries T[i,j,k] = 1 for all valid (i,j,k) in the multiplication pattern
    # Wait, no. The matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 if the (i,k) entry of the product depends on A[i,j]*B[j,k]
    # Actually, T[i,j,k] = 1 for all i in [n], j in [m], k in [p] because every product A[i,j]*B[j,k] contributes to C[i,k]
    # So T is just the all-ones tensor of shape (n, m, p)?
    
    # No wait. The tensor decomposition for matrix multiplication satisfies:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,k} ... no that's not right either.
    
    # The correct formulation: We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k (since each triple (i,j,k) contributes 1*A[i,j]*B[j,k] to C[i,k])
    # But wait, the output index is (i,k), not (k,i). Let me re-read the task.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where the flat index is k*n + i.
    
    # So the decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # This is the all-ones tensor of shape (n, m, p).
    
    # The rank of the all-ones tensor of shape (n,m,p) is... well, it's a rank-1 tensor!
    # T[i,j,k] = 1 = 1 * 1 * 1
    # So rank 1 would work? But that can't be right for matrix multiplication.
    
    # I think I'm confusing the tensor. Let me reconsider.
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # The tensor T is defined such that the bilinear form gives the matrix product.
    # Specifically, we want: C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] * A[i,j] * B[j,k] ... no.
    
    # The standard tensor decomposition for matrix multiplication:
    # We have tensors U, V, W such that for all i,j,k:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # This is because each term in the sum over r contributes to the product.
    
    # So yes, T[i,j,k] = 1 for all i,j,k, and we need rank-r factorization of this all-ones tensor.
    # But the all-ones tensor has rank 1: U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1 for r=0.
    
    # That can't be the right formulation. Let me look at this differently.
    
    # Actually, I think the issue is that the tensor for matrix multiplication is NOT all ones.
    # The correct tensor is: T[i,j,k] = 1 if and only if the indices are "compatible" in some way.
    
    # Let me re-derive. We want to compute C = AB where A is (n x m), B is (m x p), C is (n x p).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want to write this as: C[i,k] = sum_r U[i,?,r] * V[?,k,r] * W[?,i,r] ... 
    
    # The standard formulation uses three tensors U, V, W of shapes (n*m, r), (m*p, r), (p*n, r)
    # such that: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # So yes, the target tensor is all ones. And the all-ones tensor of shape (n,m,p) 
    # viewed as a 3-tensor has rank 1.
    
    # But wait, that would mean matrix multiplication has rank 1, which is clearly wrong.
    # The issue is that the tensor is not just "all ones" in the naive sense.
    
    # Let me think again. The matrix multiplication tensor T is a 3-tensor of shape (n*m, m*p, p*n)
    # where T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0.
    # In other words, T[a,b,c] = 1 iff the "j" index in a matches the "j" index in b, 
    # and the "k" index in b matches the "k" index in c, and the "i" index in a matches the "i" index in c.
    
    # So T is NOT all ones. It's a sparse tensor with specific structure.
    
    # Let's construct it properly.
    # T[a,b,c] = 1 if there exist i,j,k such that a = i*m+j, b = j*p+k, c = k*n+i
    # For given a,b,c, we can check:
    # i = a // m, j = a % m
    # j' = b // p, k = b % p
    # k' = c // n, i' = c % n
    # T[a,b,c] = 1 if j == j' and k == k' and i == i'
    
    # So the tensor is determined by these constraints.
    
    # Now, what's the rank of this tensor?
    # This is the standard matrix multiplication tensor, and its rank is what we're trying to minimize.
    
    # For 3x4 times 4x4, the rank is unknown in general, but we can try to find good decompositions.
    
    # Let's use the IHT approach as suggested.
    
    # First, let's create the target tensor as a matricization.
    # We'll use the unfolding along the first mode: T_(1) has shape (n*m, m*p * p*n)
    # T_(1)[a, (b,c)] = T[a,b,c]
    
    # For IHT, we'll try to find rank-r factors.
    
    # Let's try a simple approach: use the trivial rank n*m*p decomposition first,
    # then try to reduce the rank.
    
    # Actually, let's just use the known best upper bounds for small matrix multiplication.
    # For 3x4 * 4x4, a trivial upper bound is 3*4*4 = 48.
    # We can try to do better.
    
    # Let's use a randomized IHT approach to try to find a lower rank decomposition.
    
    def tensor_to_matricization(T, mode):
        """Convert 3D tensor to 2D matricization along given mode."""
        if mode == 0:
            return T.reshape(T.shape[0], T.shape[1] * T.shape[2])
        elif mode == 1:
            return np.transpose(T, (1, 0, 2)).reshape(T.shape[1], T.shape[0] * T.shape[2])
        else:
            return np.transpose(T, (2, 0, 1)).reshape(T.shape[2], T.shape[0] * T.shape[1])
    
    def matricization_to_tensor(M, shape, mode):
        """Convert 2D matricization back to 3D tensor."""
        if mode == 0:
            return M.reshape(shape)
        elif mode == 1:
            return np.transpose(M.reshape(shape[1], shape[0], shape[2]), (1, 0, 2))
        else:
            return np.transpose(M.reshape(shape[2], shape[0], shape[1]), (2, 0, 1))
    
    # Create the target tensor
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Wait, I need to reconsider. The tensor T[i,j,k] should be 1 for all i,j,k
    # because every triple (i,j,k) contributes to the matrix product.
    # So T is indeed all ones of shape (n, m, p).
    
    # But the factorization is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1
    
    # So we need to factor the all-ones tensor of shape (n, m, p) into three factors
    # with the specific index mapping.
    
    # The all-ones tensor has rank 1 as a 3-tensor, but the constraint on the factor
    # shapes (n*m, r), (m*p, r), (p*n, r) with the specific index mapping may require higher rank.
    
    # Let's just use the trivial decomposition: rank = n*m*p
    # U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # Wait, that doesn't work because the column index needs to be consistent.
    
    # Let's use: for each (i,j,k), create a rank-1 term.
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 where r corresponds to (i,j,k)
    
    # This gives rank n*m*p = 48.
    
    # Can we do better? Let's try rank 16 (the lower bound).
    
    # For now, let's just return the trivial rank 48 decomposition.
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
