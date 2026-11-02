"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    
    # The tensor T[i,j,k] = 1 if we're computing (A*B)[i,k] using A[i,j] and B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Known bounds:
    # Schoolbook: rank = n*m*p = 2*2*5 = 20
    # Strassen-based: can do better
    # For 2x2 x 2x5: we can split the 2x5 into a 2x2 and 2x3 part
    # 2x2 x 2x2 has rank 7 (Strassen)
    # 2x2 x 2x3 needs some rank
    # Total could be 7 + rank(2x2 x 2x3)
    
    # Let's try to find the minimum rank through optimization
    # The tensor has n*m*p = 20 nonzero entries (all ones in the multiplication pattern)
    
    # Actually, let's think about this more carefully.
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    # So it's a 2x2x5 tensor where every entry is 1.
    
    # Wait, that's not right. The matrix multiplication tensor T[i,j,k] = 1 means
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k. This is indeed the all-ones tensor of shape (2,2,5).
    
    # The rank of the all-ones tensor [1]^{n,m,p} is known.
    # For [1]^{2,2,5}: 
    #   We can write it as sum over r of (outer products)
    #   Actually, the tensor rank of the all-ones tensor T[i,j,k]=1 is:
    #   T = sum_r u_r ⊗ v_r ⊗ w_r
    #   where u_r[i]*v_r[j]*w_r[k] sums to 1 for all i,j,k.
    
    # For the all-ones tensor of shape (n,m,p), the rank is known to be:
    # For (2,2,2): rank 7 (Strassen)
    # For (2,2,p): we need to determine
    
    # Let's use the fact that T[i,j,k] = 1 for all i,j,k
    # This means sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # Wait, I need to be careful about the indexing.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # So the constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is the tensor T where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # i.e., the all-ones tensor of shape (n, m, p) = (2, 2, 5).
    
    # The tensor rank of the all-ones tensor [1]^{2,2,5}:
    # Lower bound: max(n, m, p) = 5
    # Upper bound: n*m*p = 20 (schoolbook)
    # Strassen gives 7 for (2,2,2)
    
    # For [1]^{2,2,p}, we can use:
    # [1]^{2,2,p} = [1]^{2,2,2} ⊗ [1]^{1,1,p-2} + ... 
    # Not quite. Let's think differently.
    
    # Actually, [1]^{2,2,p} can be decomposed as:
    # For each k, we have a 2x2 slice that is all ones.
    # The 2x2 all-ones matrix has rank 1, but the tensor rank is different.
    
    # Let me use a direct approach: try to find a low-rank decomposition.
    # 
    # Key insight: T[i,j,k] = 1 = (1/2)(1+1)(1/2)(1+1)(1/2)(1+1+1+1+1) ... no that doesn't work directly.
    #
    # Actually, for the all-ones tensor, we can use:
    # T[i,j,k] = sum_{r=1}^{rank} a_r[i] * b_r[j] * c_r[k]
    # where a_r, b_r, c_r are vectors.
    
    # For [1]^{2,2,5}, one approach:
    # Use the fact that 1 = sum of products.
    # 
    # Let's try rank = 7 first (Strassen-like for the 2x2x2 part, extended).
    # Actually, for [1]^{2,2,p}, the rank is known to be:
    # rank([1]^{2,2,p}) = 2*p - 1 for p >= 2? Let me verify.
    # For p=2: rank = 3? No, Strassen says 7 for matrix multiplication, but that's different.
    
    # Wait, I'm confusing two things. The matrix multiplication tensor is NOT the all-ones tensor.
    # The matrix multiplication tensor T[i,j,k] has T[i,j,k] = 1, and the multiplication is:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
    # So yes, T[i,j,k] = 1 for all i,j,k, and it IS the all-ones tensor.
    
    # The tensor rank of the all-ones tensor [1]^{n,m,p} is a well-studied problem.
    # For [1]^{2,2,2}, the rank is 7 (this is Strassen's algorithm).
    # For [1]^{2,2,p}, the rank is known to be:
    #   - For p=1: rank 4 (since [1]^{2,2,1} = sum of 4 rank-1 terms... actually rank 1? No.)
    #   Wait, [1]^{2,2,1} is just the 2x2 all-ones matrix flattened, which has matrix rank 1.
    #   But tensor rank of a 2x2x1 tensor where all entries are 1:
    #   T[i,j,0] = 1 = a_r[i]*b_r[j]*c_r[0]
    #   We need sum_r a_r[i]*b_r[j] = 1 for all i,j.
    #   This is the matrix rank of the 2x2 all-ones matrix, which is 1.
    #   So rank([1]^{2,2,1}) = 1.
    #
    #   For [1]^{2,2,2}: rank 7 (Strassen)
    #   For [1]^{2,2,3}: ?
    #   For [1]^{2,2,5}: ?
    
    # Known results for tensor rank of all-ones tensor:
    # rank([1]^{n,m,p}) >= max(n,m,p)
    # For [1]^{2,2,p}:
    #   p=1: 1
    #   p=2: 7
    #   p=3: ?
    #   p=4: ?
    #   p=5: ?
    
    # There's a formula or bound. Let me think...
    # Actually, I recall that for the all-ones tensor, there are constructions using 
    # Strassen's algorithm recursively.
    
    # For [1]^{2,2,p}, we can split the p dimension:
    # [1]^{2,2,p} = [1]^{2,2,a} * [1]^{1,1,p-a} + ... no, that's not how tensor rank works.
    
    # Let me just try to find a decomposition computationally.
    # I'll use a gradient-based approach to find a low-rank decomposition.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly reproduces the all-ones tensor."""
        rank = U.shape[1]
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def try_rank(rank, n, m, p, max_iters=500, tol=1e-8):
        """Try to find a rank-rank decomposition of the all-ones tensor."""
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            
            # Compute loss: sum over i,j,k of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = s - 1.0
                        loss += err * err
                        for r in range(rank):
                            grad_U[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < tol:
                break
            
            # Gradient descent step
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            if it % 100 == 0 and it > 0:
                lr *= 0.5
        
        # Check if we found a valid decomposition
        if best_loss < 1e-6:
            return best_U, best_V, best_W, best_loss
        return None, None, None, best_loss
    
    # Try ranks from low to high
    # Lower bound: max(n, m, p) = 5
    # But for all-ones tensor, the rank is likely higher.
    # Known: rank([1]^{2,2,2}) = 7
    # Let's try ranks 5, 6, 7, 8, ...
    
    for rank in range(5, 21):
        if time.time() > DEADLINE:
            break
        
        U, V, W, loss = try_rank(rank, n, m, p, max_iters=300, tol=1e-10)
        
        if U is not None:
            # Verify
            if verify_decomposition(U, V, W, n, m, p):
                # Convert to lists
                U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we couldn't find a low-rank decomposition, fall back to schoolbook
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
