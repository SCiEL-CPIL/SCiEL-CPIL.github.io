"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    Uses a combination of known lower-rank decompositions and a randomized search for improvement.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Time management
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, 10s safety margin
    
    # Helper function to validate a decomposition
    def validate(U, V, W, tol=1e-7):
        """
        Validate that the decomposition exactly decomposes the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        For matrix multiplication C = A*B, we need:
        C[i,k] = sum_j A[i,j] * B[j,k]
        So T[i,j,k] should be 1 if the indices correspond to a valid multiplication, 0 otherwise.
        Actually, the tensor for matrix multiplication is:
        T[i,j,k] = 1 if we're computing C[i,k] = A[i,j]*B[j,k]
        Wait, let me think again.
        
        The matrix multiplication tensor M has entries M[i,j,k] such that:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        In tensor form, this is:
        C = A ⊗ B where (A ⊗ B)[i,k] = sum_j A[i,j] B[j,k]
        
        The decomposition is:
        M = sum_r U_r ⊗ V_r ⊗ W_r
        where M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        Wait, the W factor is W[k*n+i, r], so W is indexed by (k,i).
        
        For the matrix multiplication tensor, we need:
        M[i,j,k] = 1 if the computation involves A[i,j] and B[j,k]
        
        Actually, the standard form is:
        The tensor T for matrix multiplication C=A*B has:
        T[i,j,k] = delta (the tensor that picks out the j-th element)
        
        Let me use the convention from the problem:
        U has shape (n*m, rank), indexed by (i*m+j)
        V has shape (m*p, rank), indexed by (j*p+k)
        W has shape (p*n, rank), indexed by (k*n+i)
        
        The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        For matrix multiplication, we need T[i,j,k] = 1 if we're summing over j, i.e.,
        the tensor should represent the identity in the j-direction.
        
        Actually, for C = A * B, we have C[i,k] = sum_j A[i,j] B[j,k].
        So the tensor T[i,j,k] should be 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p),
        and 0 otherwise? No, that's not right either.
        
        Let me think about it differently. The matrix multiplication can be written as:
        C = sum_j (A[:,j] * B[j,:])
        
        In tensor form with the given indexing:
        T[i,j,k] = 1 if the indices are valid, representing the fact that A[i,j] and B[j,k] are multiplied.
        
        Actually, I think the correct interpretation is that the tensor T has:
        T[i,j,k] = 1 for all valid i,j,k, and the decomposition must exactly reproduce this tensor.
        But that would just be a rank 1 tensor if all entries are 1, which is not right.
        
        Let me reconsider. The matrix multiplication tensor is defined such that:
        (A * B)[i,k] = sum_j A[i,j] * B[j,k]
        
        In the tensor decomposition framework, we have:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        And we need T[i,j,k] to be the "multiplication tensor" which is 1 when we multiply A[i,j] by B[j,k].
        
        So T[i,j,k] = 1 for all valid (i,j,k) in the range? No, that doesn't make sense for a bilinear map.
        
        Actually, I think the tensor represents the bilinear form. The multiplication C=A*B is a bilinear map from (A,B) to C.
        The tensor T has entries T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
        
        For standard matrix multiplication, T[i,j,k] = 1 if and only if the indices match, i.e.,
        T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        Wait, that's not right. Let me use the schoolbook example to understand.
        
        In the schoolbook decomposition:
        U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k)
        
        So for a given (i,j,k), T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        In the schoolbook case, for each (i,j,k), there's exactly one r (the column corresponding to (i,j,k))
        where U[i*m+j, r] = 1, V[j*p+k, r] = 1, and W[k*n+i, r] = 1.
        So T[i,j,k] = 1 for all valid (i,j,k).
        
        So the tensor T is just all ones for valid indices? That seems too simple.
        
        Oh wait, I think I'm confusing myself. Let me look at this more carefully.
        
        The matrix multiplication C = A * B where A is n x m and B is m x p gives C which is n x p.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        The tensor T for this bilinear map has entries T[i,j,k] such that:
        C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
        
        For standard matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k), because we're just
        summing over j with coefficient 1.
        
        So yes, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        And the decomposition must satisfy:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        for all valid (i,j,k).
        
        Let me verify with the schoolbook:
        For a fixed (i,j,k), in the schoolbook decomposition, there's exactly one column r (the one
        corresponding to (i,j,k)) where U[i*m+j, r] = 1, V[j*p+k, r] = 1, and W[k*n+i, r] = 1.
        For all other columns, at least one of these is 0.
        So T[i,j,k] = 1 * 1 * 1 = 1. Correct.
        
        So the task is to find U, V, W such that for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        with the minimum number of columns r (the rank).
        """
        n, m, p = DIMENSIONS
        rank = U[0][__len__] if len(U) > 0 and len(U[0]) > 0 else 0
        
        # Build the tensor T as a 3D array
        T = np.ones((n, m, p))
        
        # Compute the decomposition result
        result = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        
        # Check if the result matches T
        return np.allclose(result, T, atol=tol)
    
    # Helper to convert to numpy for faster validation
    def validate_fast(U, V, W, tol=1e-7):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Reshape for computation
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We need to sum over r for each (i,j,k)
        result = np.zeros((n, m, p))
        for r in range(rank):
            U_r = U_reshaped[:, :, r]  # (n, m)
            V_r = V_reshaped[:, :, r]  # (m, p)
            W_r = W_reshaped[:, :, r]  # (p, n)
            
            # For each (i,j,k): U_r[i,j] * V_r[j,k] * W_r[k,i]
            # This is a bit tricky to vectorize directly
            # We can do it in steps
            for i in range(n):
                for k in range(p):
                    # result[i, :, k] = sum_j U_r[i,j] * V_r[j,k] * W_r[k,i]
                    # = W_r[k,i] * (U_r[i,:] @ V_r[:,k])
                    val = W_r[k, i] * np.dot(U_r[i, :], V_r[:, k])
                    result[i, :, k] = val
        
        T = np.ones((n, m, p))
        return np.allclose(result, T, atol=tol)
    
    # Even faster validation using einsum
    def validate_einsum(U, V, W, tol=1e-7):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        U_arr = np.array(U).reshape(n, m, rank)
        V_arr = np.array(V).reshape(m, p, rank)
        W_arr = np.array(W).reshape(p, n, rank)
        
        # We want result[i,j,k] = sum_r U_arr[i,j,r] * V_arr[j,k,r] * W_arr[k,i,r]
        # Using einsum: 'ijr,jkr,kin->ijk'
        result = np.einsum('ijr,jkr,kin->ijk', U_arr, V_arr, W_arr)
        
        T = np.ones((n, m, p))
        return np.allclose(result, T, atol=tol)
    
    # Start with schoolbook decomposition
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = best_result["U"][0][__len__]
    
    # Try randomized search for lower rank decompositions
    # This is a heuristic approach - not guaranteed to find optimal, but may improve
    rng = np.random.default_rng(42)
    
    def random_search(max_rank, num_trials, timeout=60):
        """Try random searches for lower rank decompositions."""
        n, m, p = DIMENSIONS
        start = time.time()
        
        for trial in range(num_trials):
            if time.time() > timeout + start_time:
                break
            if time.time() > deadline:
                break
            
            rank = max_rank
            # Initialize with random values
            U = rng.standard_normal((n * m, rank))
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))
            
            # Convert to lists for the interface
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if validate_einsum(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}, rank
        
        return None, max_rank
    
    # Try various ranks below schoolbook
    for rank in range(best_rank - 1, 10, -1):
        if time.time() > deadline:
            break
        
        result, r = random_search(rank, num_trials=100, timeout=20)
        if result is not None:
            if r < best_rank:
                best_result = result
                best_rank = r
    
    return best_result
# EVOLVE-BLOCK-END
