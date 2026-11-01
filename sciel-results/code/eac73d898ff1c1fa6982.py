"""Program for matrix multiplication 2,2,10 tensor rank minimization."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 10]
N, M, P = DIMENSIONS
TENSOR_DIM = N * M * P  # 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Schoolbook rank is 40. We try to find a better decomposition.
    # 
    # Key insight: For M_{n,m,p}, if we fix one index, the tensor decomposes
    # into a direct sum. Specifically, for M_{2,2,10}:
    # - The 40 non-zero entries of the tensor correspond to triples (i,j,k) 
    #   where i in {0,1}, j in {0,1}, k in {0,...,9}.
    # - U has shape (4, r), V has shape (20, r), W has shape (20, r).
    #
    # The matrix multiplication tensor T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For a rank-r decomposition, we need U of shape (n*m, r) = (4, r),
    # V of shape (m*p, r) = (20, r), W of shape (p*n, r) = (20, r).
    #
    # The constraint is: for all (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all other triples, the sum = 0.
    #
    # Since the tensor is diagonal in the (i,j,k) sense (each (i,j,k) appears exactly once),
    # we need: for each (i,j,k), there exists a column c such that
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all other (i',j',k'), the sum over c is 0.
    #
    # A simple approach: use the schoolbook (rank 40) and try to merge columns.
    # Two columns c1, c2 can be merged if they don't "conflict" - i.e., for every
    # (i,j,k) that is "hit" by both, the contributions cancel out.
    #
    # Actually, let's think about this differently. The tensor rank of M_{2,2,10}
    # is related to the tensor rank of M_{2,2,1} and M_{2,2,2}.
    #
    # For M_{2,2,1}: rank is 4 (schoolbook). Can we do better?
    # The tensor has 4 non-zero entries: (0,0,0), (1,1,0), (2,0,1), (3,1,1) in (alpha, beta, gamma)
    # where alpha = i*m+j, beta = j*p+k, gamma = k*n+i.
    # Wait, let me re-derive: n=2, m=2, p=1.
    # alpha = i*2+j, beta = j*1+k, gamma = k*2+i.
    # Non-zero when k=0 only (since p=1, k can only be 0).
    # alpha = i*2+j, beta = j, gamma = i.
    # So entries: (i*2+j, j, i) for i in {0,1}, j in {0,1}.
    # (0,0,0), (1,1,0), (2,0,1), (3,1,1).
    # This is the same as the identity-structured tensor. Rank is 4.
    #
    # For M_{2,2,2}: rank is 7 (Strassen).
    # 
    # For M_{2,2,10}, we can write it as a "sum" of tensors.
    # Actually, M_{2,2,10} is NOT a direct sum of smaller tensors in a simple way
    # because the k index is shared.
    #
    # Let me think about the structure more carefully.
    # T_{alpha, beta, gamma} = 1 if alpha = i*m+j, beta = j*p+k, gamma = k*n+i for some i,j,k.
    # With n=2, m=2, p=10:
    # alpha = 2i+j (i in {0,1}, j in {0,1}) -> alpha in {0,1,2,3}
    # beta = 2*10 + k... no, beta = j*p+k = j*10+k (j in {0,1}, k in {0,...,9}) -> beta in {0,...,19}
    # gamma = k*n+i = 2k+i (k in {0,...,9}, i in {0,1}) -> gamma in {0,...,19}
    #
    # The tensor is supported on: (2i+j, 10j+k, 2k+i) for i in {0,1}, j in {0,1}, k in {0,...,9}.
    # That's 2*2*10 = 40 entries.
    #
    # The key question: what is the minimum rank?
    # Lower bound: The "slice" method. Fix alpha=0 (i=0,j=0). Then beta = k, gamma = 2k.
    # The slice T_{0, beta, gamma} is non-zero only when gamma = 2*beta (for beta in 0..9).
    # This is a vector of weight 10.
    # Similarly for other slices.
    #
    # Actually, a known lower bound for M_{n,m,p} is max(n*m, m*p, p*n) = max(4, 20, 20) = 20.
    # But this is just a lower bound. The actual rank could be higher.
    #
    # For M_{2,2,10}, I'll try to find a rank-30 or better decomposition.
    # 
    # Strategy: Use the fact that for each (i,j), the k-index creates a "line" of 10 entries.
    # For fixed (i,j), the entries are at (2i+j, 10j+k, 2k+i) for k=0..9.
    # This is a "linear" structure in k.
    #
    # Let's try a rank-r decomposition where r is around 30.
    # We'll use a parameterized family and solve for the parameters.
    #
    # Actually, let me try a different approach: use the schoolbook and try to
    # find linear dependencies among the 40 rank-1 tensors.
    #
    # The 40 rank-1 tensors are:
    # T_{i,j,k} = e_{2i+j} ⊗ e_{10j+k} ⊗ e_{2k+i}
    #
    # We want to find r < 40 such tensors (or combinations) that sum to the original.
    #
    # Let's try rank 30. We'll set up a linear system.
    #
    # Actually, for the time budget, let me just try to find a decomposition
    # by solving a linear system with integer/half-integer constraints.
    
    # Try rank 30 first (schoolbook is 40)
    for rank in [30, 28, 26, 25, 24, 22, 20]:
        if time.time() > deadline - 30:
            break
        try:
            result = _try_rank(rank)
            if result is not None:
                return result
        except Exception:
            pass
    
    # Fall back to schoolbook
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][column] = 1
                V[j * P + k][column] = 1
                W[k * N + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _try_rank(rank):
    """Try to find a rank-r decomposition using numerical linear algebra."""
    start_time = time.time()
    deadline = start_time + 120.0
    
    # We need to find U (4 x rank), V (20 x rank), W (20 x rank)
    # such that for all (alpha, beta, gamma) in the support:
    #   sum_c U[alpha, c] * V[beta, c] * W[gamma, c] = 1
    # and for all other (alpha, beta, gamma):
    #   sum_c U[alpha, c] * V[beta, c] * W[gamma, c] = 0
    #
    # This is a system of 4*20*20 = 1600 equations with 4*rank + 20*rank + 20*rank = 44*rank unknowns.
    # But it's nonlinear (products of unknowns).
    #
    # Alternative: fix U and V, then solve for W. Or use iterative methods.
    #
    # Let's use a randomized approach with local search.
    #
    # Actually, for small ranks, let's try a structured approach.
    #
    # Key observation: The tensor M_{2,2,10} has a special structure.
    # For each (i,j), the 10 entries form a "line" parameterized by k.
    # 
    # Let's try to exploit this by using rank-1 tensors that are "spread out"
    # in the k-direction.
    #
    # For now, let's try a greedy approach: start with schoolbook and try to
    # merge pairs of columns that have overlapping support.
    
    # Initialize with schoolbook
    U = np.zeros((4, 40))
    V = np.zeros((20, 40))
    W = np.zeros((20, 40))
    for c, (i, j, k) in enumerate(product(range(2), range(2), range(10))):
        U[i * 2 + j, c] = 1
        V[j * 10 + k, c] = 1
        W[k * 2 + i, c] = 1
    
    # Try to reduce rank by finding linear dependencies
    # We want to find r < 40 columns that span the same "space"
    # in the tensor sense.
    
    # This is complex. Let me try a different approach:
    # Use the fact that M_{2,2,10} can be viewed as a block matrix.
    # The multiplication of a 2x2 by 2x10 matrix can be done as:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    #
    # For each i (row of A, 0 and 1), we have:
    # C_{i, :} = A_{i,0} * B_{0, :} + A_{i,1} * B_{1, :}
    #
    # This is just the standard matrix-vector multiplication for each row.
    # The rank of the full tensor is related to the rank of 2x2x10 multiplication.
    
    # Given the complexity, let me just return None and use the schoolbook.
    # But let me try one more thing: check if rank 30 is feasible by
    # solving a least-squares problem with random initialization.
    
    if time.time() > deadline - 10:
        return None
    
    # Try random initialization and Newton's method
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        U_r = np.random.randn(4, rank) * 0.1
        V_r = np.random.randn(20, rank) * 0.1
        W_r = np.random.randn(20, rank) * 0.1
        
        # Gradient descent on the loss: sum over all (a,b,g) of (T[a,b,g] - sum_c U[a,c]V[b,c]W[g,c])^2
        # where T is 1 for the 40 support entries and 0 elsewhere.
        
        T = np.zeros((4, 20, 20))
        for i, j, k in product(range(2), range(2), range(10)):
            T[i * 2 + j, j * 10 + k, k * 2 + i] = 1
        
        lr = 0.01
        for it in range(500):
            if time.time() > deadline - 2:
                break
            
            # Compute current product
            C = np.einsum('ac,bc,gc->abg', U_r, V_r, W_r)
            error = C - T
            
            # Gradients
            # d/dU[a,c] = sum_{b,g} error[a,b,g] * V[b,c] * W[g,c]
            dU = np.einsum('abg,bc,gc->ac', error, V_r, W_r)
            dV = np.einsum('abg,ac,gc->bc', error, U_r, W_r)
            dW = np.einsum('abg,ac,bc->gc', error, U_r, V_r)
            
            # Update
            U_r -= lr * dU
            V_r -= lr * dV
            W_r -= lr * dW
            
            # Check convergence
            if np.max(np.abs(error)) < 1e-10:
                # Found a solution! Convert to rational/half-integer
                # For now, just check if it's close to integer
                if _is_near_integer(U_r) and _is_near_integer(V_r) and _is_near_integer(W_r):
                    U_int = np.round(U_r).astype(int)
                    V_int = np.round(V_r).astype(int)
                    W_int = np.round(W_r).astype(int)
                    return _to_dict(U_int, V_int, W_int)
    
    return None


def _is_near_integer(A, tol=0.01):
    return np.max(np.abs(A - np.round(A))) < tol


def _to_dict(U, V, W):
    """Convert numpy arrays to list-of-lists format."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
