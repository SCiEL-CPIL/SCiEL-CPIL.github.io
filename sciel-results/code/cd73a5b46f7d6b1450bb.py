"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; else 0."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W form a valid decomposition of the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition satisfies: for all (i,j), (j',k), (k',i'),
        sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
      = 1 if i=i', j=j', k=k' else 0
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each term r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We check: T_decomposed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # This is equivalent to: for each r, add outer(U[:,r], outer(V[:,r], W[:,r]))
    
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    T_true = _build_tensor(n, m, p)
    return np.allclose(T_decomp, T_true, atol=tol)

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_reduction(n, m, p, target_rank, time_limit=10.0):
    """Try to find a decomposition of rank target_rank using random search + refinement.
    This is a heuristic and may not succeed. Returns None if not found."""
    start_time = time.time()
    
    # Try random initialization with rank = target_rank
    # The tensor T has n*m*p non-zero entries.
    # We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    
    # Strategy: random Gaussian init, then gradient descent on
    # F = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] - T
    # Loss = ||F||_F^2
    
    # For small problems, try a few random restarts
    best_loss = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    n_restarts = 5
    n_steps = 200
    lr = 0.01
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        r = target_rank
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        T_true = _build_tensor(n, m, p)
        
        for step in range(n_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current tensor
            F = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = F - T_true
            loss = np.sum(diff**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-12:
                break
            
            # Gradients:
            # dL/dU[i,r] = 2 * sum_{j,k} diff[i,j,k] * V[j,r] * W[k,r]
            # dL/dV[j,r] = 2 * sum_{i,k} diff[i,j,k] * U[i,r] * W[k,r]
            # dL/dW[k,r] = 2 * sum_{i,j} diff[i,j,k] * U[i,r] * V[j,r]
            
            dU = 2.0 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            dV = 2.0 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            dW = 2.0 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_loss < 1e-9:
            return best_U, best_V, best_W
    
    if best_loss < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    # Schoolbook decomposition as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find lower-rank decompositions
    # Known results:
    # - 2x3x3 has rank 7 (Laderman)
    # - 3x3x3 has rank 23 (Strassen-like)
    # - 2x3x9: we can decompose as two 1x3x9 blocks (each row of A times B)
    #   or as three 2x3x3 blocks
    
    # Approach 1: Decompose into three 2x3x3 blocks
    # A is 2x3, B is 3x9. Split B into three 3x3 blocks B1, B2, B3.
    # C = A * [B1 B2 B3] = [A*B1, A*B2, A*B3]
    # Each A*B_i is 2x3, computed with rank 7 (Laderman).
    # Total rank = 3 * 7 = 21.
    
    # Approach 2: Decompose into two 1x3x9 blocks (rows of A)
    # Row 0 of A (1x3) times B (3x9) -> 1x9 output
    # Row 1 of A (1x3) times B (3x9) -> 1x9 output
    # Each 1x3x9: rank? The tensor for 1x3x9 multiplication.
    # A is 1x3 (3 entries), B is 3x9 (27 entries), C is 1x9 (9 entries).
    # C[k] = sum_j A[j] * B[j,k]
    # This is essentially: for each k, C[k] is a bilinear form in A and the k-th column of B.
    # The rank of 1x3x9: we can write C[k] = sum_j A[j] B[j,k].
    # This is a sum of 3*9 = 27 rank-1 terms (schoolbook).
    # Can we do better? Each term A[j]*B[j,k] involves one entry of A and one of B.
    # The tensor T[i,j; j',k'; i',k'] where i=0 only, so T[j, j',k'; k'] = 1 if j=j' and k=k'.
    # This is a "diagonal" type tensor. The rank is at most 27.
    # Actually, for 1xm x mxp, the rank is m*p (schoolbook) because each output C[k]
    # depends on all m entries of A and p entries of B independently.
    # Wait, no. C[k] = sum_j A[j] * B[j,k]. For fixed k, this is a dot product.
    # The tensor is: T[j, j',k'; k] = delta_{j,j'} * delta_{k',k} ... 
    # Hmm, let me think more carefully.
    # A has index j (j=0,1,2), B has indices (j,k) (j=0,1,2; k=0..8), C has index k (k=0..8).
    # T[j; j',k'; k] = 1 if j=j' and k'=k, else 0.
    # This is a tensor of shape (3, 27, 9).
    # Rank: each rank-1 term is u ⊗ v ⊗ w where u ∈ R^3, v ∈ R^27, w ∈ R^9.
    # The term contributes u[j]*v[j',k']*w[k] to T[j,j',k',k].
    # For this to be 1 when j=j', k'=k and 0 otherwise:
    # We need sum_r u_r[j] * v_r[j',k'] * w_r[k] = delta_{j,j'} * delta_{k',k}
    # One decomposition: for each (j,k), use u = e_j, v = e_{j,k}, w = e_k. That's 27 terms.
    # Can we do fewer? The tensor is essentially the "identity" between the 3-dim A-space
    # and the 27-dim B-space, projected to 9-dim C-space.
    # The rank of such a tensor is at least max(3, 27, 9) = 27? No, that's not a lower bound.
    # Actually, for the tensor T[j,j',k',k] = delta_{j,j'} delta_{k',k}, this is the
    # "slice" tensor. Its rank is 27 (each (j,k) pair needs its own term).
    # So 1x3x9 has rank 27, and two rows give 54. Same as schoolbook.
    
    # So Approach 1 (three 2x3x3 blocks with Laderman rank 7 each) gives rank 21.
    # Let me implement Laderman's algorithm for 2x3x3.
    
    # Laderman's algorithm for 2x3 x 3x3 -> 2x3:
    # A = [[a,b,c],[d,e,f]], B = [[g,h,i],[j,k,l],[m,n,o]], C = [[p,q,r],[s,t,u]]
    # p = a*g + b*j + c*m
    # q = a*h + b*k + c*n
    # r = a*i + b*l + c*o
    # s = d*g + e*j + f*m
    # t = d*h + e*k + f*n
    # u = d*i + e*l + f*o
    # 
    # Laderman uses 7 multiplications:
    # m1 = (a+d)(g+j)
    # m2 = (b+e)(h+k)
    # m3 = (c+f)(i+l)
    # m4 = (a+e-c-f)(g+j-h-k)  -- wait, let me look up the exact formula.
    # 
    # Actually, the standard Laderman algorithm for 2x3 x 3x2 (or similar) is different.
    # Let me use a known rank-7 decomposition for 2x3x3.
    # 
    # Reference: The rank of 2x3x3 matrix multiplication is 7.
    # One such decomposition:
    # Let A = [[a,b,c],[d,e,f]], B = [[g,h,i],[j,k,l],[m,n,o]]
    # 
    # m1 = a*g
    # m2 = b*h
    # m3 = c*i
    # m4 = d*j
    # m5 = e*k
    # m6 = f*l
    # m7 = (a+b+c+d+e+f)*(g+h+i+j+k+l+m+n+o) ... no, that's not right.
    # 
    # Let me think differently. The rank of the 2x3x3 tensor is 7.
    # I'll use a numerical approach to find it.
    
    # For now, let me try a direct numerical search for a rank-21 decomposition
    # by finding rank-7 decompositions for each 2x3x3 block.
    
    # Actually, let me just try to find a good decomposition numerically.
    # Try ranks from 21 upward.
    
    for target_rank in [21, 22, 23, 24, 25, 27, 30, 36, 45, 54]:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            break
        
        result = _try_rank_reduction(n, m, p, target_rank, time_limit=min(15.0, deadline - time.time() - 5))
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break  # Found a valid decomposition, use it
    
    # If no better decomposition found, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
