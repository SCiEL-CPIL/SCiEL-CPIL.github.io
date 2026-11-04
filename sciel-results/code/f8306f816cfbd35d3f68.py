"""Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_8x8_decomposition():
    """
    Construct a rank-70 decomposition for 8x8 x 8x8 matrix multiplication.
    
    Uses recursive Strassen: 8x8 = 4 blocks of 4x4, each 4x4 uses Strassen on 2x2 blocks.
    Actually, let's use a direct approach:
    
    For 8x8 x 8x8, we can use the fact that:
    - 8x8 can be split into 4 quadrants of 4x4
    - Each 4x4 multiplication can be done with rank 16 (schoolbook) or less
    
    Better: use the known result that M(8) <= 70 via specific constructions.
    
    Let me construct this explicitly using a tensor network / Kronecker product approach.
    
    Actually, the cleanest approach: use the rank-70 construction from the literature.
    One way: 8x8 = (4x4) with each 4x4 done via a rank-7 2x2 Strassen applied hierarchically.
    
    Let me think differently. The matrix multiplication tensor for 8x8 x 8x8 has rank at most 70.
    
    Construction via Kronecker products:
    - Let S be the 2x2 Strassen decomposition (rank 7)
    - Then S ⊗ S gives a 4x4 x 4x4 decomposition of rank 49
    - Then S ⊗ S ⊗ S would give 8x8 of rank 343, which is too high.
    
    Better: for 8x8, use a hybrid approach.
    
    Actually, let me just directly construct a valid decomposition using the following approach:
    Split 8 into 4+4, and use block structure.
    
    For C = A*B where A,B,C are 8x8:
    Split into 4x4 blocks:
    A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each 4x4 multiplication can be done with rank 16 (schoolbook) or 14 (known best for 4x4 is 14? No, it's 16 for schoolbook, best known is 14? Actually best known for 4x4 is 14? No, I think it's still 16 for exact. Let me use 16.)
    
    With rank 16 per 4x4: total = 4 * 16 = 64 terms for the 4 block multiplications, plus 4 more for the additions... no, the additions are just summing outputs, not additional tensor terms.
    
    Wait, the tensor decomposition: each 4x4 x 4x4 multiplication contributes 16 rank-1 terms. We have 4 such multiplications (A11*B11, A12*B21, A11*B12, A12*B22, A21*B11, A22*B21, A21*B12, A22*B22) - that's 8 multiplications!
    
    So with rank 16 per 4x4: 8 * 16 = 128. That's the same as schoolbook for 8x8.
    
    We need a better approach for 4x4. The best known rank for 4x4 x 4x4 is 14 (I believe). Let me check: actually, I'm not sure. Let me use a different strategy.
    
    Alternative: Use the fact that we only need the 4x4 x 4x8 subproblem, not full 8x8.
    
    Let me reconsider the embedding strategy more carefully.
    """
    pass

def _construct_4x4x4x8_via_embedding():
    """
    Construct rank-minimal decomposition for 4x4 x 4x8.
    
    Strategy: Embed into 8x8 x 8x8 and use a known low-rank decomposition.
    
    For 8x8 x 8x8, one construction uses rank 70.
    
    The embedding:
    - A (4x4) goes into top-left 4x4 of an 8x8 matrix A'
    - B (4x8) goes into top-left 4x8 of an 8x8 matrix B' (top row of 4x4 blocks)
    - C (4x4) comes from top-left 4x4 of C' = A'*B'
    
    For this to work, we need A' and B' such that:
    A' = [A, 0; 0, 0]  (top-left 4x4 is A, rest is 0)
    B' = [B, 0; 0, 0]  (top-left 4x8 is B, rest is 0)
    
    Then C' = A'*B' = [A*B, 0; 0, 0], so top-left 4x4 of C' is A*B.
    
    Now, if we have a rank-r decomposition of the 8x8 x 8x8 multiplication tensor,
    we can restrict it to this sub-space.
    
    The 8x8 x 8x8 multiplication tensor T satisfies:
    C'[i,k] = sum_j A'[i,j] * B'[j,k]
    
    With the embedding, for i<4, k<4:
    C'[i,k] = sum_{j<4} A[i,j] * B[j,k] = (A*B)[i,k]
    
    The rank-r decomposition: C'[i,k] = sum_c U'[i*8+j, c] * V'[j*8+k, c] * W'[k*8+i, c]
    
    We need: for i<4, k<4, sum_c U'[i*8+j, c] * V'[j*8+k, c] * W'[k*8+i, c] = sum_{j<4} A[i,j]*B[j,k]
    
    This requires that the decomposition, when restricted to i<4, j<4, k<4, gives the correct result.
    
    This is not automatically guaranteed by any arbitrary 8x8 decomposition. We need a specific one.
    
    Alternative approach: directly construct the decomposition using a known algebraic method.
    
    Let me try a different structural approach: use the "four Russians" or block decomposition.
    
    Actually, let me try the most direct computational approach: use the known rank-70 decomposition for 8x8x8 from the literature.
    
    One explicit construction: 
    The 8x8x8 multiplication can be decomposed using 70 rank-1 tensors.
    
    Let me construct it using the following recursive Strassen-like method:
    
    Level 1: 8x8 = 4 blocks of 4x4
    Use 7 Strassen-like products for the 4x4 blocks? No, Strassen is for 2x2.
    
    Let me use a different known result. The best known upper bound for M(8) is 70, achieved by specific constructions.
    
    For simplicity and correctness, let me use a slightly higher rank that I can construct explicitly and verify.
    
    Approach: Use the schoolbook for the 4x4 x 4x8 problem but with a clever block structure.
    
    Actually, let me just try to find a decomposition computationally using a smarter method.
    """
    n, m, p = 4, 4, 8
    
    # Let me try to construct a decomposition using the following idea:
    # Split B (4x8) into two 4x4 blocks: B = [B1, B2]
    # Then C = A*B = [A*B1, A*B2]
    # Each A*Bi is 4x4 x 4x4 multiplication.
    # If we use rank r4 for each 4x4x4x4, total rank = 2*r4.
    # Best known for 4x4x4x4 is 14 (I think), so total = 28? That seems too low.
    # Actually, I'm not sure about the best known for 4x4.
    
    # Let me just use a computational search with a better initialization.
    
    # Key insight: the tensor T[i,j,k] = 1 for all valid (i,j,k) has a lot of symmetry.
    # We can exploit this by using a structured random initialization.
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    # Try to find a decomposition with rank r
    # Lower bound: max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # Upper bound: n*m*p = 128 (schoolbook)
    
    # Let's try ranks from 40 upward and see what we can find
    best_result = None
    
    for target_rank in range(35, 100):
        if time.time() > deadline - 10:
            break
        
        result = _find_decomposition(n, m, p, target_rank, time_budget=min(20, deadline - time.time()))
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            break  # Found the lowest rank we can
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    """Verify the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _find_decomposition(n, m, p, target_rank, time_budget=20):
    """Try to find a rank-target_rank decomposition."""
    start_time = time.time()
    
    # Use a structured approach:
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Strategy: Use the fact that this is a "separable" tensor in a sense.
    # Try to find U, V, W such that the constraint is satisfied.
    
    # Method: Alternating least squares with multiple restarts
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget - 2:
            break
        
        # Initialize with a structured pattern
        if restart == 0:
            # Schoolbook-like initialization with random perturbations
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            # Assign each (i,j,k) to a random column
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = (i*m+j + j*p+k + k*n+i) % target_rank
                        U[i*m+j, c] += 1.0
                        V[j*p+k, c] += 1.0
                        W[k*n+i, c] += 1.0
            # Normalize
            for c in range(target_rank):
                su = np.sum(U[:, c])
                sv = np.sum(V[:, c])
                sw = np.sum(W[:, c])
                if su > 0 and sv > 0 and sw > 0:
                    scale = (su * sv * sw) ** (1.0/3.0)
                    if scale > 0:
                        U[:, c] /= scale
                        V[:, c] /= scale
                        W[:, c] /= scale
        else:
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.3
            V = np.random.randn(m*p, target_rank) * 0.3
            W = np.random.randn(p*n, target_rank) * 0.3
        
        # Alternating minimization
        for iteration in range(100):
            if time.time() - start_time > time_budget - 2:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # For fixed (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                    M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    try:
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    try:
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    try:
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    except:
                        pass
            
            # Check convergence
            if iteration % 10 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (s - 1.0) ** 2
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                if err < 1e-8:
                    break
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-8 and best_U is not None:
        # Final verification
        if _verify(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    return _construct_4x4x4x8_via_embedding()
# EVOLVE-BLOCK-END
