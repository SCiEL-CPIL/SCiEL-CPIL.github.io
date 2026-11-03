import time
import numpy as np

DIMENSIONS = [3, 3, 4]
N, M, P = 3, 3, 4
SCHOOLBOOK_RANK = N * M * P  # 36

def _build_tensor():
    """Build the matmul tensor T of shape (N*M, M*P, P*N).
    T[i*M+j, j'*P+k, k'*N+i'] = 1 iff i==i' and j==j' and k==k'.
    """
    T = np.zeros((N*M, M*P, P*N), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-6):
    """Check if sum_t U[a,t]*V[b,t]*W[c,t] == T[a,b,c] for all a,b,c."""
    D = np.einsum('at,bt,ct->abc', U, V, W)
    return np.allclose(D, T, atol=tol)

def _schoolbook():
    """Standard schoolbook decomposition of rank 36."""
    U = np.zeros((N*M, SCHOOLBOOK_RANK), dtype=np.float64)
    V = np.zeros((M*P, SCHOOLBOOK_RANK), dtype=np.float64)
    W = np.zeros((P*N, SCHOOLBOOK_RANK), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M + j, col] = 1
                V[j*P + k, col] = 1
                W[k*N + i, col] = 1
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists, rounding to nearest half-integer."""
    def to_list(arr):
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

def _round_to_half(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2) / 2

def _try_structured_merging(T, deadline):
    """
    Try to find a lower-rank decomposition by systematically merging schoolbook terms.
    
    Key insight: A rank-1 term (u ⊗ v ⊗ w) contributes u[a]*v[b]*w[c] to T[a,b,c].
    If we have a term where u is nonzero at multiple a's, v at multiple b's, w at multiple c's,
    it contributes to all combinations. We need the sum over all terms to exactly match T.
    
    Strategy: Start from schoolbook and try to "merge" pairs of terms by finding
    a single term that covers both, possibly with corrections.
    
    For 3x3*3x3, rank 21 is known. For 3x3*3x4, we try ranks 30-35.
    """
    # Precompute: for each (a, b, c) where T[a,b,c]=1, we need coverage.
    # T has 36 ones.
    
    # Let's try a more direct algebraic approach:
    # Assume we want rank r. We have r terms, each with (u_t, v_t, w_t).
    # The constraint is: for all (a,b,c), sum_t u_t[a]*v_t[b]*w_t[c] = T[a,b,c].
    
    # This is a system of 9*12*12 = 1296 equations in 9*r + 12*r + 12*r = 33r unknowns.
    # For r=30: 990 unknowns, 1296 equations -> overdetermined, but T is sparse.
    
    # Let's try a specific structured form:
    # Group by i (row of A, row of C). For each i, we have a 3x4 matmul (M x P).
    # The i-th "slice" of T: T[i*M+j, j'*P+k, k*N+i] for j,j'=0..2, k=0..3.
    # This is a 3x4 matrix multiplication tensor (rank 12 schoolbook, maybe lower).
    
    # Actually, the full tensor is a sum over i of "slice" tensors, but they're not
    # independent because the indices are coupled.
    
    # Let me try: for each i, use a 3x4 matmul decomposition. If we can find
    # a rank-r_i decomposition for 3x4 matmul, total rank would be sum_i r_i.
    # But that's not how it works - the terms can share structure across i.
    
    # Alternative: Use the fact that for fixed j (column of A, row of B),
    # we have a sum over k of A[i,j]*B[j,k]*C[k,i]... no, that's the output.
    
    # Let me try a concrete small-rank search with a specific parameterization.
    
    # Parameterization: Assume U, V, W have a block structure.
    # For rank r, let's try r = 30, 31, 32, 33, 34, 35.
    
    # For each r, we'll use a randomized approach with structured initialization.
    
    for target_rank in range(35, 25, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(attempt * 1000 + target_rank)
            
            # Initialize with a perturbed schoolbook-like structure
            U = np.zeros((N*M, target_rank), dtype=np.float64)
            V = np.zeros((M*P, target_rank), dtype=np.float64)
            W = np.zeros((P*N, target_rank), dtype=np.float64)
            
            # Fill in a subset of schoolbook terms
            all_terms = [(i, j, k) for i in range(N) for j in range(M) for k in range(P)]
            rng.shuffle(all_terms)
            
            for t in range(target_rank):
                if t < 36:
                    # Use first 36 terms as schoolbook
                    i, j, k = all_terms[t]
                    U[i*M + j, t] = 1
                    V[j*P + k, t] = 1
                    W[k*N + i, t] = 1
                else:
                    # Extra terms: random small values
                    U[:, t] = rng.normal(0, 0.1, N*M)
                    V[:, t] = rng.normal(0, 0.1, M*P)
                    W[:, t] = rng.normal(0, 0.1, P*N)
            
            # Try to optimize: minimize ||D - T||_F^2 where D = sum_t u_t ⊗ v_t ⊗ w_t
            # Use gradient descent with a few iterations
            lr = 0.01
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Compute D = einsum('at,bt,ct->abc', U, V, W)
                D = np.einsum('at,bt,ct->abc', U, V, W)
                err = D - T
                
                # Gradients
                # dL/dU[a,t] = sum_{b,c} err[a,b,c] * V[b,t] * W[c,t]
                # dL/dV[b,t] = sum_{a,c} err[a,b,c] * U[a,t] * W[c,t]
                # dL/dW[c,t] = sum_{a,b} err[a,b,c] * U[a,t] * V[b,t]
                
                gU = np.einsum('abc,bt,ct->at', err, V, W)
                gV = np.einsum('abc,at,ct->bt', err, U, W)
                gW = np.einsum('abc,at,bt->ct', err, U, V)
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                # Check convergence
                if np.max(np.abs(err)) < 1e-10:
                    break
            
            # Round to half-integers and verify
            U_r = _round_to_half(U)
            V_r = _round_to_half(V)
            W_r = _round_to_half(W)
            
            if _verify(U_r, V_r, W_r, T):
                return U_r, V_r, W_r, target_rank
    
    return None

def _try_kronecker_extension(T, deadline):
    """
    Try to extend a known 3x3*3x3 rank-21 decomposition to 3x3*3x4.
    
    For 3x3*3x3, there are known rank-21 decompositions.
    For the 4th column, we need to add terms.
    
    Actually, let's try a different approach: use the fact that
    3x3*3x4 can be seen as 3x3*3x3 plus 3x3*3x1.
    The 3x3*3x1 part has rank 9 (schoolbook) or maybe lower.
    
    But this decomposition isn't straightforward because the tensors aren't
    simply additive in the rank sense.
    
    Let's try: assume the rank-r decomposition has a block structure where
    some terms only affect the first 3 columns of B (and corresponding C),
    and some terms handle the 4th column.
    """
    # This is complex. Let's try a simpler structured search.
    return None

def construct():
    deadline = time.time() + 130  # 130s to be safe (150s kill)
    
    T = _build_tensor()
    
    # Start with schoolbook
    U_best, V_best, W_best = _schoolbook()
    best_rank = SCHOOLBOOK_RANK
    
    # Try to find better decompositions
    result = _try_structured_merging(T, deadline)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            U_best, V_best, W_best = U, V, W
            best_rank = rank
    
    # Try Kronecker extension
    if time.time() < deadline:
        result2 = _try_kronecker_extension(T, deadline)
        if result2 is not None:
            U, V, W, rank = result2
            if rank < best_rank:
                U_best, V_best, W_best = U, V, W
                best_rank = rank
    
    # Return best result
    return _to_lists(U_best, V_best, W_best)
