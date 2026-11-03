"""Matrix multiplication tensor decomposition for 2x3 * 3x10."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W[k*n+i, r] corresponds to output position (i,k) in the result C = A*B.
    The tensor T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i,j,k.
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                total = np.dot(u_val, v_val) * 0 + np.sum(u_val * v_val * w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _is_valid_entries(U, V, W, tol=1e-9):
    """Check that all entries are integers or half-integers (multiples of 0.5)."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if val is None:
                    return False
                v = float(val)
                if abs(v * 2 - round(v * 2)) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank

def _build_strassen55(n, m, p):
    """
    Attempt to construct a rank-55 decomposition for 2x3 * 3x10.
    
    Strategy: Use the structure of Strassen's algorithm.
    2x3*3x10 can be viewed as blocks.
    
    Let's think about this differently. The tensor M_{<2,3,10>} has rank at most 60.
    
    One known approach: 
    M_{<2,3,10>} is a sub-tensor of M_{<3,4,12>} or similar.
    
    Actually, let's use the fact that:
    M_{<2,3,10>} can be decomposed using 55 terms by exploiting the small dimensions.
    
    Here's a concrete construction attempt:
    We split the 3 columns into 2+1 and the 10 rows into 5+5.
    
    Actually, let me try a direct search approach with a structured initialization.
    
    For now, return None to indicate we need to search.
    """
    return None

def _try_drop_term(U, V, W, n, m, p, rank, deadline):
    """Try to drop one term from a rank-r decomposition to get rank r-1.
    
    Given a valid rank-r decomposition, try to set one term's contribution to zero
    by adjusting the other terms. This is equivalent to checking if one of the 
    rank-1 tensors is in the span of the others.
    """
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each term t, check if we can express U[:,t] * V[:,t] * W[:,t] 
    # as a linear combination of the other terms.
    # This is hard in general. Instead, try a simpler approach:
    # Try to find a vector (a, b, c) such that setting term t to zero
    # and adjusting all other terms still gives the same tensor.
    
    # A simpler approach: try to find which term is "redundant" by checking
    # if the 3D array formed by the outer products has linearly dependent columns.
    
    # Build the tensor matrix: for each (i,j,k), the contribution is U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
    # The tensor is the sum over t of these rank-1 contributions.
    
    # Number of equations: n*m*p = 60
    # Number of unknowns per term: we can scale term t by a scalar s_t
    # So the tensor equation is: sum_t s_t * T_t = T_schoolbook
    # where T_t is the rank-1 tensor for term t.
    
    # This gives us n*m*p equations in rank unknowns (s_t).
    # If the system is underdetermined (rank < n*m*p), we might find a non-trivial solution
    # where one s_t = 0.
    
    # Build the matrix A where A[i,j,k, t] = U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    # We need A @ s = b where b[i,j,k] = 1 (the target tensor values)
    
    num_eq = n * m * p
    A = np.zeros((num_eq, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for t in range(rank):
                    A[idx, t] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                idx += 1
    
    b = np.ones(num_eq)
    
    # Solve A @ s = b. If there are multiple solutions, try to find one with a zero entry.
    # Use least squares to get the minimum norm solution, then check null space.
    try:
        s_min, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return None
    
    # Check if the solution is valid (all entries should be close to 1 for a valid decomposition)
    # Actually, for the schoolbook, s should be all 1s.
    
    # Check if the system is consistent
    if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-6:
        return None
    
    # Check rank of A
    if rank_A >= rank:
        # Full column rank, unique solution, can't drop any term
        return None
    
    # Find null space vectors
    # The null space of A gives directions in which we can modify s while keeping the tensor the same.
    # We want to find a null space vector v such that s + alpha*v has a zero entry for some alpha > 0.
    
    # Get singular vectors for zero singular values
    if rank_A < rank:
        # Null space is spanned by V[:, rank_A:]
        null_space = sv[:, rank_A:]  # This is the V^T rows corresponding to zero singular values
        # Actually, np.linalg.lstsq returns Vt where Vt[t, :] are the right singular vectors
        # The null space is spanned by Vt[rank_A:, :]
        null_basis = Vt = sv
        # Wait, sv is the right singular vectors (V^T), shape (rank, rank)
        # Null space vectors are rows of Vt[rank_A:]
        
        for ni in range(rank_A, rank):
            v = null_basis[ni, :]
            # s + alpha * v should have a zero entry
            # For each t where v[t] != 0: alpha = -s[t] / v[t]
            # We need alpha > 0 for at least one t, and s[t] + alpha*v[t] >= 0 for all t (to keep entries non-negative, though not strictly required)
            # Actually, we just need s[t] + alpha*v[t] to be valid (integer/half-integer).
            # Let's just check if we can get any s[t] + alpha*v[t] = 0 with alpha making all entries valid.
            
            for t in range(rank):
                if abs(v[t]) < 1e-12:
                    continue
                alpha = -s_min[t] / v[t]
                if alpha <= 1e-12:
                    continue
                new_s = s_min + alpha * v
                # Check if new_s has a zero entry and all entries are "reasonable" (close to integers/half-integers)
                has_zero = False
                valid = True
                for t2 in range(rank):
                    val = new_s[t2]
                    if abs(val) < 1e-8:
                        has_zero = True
                        continue
                    # Check if val is close to an integer or half-integer
                    if abs(val * 2 - round(val * 2)) > 1e-6:
                        valid = False
                        break
                    if abs(val) > 10:  # sanity check
                        valid = False
                        break
                if has_zero and valid:
                    # Construct the new decomposition
                    new_U = [list(U[i]) for i in range(n*m)]
                    new_V = [list(V[i]) for i in range(m*p)]
                    new_W = [list(W[i]) for i in range(p*n)]
                    # Scale each term
                    for t in range(rank):
                        scale = new_s[t]
                        if abs(scale) < 1e-8:
                            # This term is dropped
                            continue
                        for i in range(n*m):
                            new_U[i][t] = U[i][t] * scale
                        for i in range(m*p):
                            new_V[i][t] = V[i][t] * scale
                        for i in range(p*n):
                            new_W[i][t] = W[i][t] * scale
                    # Verify
                    if _verify(new_U, new_V, new_W, n, m, p):
                        if _is_valid_entries(new_U, new_V, new_W):
                            return new_U, new_V, new_W, rank - 1
    return None

def _random_search_rank(n, m, p, target_rank, max_iters, deadline, seed=42):
    """Random search for a decomposition of the given rank."""
    rng = random.Random(seed)
    best_result = None
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Initialize with random small integer/half-integer values
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        W = [[0.0] * target_rank for _ in range(p * n)]
        
        # Start from schoolbook and randomly zero out some entries
        U0, V0, W0, _ = _build_schoolbook(n, m, p)
        
        # Try to compress: randomly perturb and see if we can maintain validity
        # This is very hard. Let's try a different approach.
        
        # Actually, for random search to work, we need a better initialization.
        # Let's just check the deadline and return None if we haven't found anything.
        pass
    
    return best_result

def _structured_search(n, m, p, deadline):
    """Structured search for rank reduction.
    
    Try dropping terms from the schoolbook decomposition one at a time.
    """
    U, V, W, rank = _build_schoolbook(n, m, p)
    
    current_U, current_V, current_W = [list(r) for r in U], [list(r) for r in V], [list(r) for r in W]
    current_rank = rank
    
    # Try dropping one term
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        
        result = _try_drop_term(current_U, current_V, current_W, n, m, p, current_rank, deadline)
        if result is not None:
            current_U, current_V, current_W, current_rank = result
        else:
            break
    
    return current_U, current_V, current_W, current_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Start with schoolbook
    U, V, W, rank = _build_schoolbook(n, m, p)
    
    # Try structured search to reduce rank
    if time.time() < deadline - 10:
        U2, V2, W2, rank2 = _structured_search(n, m, p, deadline)
        if rank2 < rank:
            U, V, W, rank = U2, V2, W2, rank2
    
    # Return the best found
    return {"U": U, "V": V, "W": W}
