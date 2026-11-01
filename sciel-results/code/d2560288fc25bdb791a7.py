"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

_START_TIME = time.time()
_TIMEOUT = 130  # seconds, safety margin before 150s kill

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k,l] where C[i,k] = sum_j A[i,j]*B[j,k].
    The tensor has entries T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    Actually the multiplication tensor is 3-tensor: M[i,j,k] for (i in n, j in m, k in p)
    representing the coefficient of A[i,j]*B[j,k] in C[i,k].
    For a rank-r decomposition: M = sum_t U[:,t] V[:,t] W[:,t] (outer products)
    where U is (n*m, r), V is (m*p, r), W is (p*n, r).
    M[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    """
    # Flatten indices
    # U index: i*m + j  (i in [0,n), j in [0,m))
    # V index: j*p + k  (j in [0,m), k in [0,p))
    # W index: k*n + i  (k in [0,p), i in [0,n))
    # The tensor M is a 3D array of shape (n*m, m*p, p*n)
    # M[a, b, c] = 1 iff there exist i,j,k such that:
    #   a = i*m+j, b = j*p+k, c = k*n+i
    #   i.e., j = a % m, i = a // m, k = b % p, j2 = b // p, i2 = c // n, k2 = c % n
    #   and j == j2 and i == i2 and k == k2
    M = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0
    return M

def _validate(U, V, W, M, tol=1e-9):
    """Check if U, V, W decompose M: M[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]"""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Compute tensor
    # M_recon[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    # This is a 3-tensor contraction
    # Efficient: for each t, add outer product U[:,t] @ V[:,t] @ W[:,t]
    M_recon = np.zeros_like(M)
    for t in range(rank):
        M_recon += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
    return np.allclose(M_recon, M, atol=tol)

def _validate_fast(U, V, W, M):
    """Faster validation using einsum"""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    M_recon = np.einsum('at,bt,ct->abc', U, V, W)
    return np.allclose(M_recon, M, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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

def _try_strassen_like(n, m, p):
    """
    Try a Strassen-style approach.
    For 2x2 * 2x2, Strassen gives rank 7 instead of 8.
    For 2x2 * 2x9, we can think of B as 2x2 * (2x9/2) but 9 is odd.
    
    Let's try: split B into B1 (2x4) and B2 (2x5).
    A (2x2) * B1 (2x4) -> C1 (2x4), rank at most 2*2*4=16, Strassen-like could give less.
    A (2x2) * B2 (2x5) -> C2 (2x5), rank at most 2*2*5=20.
    
    Actually, let me think differently. The multiplication tensor for (2,2,9)
    has a natural structure. Let me try to find a decomposition with rank < 36.
    
    Key insight: The tensor M[i*m+j, j*p+k, k*n+i] = 1.
    For n=2, m=2, p=9:
    U has 4 rows, V has 18 rows, W has 18 rows.
    
    Let me try a structured search over small ranks with entries in {-1, 0, 1, 1/2}.
    """
    pass

def _search_structured(n, m, p, M, max_rank=35, entries=None):
    """
    Structured search: try to find a decomposition of rank r < n*m*p.
    Use a greedy/iterative approach: start with rank r, try to fill in entries.
    """
    if entries is None:
        entries = [-1, -0.5, 0, 0.5, 1]
    
    target_rank = n * m * p
    # Try ranks from target_rank-1 down
    for r in range(target_rank - 1, 15, -1):
        if _time_left() < 5:
            break
        # For each rank r, try random structured assignments
        # This is a huge search space, so use a heuristic approach
        # 
        # Approach: The tensor M has n*m*p = 36 non-zero entries.
        # Each rank-1 term U[:,t] V[:,t] W[:,t] can "cover" multiple entries.
        # A rank-1 term with U[i*m+j,t]=1, V[j'*p+k,t]=1, W[k'*n+i',t]=1
        # contributes to M[i*m+j, j'*p+k, k'*n+i'].
        # For it to be valid, we need j=j', i=i', k=k' for the contribution to M.
        # But a single rank-1 term can contribute to multiple (a,b,c) positions
        # if the indices align.
        
        # Actually, a rank-1 term U[:,t], V[:,t], W[:,t] contributes to ALL
        # (a,b,c) where U[a,t]*V[b,t]*W[c,t] != 0.
        # For the sum over t to equal M, we need the superposition to work out.
        
        # Let me try a different approach: use the fact that for small n,m,
        # we can enumerate possible "patterns".
        
        # For n=2, m=2: U has 4 rows. Each column of U is a vector in {entries}^4.
        # There are 5^4 = 625 possible column patterns for U.
        # Similarly for V (18 rows): 5^18 is too large.
        
        # Better approach: use the structure. The multiplication tensor for
        # (n,m,p) has the property that it's "separable" in a certain way.
        
        # Let me try: for each rank r, use a randomized search with validation.
        # Generate random U, V, W with entries from {entries}, validate.
        # This is very unlikely to find a solution, but let's try a smarter approach.
        
        # Smarter: use the fact that M is sparse (36 non-zeros out of 4*18*18=1296).
        # A valid decomposition must have sum_t U[a,t]*V[b,t]*W[c,t] = M[a,b,c].
        
        # Let me try a systematic approach for small ranks:
        # Fix the structure of U, V, W to have limited support.
        
        # Actually, let me just try a few hand-crafted candidates.
        pass
    
    return None

def _hand_crafted_candidates(n, m, p):
    """
    Try specific hand-crafted decompositions with rank < 36.
    
    For (2,2,9):
    The schoolbook has rank 36. Can we do better?
    
    The lower bound for matrix multiplication rank R(n,m,p) >= max(nm, mp, np) = max(4, 18, 18) = 18.
    Also R(n,m,p) >= n*m*p / max(n,m,p) ... no, that's not right.
    
    Known: R(2,2,2) = 7 (Strassen).
    For (2,2,9), we can use the (2,2,2) Strassen decomposition on blocks.
    
    Split B (2x9) into three 2x3 blocks: B = [B1 | B2 | B3].
    C = A*B = [A*B1 | A*B2 | A*B3].
    Each A*Bj is (2x2)*(2x3) = (2x3).
    R(2,2,3): lower bound max(4,6,6)=6. Schoolbook = 12.
    Can we do R(2,2,3) < 12?
    
    For (2,2,3): 
    U: 4 rows, V: 6 rows, W: 6 rows.
    M[i*2+j, j*3+k, k*2+i] = 1.
    Non-zeros: (0,0,0), (0,3,1), (0,6,2) [i=0,j=0,k=0,1,2]
    (1,1,0), (1,4,1), (1,7,2) [i=0,j=1,k=0,1,2] -- wait let me recompute
    i=0,j=0: a=0, b=0,1,2, c=0,2,4 -> M[0,0,0]=1, M[0,1,2]=1, M[0,2,4]=1
    i=0,j=1: a=1, b=3,4,5, c=0,2,4 -> M[1,3,0]=1, M[1,4,2]=1, M[1,5,4]=1
    i=1,j=0: a=2, b=0,1,2, c=1,3,5 -> M[2,0,1]=1, M[2,1,3]=1, M[2,2,5]=1
    i=1,j=1: a=3, b=3,4,5, c=1,3,5 -> M[3,3,1]=1, M[3,4,3]=1, M[3,5,5]=1
    
    So 12 non-zeros. Schoolbook rank 12.
    
    Can we do rank 11 or less for (2,2,3)?
    
    This is getting complex. Let me just try a randomized search with a time limit.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    M = _build_target_tensor(n, m, p)
    
    # Try to find a lower rank decomposition
    # Strategy: Use a structured randomized search
    # For ranks from 35 down to some minimum, try random assignments
    
    # The search space is enormous, so let's use a smarter approach.
    # 
    # Key observation: The multiplication tensor M[i*m+j, j*p+k, k*n+i] = 1
    # has a very specific structure. Each "term" in the decomposition
    # corresponds to a choice of (i,j,k) pattern.
    #
    # For a rank-1 term to contribute to M[a,b,c], we need:
    # U[a,t] * V[b,t] * W[c,t] to be non-zero.
    # The sum over all t must give M[a,b,c].
    #
    # Let me try a different structural approach:
    # Group the 36 schoolbook terms and see if some can be merged.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can potentially be
    # merged if they share structure.
    
    # For now, let's do a limited randomized search for rank 35, 34, 33, 32
    # with entries in {-1, 0, 1}
    
    np.random.seed(42)
    
    for target_r in range(best_rank - 1, max(best_rank - 8, 17), -1):
        if _time_left() < 10:
            break
        
        # Try a few random initializations
        n_tries = 20
        for trial in range(n_tries):
            if _time_left() < 5:
                break
            
            # Generate random sparse matrices with entries in {-1, 0, 1}
            # Make them sparse to increase chance of matching the sparse M
            U = np.zeros((n * m, target_r))
            V = np.zeros((m * p, target_r))
            W = np.zeros((p * n, target_r))
            
            for t in range(target_r):
                # Each column has a few non-zeros
                nnz_u = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                nnz_v = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                nnz_w = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                
                u_idx = np.random.choice(n * m, size=nnz_u, replace=False)
                v_idx = np.random.choice(m * p, size=nnz_v, replace=False)
                w_idx = np.random.choice(p * n, size=nnz_w, replace=False)
                
                U[u_idx, t] = np.random.choice([-1, 1], size=nnz_u)
                V[v_idx, t] = np.random.choice([-1, 1], size=nnz_v)
                W[w_idx, t] = np.random.choice([-1, 1], size=nnz_w)
            
            if _validate_fast(U, V, W, M):
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Convert to int where possible
                for i in range(len(U_list)):
                    for j in range(len(U_list[i])):
                        if U_list[i][j] == int(U_list[i][j]):
                            U_list[i][j] = int(U_list[i][j])
                for i in range(len(V_list)):
                    for j in range(len(V_list[i])):
                        if V_list[i][j] == int(V_list[i][j]):
                            V_list[i][j] = int(V_list[i][j])
                for i in range(len(W_list)):
                    for j in range(len(W_list[i])):
                        if W_list[i][j] == int(W_list[i][j]):
                            W_list[i][j] = int(W_list[i][j])
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no improvement found, return schoolbook
    return best
# EVOLVE-BLOCK-END
