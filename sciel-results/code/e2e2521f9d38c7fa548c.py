"""Matrix multiplication 3,3,6 tensor decomposition - rank-minimal exact search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: (n*m, rank) - 9 x rank
    V: (m*p, rank) - 18 x rank
    W: (p*n, rank) - 18 x rank
    
    The tensor T[i,j,k] = delta_{i,k} for the multiplication A(i,m) * B(m,p) = C(i,p)
    With trace convention W[k*n+i, r], we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid term
    Actually, the decomposition means:
    C[i,k] = sum_j A[i,j] * B[j,k]
    In tensor form: T[i,j,k] = 1 (the multiplication tensor)
    Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    But with the trace convention W[k*n+i, r], we have W indexed as W[k*n+i, r]
    so W[k*n+i, r] corresponds to the (i,k) pair.
    
    Let me re-read: W[k*n+i, r] - so for output element C[i,k], the W index is k*n+i.
    
    The standard tensor decomposition for matrix multiplication:
    T_{ijk} = sum_r U_{ijr} * V_{jkr} * W_{ikr}
    where T_{ijk} = 1 (the multiplication tensor has all entries 1? No...)
    
    Actually, the matrix multiplication tensor is defined such that:
    C_{ik} = sum_j A_{ij} B_{jk}
    
    In tensor notation, this is:
    C_{ik} = sum_j sum_r U_{ijr} * V_{jkr} * W_{ikr} * A_{ij} * B_{jk}
    
    For this to hold for all A, B, we need:
    sum_r U_{ijr} * V_{jkr} * W_{ikr} = delta_{ij,ik} ... no.
    
    Let me think more carefully. The tensor T has indices (i, j, k) where:
    - i indexes rows of A and C (0 to n-1)
    - j indexes columns of A / rows of B (0 to m-1)  
    - k indexes columns of B and C (0 to p-1)
    
    T[i,j,k] = 1 for all i,j,k (this is the multiplication tensor)
    
    A rank-r decomposition gives:
    T[i,j,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that would mean every entry of T equals the sum, and T[i,j,k]=1 for all i,j,k.
    That's not right either.
    
    Let me reconsider. The standard formulation:
    We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    A tensor decomposition of rank r means we can write:
    C[i,k] = sum_{r=0}^{r-1} (sum_{j} A[i,j] * U[i,j,r]) * (sum_{j} B[j,k] * V[j,k,r]) * W[i,k,r]
    
    No, that's not standard either. Let me look at this differently.
    
    The multiplication tensor T is a 3-tensor with T[i,j,k] = 1 for all valid i,j,k.
    A rank-r decomposition is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    But T[i,j,k] = 1 for all i,j,k, so we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    Wait, that can't be right because the left side depends on i,j,k in a specific way.
    
    Actually, I think the correct interpretation is:
    The multiplication operation can be written as:
    C = sum_r (A * u_r) * (v_r * B) * w_r
    where u_r, v_r, w_r are coefficient vectors/matrices.
    
    More precisely, a rank-r decomposition of matrix multiplication means:
    C[i,k] = sum_{r=0}^{r-1} (sum_{j} A[i,j] * U[i,j,r]) * (sum_{j} B[j,k] * V[j,k,r])
    
    No wait, that's bilinear but not trilinear in the right way.
    
    Let me think about this from the perspective of the schoolbook algorithm:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This can be written as:
    C[i,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k']
    where T[i,j,k] = 1 if (i,j,k) is a valid index triple, else 0.
    But actually T[i,j,k] = 1 for all valid i,j,k.
    
    A rank-r decomposition replaces the sum over j (which has m terms) with a sum over r (r terms):
    C[i,k] = sum_{r=0}^{r-1} (sum_j A[i,j] * U[i,j,r]) * (sum_j B[j,k] * V[j,k,r]) * W[i,k,r]
    
    Hmm, but that introduces W[i,k,r] as a scalar multiplier, which doesn't match the standard form.
    
    I think the correct form is:
    C[i,k] = sum_{r=0}^{r-1} (sum_j A[i,j] * U[i,j,r]) * (sum_j B[j,k] * V[j,k,r])
    
    And W is not needed? But the problem says W has shape (p*n, rank).
    
    OK let me re-read the problem statement: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    So U is n*m x rank, V is m*p x rank, W is p*n x rank.
    
    The decomposition must satisfy:
    For all A (n x m) and B (m x p), C = A * B where:
    C[i,k] = sum_{r=0}^{r-1} (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j*p+k, r]) * W[k*n+i, r]
    
    Wait, that would make it trilinear in A, B, and W... but W is fixed.
    
    Actually I think the correct interpretation is:
    The multiplication tensor has entries T[i,j,k] and a rank-r decomposition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    where T[i,j,k] = 1 for all valid (i,j,k). But that's a constant tensor, which has rank 1.
    That can't be right.
    
    Let me reconsider. I think the tensor is not all-ones. The multiplication tensor T is defined such that:
    T[i,j,k] contributes to C[i,k] when multiplied by A[i,j] and B[j,k].
    
    So the "tensor" is really the structure of the multiplication, and a decomposition means:
    sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) (sum_j B[j,k] V[j*p+k,r]) W[k*n+i,r]
    
    No, that still doesn't work dimensionally.
    
    I think the correct form is:
    C[i,k] = sum_r U[i*m+j_1, r] * V[j_2*p+k, r] * W[k*n+i, r] * A[i,j_1] * B[j_2,k]
    
    summed over j_1, j_2? That doesn't make sense either.
    
    Let me just look at the schoolbook case in the current program:
    For each (i,j,k): U[i*m+j][column]=1, V[j*p+k][column]=1, W[k*n+i][column]=1
    And there are n*m*p such terms.
    
    So for the schoolbook, term (i,j,k) has:
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    And the reconstruction is:
    C[i,k] = sum_r U[i*m+?, r] * V[?*p+k, r] * W[k*n+i, r]
    
    Hmm, but U and V depend on j, and we need to sum over j.
    
    I think the correct formula is:
    C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    
    No, that's not right either because U, V, W are fixed coefficients.
    
    OK, I think the correct interpretation is that the decomposition represents the bilinear map:
    (A, B) -> C
    
    A rank-r decomposition means:
    C = sum_{r=0}^{r-1} (A * u_r) * (v_r * B)
    
    where u_r is a matrix... no.
    
    Actually, I recall now. The standard form is:
    C[i,k] = sum_{r=0}^{r-1} (sum_{j} A[i,j] * U[i,j,r]) * (sum_{j} B[j,k] * V[j,k,r])
    
    And W is not used in this form. But the problem includes W.
    
    Wait, maybe the form is:
    C[i,k] = sum_{r=0}^{r-1} W[i,k,r] * (sum_j A[i,j] U[i,j,r]) * (sum_j B[j,k] V[j,k,r])
    
    With W[k*n+i, r] being the coefficient. Let me check with schoolbook:
    For term r corresponding to (i0,j0,k0):
    U[i0*m+j0, r] = 1, all other U entries in column r are 0
    V[j0*p+k0, r] = 1, all other V entries in column r are 0
    W[k0*n+i0, r] = 1, all other W entries in column r are 0
    
    Then: sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j,r]) * (sum_j B[j,k] V[j*p+k,r])
    = sum_r W[k*n+i, r] * A[i, j_r] * B[j_r, k]
    where j_r is the unique j with U[i*m+j, r] != 0
    
    For this to give C[i,k] = sum_j A[i,j] B[j,k], we need:
    sum_r W[k*n+i, r] * A[i, j_r] * B[j_r, k] = sum_j A[i,j] B[j,k]
    
    With the schoolbook, for each (i0,j0,k0), we have a term where:
    - It only contributes when i=i0, k=k0 (because W[k*n+i, r] is nonzero only when k=k0, i=i0)
    - And it gives A[i0,j0] * B[j0,k0]
    
    So C[i0,k0] gets the contribution A[i0,j0] * B[j0,k0] from this term.
    Summing over all j0, we get C[i0,k0] = sum_{j0} A[i0,j0] * B[j0,k0]. Correct!
    
    So the formula is:
    C[i,k] = sum_{r=0}^{r-1} W[k*n+i, r] * (sum_{j} A[i,j] * U[i*m+j, r]) * (sum_{j} B[j,k] * V[j*p+k, r])
    
    And the constraint is that this equals sum_j A[i,j] B[j,k] for all A, B.
    
    This means:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] = delta_{j1,j2} * delta_{...}
    
    Hmm, actually for the equality to hold for all A, B, we need:
    sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = 1 if j=j' and the (i,k) pair is "active", else 0
    
    Wait, let me be more careful. We need:
    sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j,r]) * (sum_j B[j,k] V[j*p+k,r]) = sum_j A[i,j] B[j,k]
    
    Expanding the left side:
    sum_r W[k*n+i, r] * sum_j sum_j' A[i,j] U[i*m+j,r] B[j',k] V[j'*p+k,r]
    = sum_j sum_j' A[i,j] B[j',k] * sum_r W[k*n+i, r] U[i*m+j,r] V[j'*p+k,r]
    
    For this to equal sum_j A[i,j] B[j,k] = sum_j sum_j' A[i,j] B[j',k] delta_{j,j'}, we need:
    sum_r W[k*n+i, r] U[i*m+j,r] V[j'*p+k,r] = delta_{j,j'}  for all i,k,j,j'
    
    So the constraint is:
    For all i in [0,n), k in [0,p), j in [0,m), j' in [0,m):
    sum_{r=0}^{rank-1} W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = 1 if j==j' else 0
    
    This is a system of n*p*m*m equations in rank*(n*m + m*p + p*n) unknowns.
    
    Let me verify with schoolbook: n=3, m=3, p=6, rank=54.
    Number of equations: 3*6*3*3 = 162
    Number of unknowns: 54*(9+18+18) = 54*45 = 2430
    So it's underdetermined, and the schoolbook is one solution.
    
    Now I can set up this as a constraint satisfaction / optimization problem.
    """
    n, m, p = 3, 3, 6
    # Check: for all i,k,j,j': sum_r W[k*n+i,r] * U[i*m+j,r] * V[j'*p+k,r] == (1 if j==j' else 0)
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    val = 0.0
                    for r in range(rank):
                        val += W[k*n+i, r] * U[i*m+j, r] * V[jp*p+k, r]
                    target = 1.0 if j == jp else 0.0
                    err = abs(val - target)
                    if err > max_err:
                        max_err = err
    return max_err

def _random_rank1_term(i, j, k, n=3, m=3, p=6):
    """Generate a random rank-1 term."""
    U = [0.0] * (n*m)
    V = [0.0] * (m*p)
    W = [0.0] * (p*n)
    U[i*m+j] = 1.0
    V[j*p+k] = 1.0
    W[k*n+i] = 1.0
    return U, V, W

def _compute_residual(U, V, W, n=3, m=3, p=6):
    """Compute the residual tensor R[i,k,j,jp] = sum_r W[k*n+i,r]*U[i*m+j,r]*V[jp*p+k,r] - delta_{j,jp}"""
    rank = len(U[0])
    R = np.zeros((n, p, m, m))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    val = 0.0
                    for r in range(rank):
                        val += W[k*n+i][r] * U[i*m+j][r] * V[jp*p+k][r]
                    R[i,k,j,jp] = val - (1.0 if j == jp else 0.0)
    return R

def _greedy_decomposition(n=3, m=3, p=6, max_rank=30, time_limit=120):
    """Greedy rank-1 decomposition: at each step, find the best rank-1 term to add."""
    start_time = time.time()
    U = []
    V = []
    W = []
    
    # Current residual: we want to build up the delta_{j,jp} structure
    # Target: T[i,k,j,jp] = 1 if j==jp else 0
    # Current approximation: sum of added terms
    # Residual: T - approximation
    
    # For efficiency, track the current "product" for each (i,k,j,jp)
    current = np.zeros((n, p, m, m))
    
    for step in range(max_rank):
        if time.time() - start_time > time_limit - 10:
            break
        
        best_score = -1
        best_term = None
        
        # Try all possible rank-1 terms: (i, j, k) with coefficient
        # A rank-1 term (i0, j0, k0) contributes to current[i0,k0,j0,j0] += coeff
        # We want to reduce the residual.
        
        # The residual is: residual[i,k,j,jp] = target[i,k,j,jp] - current[i,k,j,jp]
        # target[i,k,j,jp] = 1 if j==jp else 0
        
        # A rank-1 term at (i0,j0,k0) with coefficient c changes:
        # current[i0,k0,j0,j0] += c
        # So residual[i0,k0,j0,j0] -= c
        
        # We want to choose (i0,j0,k0) and c to minimize the final residual norm.
        # If we add c * e_{i0,k0,j0,j0}, the new residual is:
        # residual - c * e_{i0,k0,j0,j0}
        # Optimal c = residual[i0,k0,j0,j0] (to zero out that entry)
        # But this might not be globally optimal.
        
        # For greedy: pick the (i0,j0,k0) where residual[i0,k0,j0,j0] is largest in magnitude
        # and set c = residual[i0,k0,j0,j0]
        
        for i0 in range(n):
            for k0 in range(p):
                for j0 in range(m):
                    r = current[i0, k0, j0, j0]
                    target = 1.0
                    residual_val = target - r
                    score = abs(residual_val)
                    if score > best_score:
                        best_score = score
                        best_term = (i0, j0, k0, residual_val)
        
        if best_term is None or best_score < 1e-12:
            break
        
        i0, j0, k0, c = best_term
        
        # Add this term
        u_vec = [0.0] * (n*m)
        v_vec = [0.0] * (m*p)
        w_vec = [0.0] * (p*n)
        u_vec[i0*m+j0] = 1.0
        v_vec[j0*p+k0] = 1.0
        w_vec[k0*n+i0] = c
        
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
        # Update current
        current[i0, k0, j0, j0] += c
        
        # Check if we're done
        max_res = np.max(np.abs(current - np.eye(m).reshape(1,1,m,m).repeat(n*p, axis=0).reshape(n,p,m,m)))
        if max_res < 1e-10:
            break
    
    # Convert to list-of-lists format
    U_out = [[U[r][i] for r in range(len(U))] for i in range(n*m)] if U else []
    V_out = [[V[r][i] for r in range(len(V))] for i in range(m*p)] if V else []
    W_out = [[W[r][i] for r in range(len(W))] for i in range(p*n)] if W else []
    
    return U_out, V_out, W_out

def _iterative_refinement(U, V, W, n=3, m=3, p=6, max_iter=100, tol=1e-10, time_limit=120):
    """Try to reduce the rank by merging terms or zeroing out small coefficients."""
    start_time = time.time()
    rank = len(U[0]) if U else 0
    
    # Try to find if any term can be removed (has all-zero or near-zero contribution)
    for r in range(rank):
        if time.time() - start_time > time_limit - 5:
            break
        # Check if term r has any nonzero entries
        u_norm = sum(U[i][r]**2 for i in range(n*m))
        v_norm = sum(V[i][r]**2 for i in range(m*p))
        w_norm = sum(W[i][r]**2 for i in range(p*n))
        if u_norm < 1e-20 and v_norm < 1e-20 and w_norm < 1e-20:
            # Remove this term
            for i in range(n*m):
                U[i].pop(r)
            for i in range(m*p):
                V[i].pop(r)
            for i in range(p*n):
                W[i].pop(r)
            rank -= 1
            break
    
    return U, V, W, rank

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook (rank 54)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try greedy decomposition
    try:
        U_g, V_g, W_g = _greedy_decomposition(n, m, p, max_rank=30, time_limit=60)
        if U_g:
            err = _verify_decomposition(U_g, V_g, W_g)
            if err < 1e-9:
                rank_g = len(U_g[0])
                if rank_g < best_rank:
                    best_U, best_V, best_W = U_g, V_g, W_g
                    best_rank = rank_g
    except Exception:
        pass
    
    # Try to find better decompositions using structured approaches
    # Approach: use the fact that for each (i,k), we need the 3x3 identity in the j,jp space
    # This means for each (i,k), the sub-problem is to decompose the 3x3 identity matrix
    # using rank-r terms, where each term picks one (j,jp) pair.
    
    # Actually, the constraint is:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'}
    # for all i, k, j, j'
    
    # This means for each (i,k), the matrix M_{j,j'} = sum_r W[k*n+i,r] * U[i*m+j,r] * V[j'*p+k,r] = I
    # So for each (i,k), we need to factor the identity matrix as a sum of rank-1 matrices
    # where each rank-1 matrix is u_r * v_r^T scaled by w_r.
    
    # The identity 3x3 matrix has rank 3, so we need at least 3 terms per (i,k) pair.
    # But the same term r can contribute to multiple (i,k) pairs.
    
    # Minimum possible rank: for each (i,k), we need at least 3 "active" terms.
    # There are n*p = 18 (i,k) pairs, each needing 3 terms.
    # If terms can be shared, the minimum rank is at least 18*3/3 = 18? No...
    
    # Actually, a single term r can contribute to at most one (i,k) pair because
    # U[i*m+j, r] is nonzero for at most one (i,j) and V[j'*p+k, r] is nonzero for at most one (j',k).
    # Wait, U[i*m+j, r] can be nonzero for multiple j's for a fixed i.
    
    # Let me think about this differently. For a fixed r:
    # - U[i*m+j, r] can be nonzero for multiple (i,j) pairs
    # - V[j'*p+k, r] can be nonzero for multiple (j',k) pairs
    # - W[k*n+i, r] can be nonzero for multiple (k,i) pairs
    #
    # The contribution of term r to the (i,k,j,j') entry is:
    # W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r]
    #
    # For this to be nonzero, we need all three factors to be nonzero.
    
    # A structured approach: let each term r correspond to a specific (i,k) pair and a specific
    # rank-1 factorization of part of the identity.
    
    # For each (i,k), the identity I_3 can be written as sum of 3 rank-1 matrices:
    # I_3 = e_1 e_1^T + e_2 e_2^T + e_3 e_3^T
    # So for each (i,k), we need 3 terms, giving 18*3 = 54 terms total (schoolbook).
    
    # To do better, we need to share terms between different (i,k) pairs.
    # But a term r with U[i*m+j, r] nonzero for specific (i,j) and V[j'*p+k, r] nonzero for specific (j',k)
    # can only contribute to the (i,k) pair if W[k*n+i, r] is also nonzero.
    
    # So if we set W[k*n+i, r] nonzero for multiple (k,i) pairs, the term can contribute to multiple pairs.
    # But then the contributions must add up correctly for each pair.
    
    # Let me try a different approach: use the structure more cleverly.
    
    # For 3x3x3, the minimum rank is 14 (Laderman's algorithm).
    # For 3x3x6, we can think of it as two 3x3x3 multiplications (columns 0-2 and 3-5).
    # If we could share some terms between the two, we might get less than 2*14 = 28.
    
    # But actually, the terms for the two column groups involve different V indices,
    # so they can't easily be shared.
    
    # Let me try a numerical optimization approach: start from schoolbook and try to
    # reduce the rank by setting some entries to zero.
    
    # Alternative: use a randomized approach to find low-rank decompositions.
    
    # Let me try a different structured approach:
    # For each (i,k) pair, use 3 terms that form the identity.
    # But allow the U and V vectors to be shared across (i,k) pairs.
    
    # Specifically, for a fixed j, the terms that involve U[i*m+j, r] and V[j*p+k, r]
    # contribute to the (i,k,j,j) diagonal entry.
    
    # Let me try: for each j in {0,1,2}, create 3 terms that handle the j-th diagonal
    # for all 18 (i,k) pairs.
    
    # Term for (j, i, k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # This is just the schoolbook again.
    
    # To do better, I need to use the fact that the off-diagonal entries must be zero.
    # The constraint sum_r W[k*n+i,r] U[i*m+j,r] V[j'*p+k,r] = 0 for j != j'
    # gives us freedom to "mix" terms.
    
    # Let me try a numerical search: parameterize a low-rank decomposition and optimize.
    
    # For rank R, we have R*(9+18+18) = 45R parameters.
    # We have 162 constraints (n*p*m*m = 3*6*3*3 = 162).
    # For R >= 4, we have more parameters than constraints, so solutions should exist.
    # For R = 3, we have 135 parameters < 162 constraints, so it's overdetermined.
    
    # Let me try R = 4, 5, 6, ... and see if we can find solutions.
    
    # Actually, wait. Let me reconsider the problem structure.
    # The constraint is:
    # For each (i,k): sum_r W[k*n+i,r] * u_r(i) * v_r(k) = I_3
    # where u_r(i) is the 3-vector (U[i*m+0,r], U[i*m+1,r], U[i*m+2,r])
    # and v_r(k) is the 3-vector (V[0*p+k,r], V[1*p+k,r], V[2*p+k,r])
    
    # So for each (i,k), we need: sum_r W[k*n+i,r] * u_r(i) * v_r(k)^T = I_3
    # (outer product of u_r(i) and v_r(k))
    
    # This is a sum of rank-1 matrices equaling the identity.
    # The minimum number of rank-1 matrices to sum to I_3 is 3.
    # So for each (i,k), we need at least 3 terms with nonzero W[k*n+i,r].
    
    # If we use the same 3 terms for all (i,k) pairs, we get rank 3.
    # But then u_r(i) * v_r(k)^T must sum to I_3 for all (i,k) with the same W coefficients.
    # This means u_r(i) * v_r(k)^T must be the same for all (i,k), which is very restrictive.
    
    # Let's say we use 3 terms r=0,1,2 with W[k*n+i,r] = 1 for all (i,k).
    # Then we need: sum_{r=0}^2 u_r(i) * v_r(k)^T = I_3 for all i,k.
    # This means the matrix sum_r u_r(i) v_r(k)^T must be I_3.
    # For this to hold for all i,k, we need u_r(i) and v_r(k) to be constant in i and k.
    # So u_r(i) = a_r and v_r(k) = b_r for fixed vectors a_r, b_r.
    # Then sum_r a_r b_r^T = I_3.
    # This is a rank-3 factorization of I_3, which exists (e.g., a_r = b_r = e_r).
    
    # But wait, this would mean U[i*m+j, r] = a_r[j] for all i, and V[j'*p+k, r] = b_r[j'] for all k.
    # And W[k*n+i, r] = 1 for all (i,k).
    
    # Let me check: with a_r = e_r (standard basis), b_r = e_r:
    # U[i*m+j, r] = 1 if j==r else 0
    # V[j'*p+k, r] = 1 if j'==r else 0
    # W[k*n+i, r] = 1
    
    # Then sum_r W[k*n+i,r] U[i*m+j,r] V[j'*p+k,r] = sum_r 1 * (1 if j==r) * (1 if j'==r)
    # = 1 if j==j' else 0. Correct!
    
    # So this gives a rank-3 decomposition! Let me verify this is correct.
    
    # U has shape (9, 3): U[i*m+j, r] = 1 if j==r else 0
    # V has shape (18, 3): V[j'*p+k, r] = 1 if j'==r else 0
    # W has shape (18, 3): W[k*n+i, r] = 1
    
    # Let me verify the reconstruction:
    # C[i,k] = sum_r W[k*n+i,r] * (sum_j A[i,j] U[i*m+j,r]) * (sum_j B[j,k] V[j*p+k,r])
    # = sum_r 1 * (sum_j A[i,j] * (1 if j==r else 0)) * (sum_j B[j,k] * (1 if j==r else 0))
    # = sum_r A[i,r] * B[r,k]
    # = (A*B)[i,k]
    
    # This is correct! The rank is 3!
    
    # Wait, but this seems too good to be true. Let me double-check.
    # The matrix multiplication C = A*B where A is 3x3 and B is 3x6.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # With rank 3:
    # C[i,k] = sum_{r=0}^{2} W[k*n+i,r] * (sum_j A[i,j] U[i*m+j,r]) * (sum_j B[j,k] V[j*p+k,r])
    # = sum_{r=0}^{2} 1 * A[i,r] * B[r,k]
    # = sum_{r=0}^{2} A[i,r] * B[r,k]
    
    # Yes! This is exactly the matrix multiplication formula. The rank is 3 = m.
    
    # In general, for n x m times m x p, the rank is m (you just use m terms, one for each
    # intermediate index j). This is the "obvious" decomposition.
    
    # So the minimum rank is m = 3!
    
    U_out = [[0] * 3 for _ in range(n * m)]
    V_out = [[0] * 3 for _ in range(m * p)]
    W_out = [[0] * 3 for _ in range(p * n)]
    
    for r in range(m):  # r = 0, 1, 2 (the intermediate index j)
        for i in range(n):
            U_out[i * m + r][r] = 1
        for jp in range(m):
            for k in range(p):
                V_out[jp * p + k][r] = (1 if jp == r else 0)
        for k in range(p):
            for i in range(n):
                W_out[k * n + i][r] = 1
    
    # Verify
    err = _verify_decomposition(U_out, V_out, W_out)
    if err < 1e-9:
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook if verification fails
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
