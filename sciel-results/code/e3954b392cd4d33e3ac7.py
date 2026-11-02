"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    We use a structured search: exploit the Kronecker/product structure.
    
    The tensor for (n,m,p) = (2,2,8) has size (n*m, m*p, p*n) = (4, 16, 16).
    
    Key insight: The 2x2 * 2x8 multiplication can be decomposed by treating
    the 2x8 matrix as two 2x4 blocks or by exploiting the small left matrix.
    
    For A (2x2) * B (2x8):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Since n=2 is small, we can try rank-deficient decompositions.
    
    Strategy: Use the fact that for fixed i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    This is a linear combination of the two rows of B with coefficients from row i of A.
    
    The tensor T[i,j,k] = delta(i, row of A) * delta(j, col of A) ... 
    Actually, the matmul tensor M_{(i,j),(j',k)} where the contraction is over j.
    M[i,j; j',k] = 1 if j==j' else 0, and this gives C[i,k] = sum_j A[i,j] B[j,k].
    
    So the tensor is: T[(i,j),(j',k)] = delta_{j,j'}
    where (i,j) indexes the 4 inputs of A, (j',k) indexes the 16 inputs of B.
    
    Rank of this tensor: The tensor is essentially the identity in the j/j' space
    tensor with identity in i/k... no, let's think more carefully.
    
    T is a 4 x 16 x 16 tensor. T[(i,j),(j',k)] = 1 if j==j' else 0.
    
    We can write T = sum_{j=0}^{1} (e_i * e_j) (e_j * e_k) ... but e_i is 2-dim, e_j is 2-dim, e_k is 8-dim.
    
    Actually: T[(i,j),(j',k)] = delta_{j,j'}
    
    A rank-1 decomposition term would be u[(i,j)] * v[(j',k)] * w[(k,i)] ... wait, the output is indexed by (k,i) with W[k*n+i, r].
    
    Output C[i,k] = sum_r U[i*m+j... no. Let me re-read.
    
    U has shape (n*m, rank) = (4, rank). Index: i*m+j.
    V has shape (m*p, rank) = (16, rank). Index: j*p+k.
    W has shape (p*n, rank) = (16, rank). Index: k*n+i.
    
    C[i,k] = sum_{j} sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    We need this to equal sum_j A[i,j] * B[j,k].
    
    So for each (i,k), sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    This must hold for all A, B. So we need:
    For each (i,k), the bilinear form in (A, B) must match.
    
    A[i,j] B[j,k] is a specific term. The tensor T[i,k; i',j'; j',k'] ... this is getting complex.
    
    Let me just use the known result: the tensor for multiplying an m x n matrix by an n x p matrix has rank at most m*n*p (schoolbook) and at least max(m*n, n*p, m*p) = max(4, 16, 16) = 16.
    
    For (2,2,8): lower bound is max(2*2, 2*8, 2*8) = max(4, 16, 16) = 16.
    
    Can we achieve rank 16? Let's try.
    
    Actually, for rectangular matrix multiplication, the rank is known to be m*n*p in general, but for small cases there can be savings.
    
    Let me try a direct construction for rank 16.
    
    We need: sum_j A[i,j] B[j,k] = sum_r U[i*2+j, r] V[j*8+k, r] W[k*2+i, r]
    
    For this to hold for all A, B, we need the tensor identity. Let's think of it as:
    For each (i,k), the coefficient of A[i',j'] B[j'',k'] in the RHS must equal delta_{i,i'} delta_{k,k'} delta_{j',j''}.
    
    RHS coefficient of A[i',j'] B[j'',k'] = sum_r U[i'*2+j', r] V[j''*8+k', r] W[k'*2+i', r]
    
    Wait, that's not right either. Let me be more careful.
    
    C[i,k] = sum_j sum_r U[i*2+j, r] * V[j*8+k, r] * W[k*2+i, r]
    
    This is a sum over j and r. For this to equal sum_j A[i,j] B[j,k] for all A, B:
    
    The term A[i,j] B[j,k] appears. So we need:
    sum_j sum_r U[i*2+j, r] V[j*8+k, r] W[k*2+i, r] = sum_j A[i,j] B[j,k]
    
    For arbitrary A, B, this means the bilinear form must match. Specifically, for each pair (j, j'), the coefficient of A[i,j'] B[j',k] must be delta_{j,j'}.
    
    Hmm, this is a 3-way tensor equation. Let me just try to construct it.
    
    For rank 16, let's try:
    For each (i, k) pair... no, that's 16 pairs but we also need to handle the j dimension.
    
    Actually, let's try: for each j in {0,1} and each k in {0,...,7}, define a rank-1 term.
    That's 2*8 = 16 terms.
    
    Term (j, k): 
    U[i*2+j, (j,k)] = delta_{i, i0} for some i0? No, we need it to work for both i=0 and i=1.
    
    Let me try: U[i*2+j, r] where r indexes (j', k') with j' in {0,1}, k' in {0..7}.
    Set U[i*2+j, (j',k')] = delta_{j,j'} * something depending on i?
    
    If U[i*2+j, (j',k')] = delta_{j,j'} * a_i, then:
    sum_j sum_{j',k'} delta_{j,j'} a_i V[j'*8+k, (j',k')] W[k*2+i, (j',k')]
    = sum_j a_i V[j*8+k, (j,k)] W[k*2+i, (j,k)]
    
    We need this to equal sum_j A[i,j] B[j,k]. So we need a_i V[j*8+k, (j,k)] W[k*2+i, (j,k)] = A[i,j] B[j,k].
    
    This requires a_i to depend on A[i,j], which it can't since U is fixed. So this doesn't work directly.
    
    The issue is that U can't depend on A. So we need a different structure.
    
    Let's try: U[i*2+j, r] = delta_{j,j'} for r=(i', j', k') with i'=i? No, U doesn't depend on i in the index r.
    
    OK here's a cleaner approach. The tensor is T_{(i,j),(j',k),(k',i')} = delta_{j,j'} delta_{i,i'} delta_{k,k'}.
    
    We want T = sum_r u_r * v_r * w_r where u_r is a vector of length 4, v_r length 16, w_r length 16.
    
    T_{(i,j),(j',k),(k',i')} = sum_r u_r[i,j] v_r[j',k] w_r[k',i']
    
    And we need this = delta_{j,j'} delta_{i,i'} delta_{k,k'}.
    
    Let's try rank = 16 with terms indexed by (j, i') or (j', i) etc.
    
    Try: terms indexed by (j, i) for j in {0,1}, i in {0,1}... that's only 4 terms, too few.
    
    Try terms indexed by (j, k) for j in {0,1}, k in {0..7}: 16 terms.
    u_r[i,j'] = delta_{j',j} * c_{i} ... but c_i can't depend on i in a way that helps.
    
    Actually, let's just try: u_r[i,j'] = delta_{j',j}, v_r[j'',k'] = delta_{j'',j} delta_{k',k}, w_r[k'',i''] = delta_{k'',k} delta_{i'',i}.
    
    Then sum_r u_r[i,j'] v_r[j'',k'] w_r[k'',i''] = sum_{j,k} delta_{j',j} delta_{j'',j} delta_{k',k} delta_{k'',k} delta_{i'',i}
    = delta_{j',j''} delta_{k',k''} delta_{i'',i} (summed over j,k, but only one term survives for given j',k')
    = delta_{j',j''} delta_{k',k''} delta_{i'',i}
    
    But we need delta_{j,j'} delta_{i,i'} delta_{k,k'}. This gives delta_{j',j''} delta_{k',k''} delta_{i'',i} which is not the same.
    
    The indices don't match up. Let me re-examine what the tensor should be.
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    So T should satisfy: C[i,k] = sum_{i',j'} sum_{j'',k'} sum_{k'',i''} T[(i',j'),(j'',k''),(k'',i'')] A[i',j'] B[j'',k'']
    
    And we need C[i,k] = sum_j A[i,j] B[j,k].
    
    So T[(i',j'),(j'',k''),(k'',i'')] must be such that summing gives delta_{i,i'} delta_{k,k''} delta_{j',j''} ... 
    
    Actually: sum_{i',j',j'',k'',k'',i''} T[...] A[i',j'] B[j'',k''] = C[i,k] = sum_j A[i,j] B[j,k]
    
    So T[(i',j'),(j'',k''),(k'',i'')] = delta_{i,i'} delta_{j',j''} delta_{k'',k} ... but k is the output index, not an input.
    
    I think I'm overcomplicating this. Let me just use the schoolbook and try to find a better rank via search or known results.
    
    For 2x2 * 2x8, the rank is known to be at most 16 (schoolbook gives 32). Let me try to construct rank 16.
    
    Actually, let me just try a simple search with random initialization and check validity.
    """
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    nm = n * m  # 4
    mp = m * p  # 16
    pn = p * n  # 16
    
    # Schoolbook rank
    rank_school = n * m * p  # 32
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(nm)]
        V = [[0] * rank_school for _ in range(mp)]
        W = [[0] * rank_school for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = make_schoolbook()
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Use random search with validation
    
    def validate(U, V, W, rank):
        """Check if the decomposition is exact."""
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This must equal sum_j A[i,j] B[j,k] for all A, B.
        # Test with random matrices
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_actual = np.zeros((n, p))
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            for i in range(n):
                for k in range(p):
                    total = 0.0
                    for j in range(m):
                        for r in range(rank):
                            total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    C_actual[i, k] = total
            if not np.allclose(C_actual, C_expected, atol=1e-9):
                return False
        return True
    
    # Try rank 16 with structured approach
    # For 2x2 * 2x8, try to use the structure
    
    # Attempt: use the fact that n=2 is small
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # We can try to represent this with rank 16 by having terms for each (j, k) pair
    # but we need to handle the i dimension too.
    
    # Let's try: for each (i, j, k), we have a term, but that's 32.
    # To get 16, we need to share some structure.
    
    # Idea: For fixed j, the contribution is A[i,j] * B[j,k]
    # We can factor A[i,j] * B[j,k] using a rank-1 in (i,k) space for each j.
    # But A[i,j] depends on i and j, B[j,k] depends on j and k.
    
    # For j=0: A[i,0]*B[0,k] for i=0,1 and k=0..7
    # This is a 2x8 matrix A_col0 * B_row0 which is an outer product.
    # Similarly for j=1.
    
    # So C = A[:,0] * B[0,:] + A[:,1] * B[1,:]
    # This is a sum of two outer products, each of rank 1 in the (i,k) space.
    # But the tensor rank is different from matrix rank.
    
    # The tensor for this specific structure might have lower rank.
    
    # Let me just try a numerical approach: random search for rank 16, 20, 24, 28.
    
    for target_rank in [16, 18, 20, 22, 24, 26, 28, 30]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization with small values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Convert to lists of floats (will check if they can be rounded to integers/halves)
            # First check if this random guess is close to valid
            # We need exact decomposition, so let's check with rational rounding
            
            # Try rounding to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            U_list = [[float(x) for x in row] for row in U_rounded]
            V_list = [[float(x) for x in row] for row in V_rounded]
            W_list = [[float(x) for x in row] for row in W_rounded]
            
            if validate(U_list, V_list, W_list, target_rank):
                if target_rank < best_rank:
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    if best_rank == rank_school:
        return best
    
    return best
# EVOLVE-BLOCK-END
