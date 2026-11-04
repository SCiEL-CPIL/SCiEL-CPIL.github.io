"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
from fractions import Fraction
from itertools import combinations
import time

# EVOLVE-BLOCK-START

def _matmul_tensor_matricization(n, m, p):
    """
    Build the matricization of the matrix multiplication tensor.
    The tensor T_{i,j,k} is nonzero (=1) when multiplying A[i,j] * B[j,k] gives C[i,k].
    Matricize: reshape T into a matrix of shape (n*p, m*p*n) by grouping first and second indices.
    Actually, for matrix multiplication, the tensor T: I x J x K -> (1,0,0) at positions (i,k,j)
    where the multiplication A[i,j]*B[j,k] contributes to C[i,k].
    
    Standard: T is a 3-tensor with T[i,k,j] = delta(i,k,j) meaning:
      C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor T_{i,k,j} is 1 if and only if this term exists, which is always for valid i,j,k.
    
    Actually, the matrix multiplication tensor has entries T_{i,j,k} where:
      C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    We can view this as a tensor M with shape (n*p, m, m) or similar. Let me be precise.
    
    The matricization for rank: We have a tensor T of dimensions (n, m, p) where
    T[i,j,k] contributes to the multiplication. Actually for 2x5x5:
    - n=2 rows of A
    - m=5 columns of A / rows of B  
    - p=5 columns of B
    
    The tensor is T[i,j,k] = A[i,j] * B[j,k] contributing to C[i,k].
    But as a bilinear map: T maps (A,B) -> C.
    
    For rank computation, matricize along one mode.
    Mode-1 unfolding (fix k, vary i and j): shape (n*m, p)
    T[j, k] for fixed k: entry at row (i*m+j), col k = delta(i,j,k)
    
    Actually the multiplication tensor M[n,m,p] has entries:
    M[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    But the tensor itself as a trilinear form: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the 
    coefficient is 1. Wait, that's not right either.
    
    Let me think differently. The matrix multiplication tensor for n x m by m x p:
    It's a 3-tensor of shape (n*m, m, m)? No.
    
    Standard formulation: T is a tensor in R^{n} ⊗ R^{p} ⊗ R^{m} (or some permutation)
    where T_{i,k,j} = 1 for all valid i,j,k? No, that gives rank n*p which is wrong.
    
    Actually: T_{i,k,j} is the coefficient such that C_{i,k} = sum_j A_{i,j} B_{j,k} T_{i,k,j}
    Since C_{i,k} = sum_j A_{i,j} B_{j,k} * 1, we have T_{i,k,j} = 1 for all i,j,k.
    But that's just a rank-1 tensor, which is clearly wrong.
    
    I'm confusing myself. Let me use the standard matricization.
    
    For matrix multiplication M_{n,m,p}:
    The tensor is in R^{n*p} ⊗ R^{m} ⊗ R^{m} (one valid matricization).
    Actually, let's use: T reshaped to matrix of shape (n*p, m*p) by flattening (i,k) as row index and j as... 
    
    I'll just build the (n*p) x (m*m) matricization where:
    Row index: i*p + k (output C[i,k])
    Col index: j1 * m + j2? No...
    
    Let me use the standard: the matrix multiplication tensor T has a mode-1 matricization
    of shape (n*p, m*m) where:
    M[row, col] with row = i*p + k, col = j*p + l? 
    
    I think the cleanest: T is a tensor of shape (n, p, m) where T[i,k,j] = 1 iff the term 
    A[i,j]*B[j,k] contributes to C[i,k]. Since every valid combination contributes, T[i,k,j]=1 
    for all i,k,j. This tensor has n*p*m entries, all 1, and its rank is... well it's just 
    the all-ones tensor which has rank 1. That's clearly not the multiplication tensor.
    
    OK I need to be very precise. The matrix multiplication tensor is:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^{(A)} ⊗ e_j^{(B)} ⊗ e_k^{(C)}
    No wait.
    
    T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} such that 
    <T, A⊗B> = C
    
    So T_{i,j; j,k; i,k} = 1 for valid indices. Flattening:
    - First mode: (i,j) with n*m elements
    - Second mode: (j,k) with m*p elements  
    - Third mode: (i,k) with n*p elements
    
    Matricization along third mode (fix the output, vary inputs):
    Shape: (n*m * m*p, n*p) = (m² n p, n p)
    
    But actually for rank, any full matricization works. Let me use the matricization
    where we fix the output index (i,k) and vary the input indices.
    
    Let's build matrix M of shape (m*p, n*p) where:
    M[j, i*k'] ... hmm.
    
    Actually simplest: the tensor T can be viewed as a linear map from R^{n*m} × R^{m*p} to R^{n*p}.
    For rank, matricize as a (n*p) × (n*m * m*p) matrix or equivalently check rank of 
    the (n*p × m*p) matrix for each fixed... 
    
    You know what, for this small case (2,5,5), let me just directly construct the matricization
    and compute its rank using exact arithmetic.
    """
    # Build the matricization: T as a matrix of shape (m*p, n*p) is not right either.
    # 
    # Correct approach: T_{α,β,γ} where α=(i,j)∈[0,nm), β=(j,k)∈[0,mp), γ=(i,k)∈[0,np)
    # T_{α,β,γ} = 1 if α=j1 matches β's j2 and α's i matches γ's i1 and β's k matches γ's k2
    # i.e., T_{(i1,j),(j,k),(i2,k2)} = δ(i1,i2)*δ(j,j)*δ(k,k2)
    # So T_{α,β,γ}=1 iff α[0]=γ[0] and α[1]=β[0] and β[1]=γ[1]
    
    # Matricize along γ (third mode): matrix of shape (nm*mp, np)
    # Or more useful: for each output γ, we have a vector of which (α,β) pairs contribute.
    
    # Let me build the (np) × (nm*mp) matrix where row γ, column (α,β) = T_{α,β,γ}
    # This matrix has rank = tensor rank (for this matricization).
    
    np_dim = n * p
    nm_dim = n * m
    mp_dim = m * p
    total_input = nm_dim * mp_dim
    
    # For each output (i,k), which input pairs (i1,j),(j1,k1) contribute?
    # (i1,j),(j1,k1) -> (i,k) iff i1=i, j=j1, k1=k
    # So for output (i,k), exactly one input pair: (i,k) as α=(i,k... wait α=(i,j) and β=(j,k)
    # α = (i, j) where j ranges 0..m-1, β = (j, k)
    # For fixed output (i,k): α=(i,j), β=(j,k) for each j
    # So m terms contribute, each with value 1.
    
    # The matricized matrix M has shape (np, nm*mp)
    M = np.zeros((np_dim, total_input), dtype=int)
    for i in range(n):
        for k in range(p):
            gamma = i * p + k
            for j in range(m):
                alpha = i * m + j
                beta = j * p + k
                col = alpha * mp_dim + beta
                M[gamma, col] = 1
    
    return M  # shape (n*p, n*m*m*p)

def _compute_exact_rank(matrix):
    """Compute exact rank of integer matrix using fraction-based Gaussian elimination."""
    nrows = len(matrix)
    ncols = len(matrix[0]) if nrows > 0 else 0
    if nrows == 0 or ncols == 0:
        return 0
    
    # Use fractions for exact arithmetic
    M = [[Fraction(matrix[i][j]) for j in range(ncols)] for i in range(nrows)]
    
    rank = 0
    for col in range(ncols):
        if rank >= nrows:
            break
        # Find pivot row
        pivot_row = -1
        for r in range(rank, nrows):
            if M[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        # Swap
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        # Eliminate
        pivot_val = M[rank][col]
        for r in range(rank + 1, nrows):
            if M[r][col] != 0:
                factor = M[r][col] / pivot_val
                for c in range(col, ncols):
                    M[r][c] -= factor * M[rank][c]
        rank += 1
    
    return rank

def _extract_decomposition_from_rank(n, m, p, M, rank):
    """
    Given the matricized matrix M of shape (n*p, n*m*m*p) and its rank,
    extract a rank decomposition.
    
    The tensor T has entries T_{alpha,beta,gamma} where the matricization along gamma
    gives M[gamma, (alpha,beta)] = T_{alpha,beta,gamma}.
    
    A rank-r decomposition: T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    means T_{alpha,beta,gamma} = sum_t U[alpha,t]*V[beta,t]*W[gamma,t]
    
    From the matricization M[gamma, (alpha,beta)] = sum_t W[gamma,t] * (U[alpha,t]*V[beta,t])
    Let's define for each t, the outer product u_t ⊗ v_t is a vector in R^{nm*mp}
    indexed by (alpha,beta). So M = W @ P where W is (np, r) and P is (r, nm*mp).
    
    Actually M[gamma, alpha*mp+beta] = sum_t W[gamma,t] * Q_t[alpha, beta]
    where Q_t = u_t ⊗ v_t reshaped.
    
    So M = W * P where P[t, alpha*mp+beta] = U[alpha,t]*V[beta,t].
    
    To extract: find a basis for the column space of M (or row space).
    """
    nrows = len(M)
    ncols = len(M[0])
    
    # Use exact SVD-like approach: find basis for row space
    # Convert to fractions
    M_frac = [[Fraction(M[i][j]) for j in range(ncols)] for i in range(nrows)]
    
    # Perform exact Gaussian elimination to find basis
    M_copy = [row[:] for row in M_frac]
    basis_indices = []  # indices of pivot rows
    rank_found = 0
    for col in range(ncols):
        if rank_found >= rank:
            break
        pivot_row = -1
        for r in range(rank_found, nrows):
            if M_copy[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        M_copy[rank_found], M_copy[pivot_row] = M_copy[pivot_row], M_copy[rank_found]
        basis_indices.append(pivot_row)
        pivot_val = M_copy[rank_found][col]
        for r in range(rank_found + 1, nrows):
            if M_copy[r][col] != 0:
                factor = M_copy[r][col] / pivot_val
                for c in range(col, ncols):
                    M_copy[r][c] -= factor * M_copy[rank_found][c]
        rank_found += 1
    
    if rank_found != rank:
        # Fallback to schoolbook if something went wrong
        return None
    
    # The basis vectors (rows of M at basis_indices) span the row space.
    # M = A @ B where A is (nrows, rank) with orthonormal-like columns and B is (rank, ncols)
    # But we need: M[gamma, ab] = sum_t W[gamma,t] * P[t, ab]
    
    # Find left inverse: since the basis rows are linearly independent,
    # for each row gamma, express M[gamma] as a linear combination of basis rows.
    
    # Build the transformation: C such that M[gamma, :] = sum_t C[gamma,t] * M[basis_indices[t], :]
    # This is solving: for each gamma, M[gamma, :] = C[gamma, :] @ M_basis
    
    nm_dim = n * m
    mp_dim = m * p
    np_dim = n * p
    
    # M_basis has shape (rank, ncols)
    M_basis = np.array([M_frac[bi] for bi in basis_indices], dtype=object)
    
    # For each gamma, solve M_basis @ x = M_frac[gamma] for x
    # Since rank rows are independent, this is a system of ncols equations in rank unknowns.
    # We only need rank independent columns. Pick the pivot columns.
    
    # Find pivot columns from the elimination
    pivot_cols = []
    M_temp = [row[:] for row in M_frac]
    r2 = 0
    for col in range(ncols):
        if r2 >= rank:
            break
        pivot_row2 = -1
        for rr in range(r2, nrows):
            if M_temp[rr][col] != 0:
                pivot_row2 = rr
                break
        if pivot_row2 == -1:
            continue
        M_temp[r2], M_temp[pivot_row2] = M_temp[pivot_row2], M_temp[r2]
        pivot_cols.append(col)
        pv = M_temp[r2][col]
        for rr in range(r2 + 1, nrows):
            if M_temp[rr][col] != 0:
                factor = M_temp[rr][col] / pv
                for cc in range(col, ncols):
                    M_temp[rr][cc] -= factor * M_temp[r2][cc]
        r2 += 1
    
    # Now use the pivot columns to solve
    # A_sub = M_basis[:, pivot_cols] (rank x rank matrix)
    # For each gamma: C[gamma, :] @ A_sub = M_frac[gamma, pivot_cols]
    # So C[gamma, :] = M_frac[gamma, pivot_cols] @ A_sub^{-1}
    
    A_sub = [[M_frac[basis_indices[t]][pc] for pc in pivot_cols] for t in range(rank)]
    
    # Compute inverse of A_sub using fraction arithmetic
    A_inv = np.zeros((rank, rank), dtype=object)
    # Augmented matrix [A | I]
    aug = []
    for i in range(rank):
        row = A_sub[i][:] + [Fraction(1) if j == i else Fraction(0) for j in range(rank)]
        aug.append(row)
    
    for col in range(rank):
        pivot = -1
        for r in range(col, rank):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            return None  # singular, shouldn't happen
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for r in range(rank):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    
    A_inv = [[aug[i][rank + j] for j in range(rank)] for i in range(rank)]
    
    # Compute W: W[gamma, t] = sum_{pc} M_frac[gamma][pc] * A_inv[?, pc]
    # C[gamma, t] = sum_{s} M_frac[gamma][pivot_cols[s]] * A_inv[t, s] ... 
    # Actually: C[gamma] = M_frac[gamma, pivot_cols] @ A_inv
    # where M_frac[gamma, pivot_cols] is a 1xrank row vector
    
    W = [[0] * rank for _ in range(np_dim)]
    for gamma in range(np_dim):
        rhs = [M_frac[gamma][pc] for pc in pivot_cols]  # length rank
        for t in range(rank):
            val = Fraction(0)
            for s in range(rank):
                val += rhs[s] * A_inv[t][s]
            W[gamma][t] = val
    
    # Now for each t, P[t, alpha*mp+beta] should satisfy:
    # M[basis_indices[t], :] = sum_t' W[basis_indices[t], t'] * P[t', :]
    # Since basis rows are the "standard" rows, and W[basis_indices, :] is the identity-like...
    
    # Actually, from M = W @ P (in the tensor sense), and we've found W,
    # we can get P from the basis rows:
    # M[basis_indices[t], ab] = sum_{t'} W[basis_indices[t], t'] * P[t', ab]
    # The matrix W_basis[t, t'] = W[basis_indices[t], t']
    
    W_basis = [[W[basis_indices[t]][t2] for t2 in range(rank)] for t in range(rank)]
    
    # Find W_basis inverse
    W_basis_inv = np.zeros((rank, rank), dtype=object)
    aug2 = []
    for i in range(rank):
        row = W_basis[i][:] + [Fraction(1) if j == i else Fraction(0) for j in range(rank)]
        aug2.append(row)
    
    for col in range(rank):
        pivot = -1
        for r in range(col, rank):
            if aug2[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            return None
        aug2[col], aug2[pivot] = aug2[pivot], aug2[col]
        pv = aug2[col][col]
        aug2[col] = [x / pv for x in aug2[col]]
        for r in range(rank):
            if r != col and aug2[r][col] != 0:
                factor = aug2[r][col]
                aug2[r] = [a - factor * b for a, b in zip(aug2[r], aug2[col])]
    
    W_basis_inv = [[aug2[i][rank + j] for j in range(rank)] for i in range(rank)]
    
    # P[t, ab] = sum_{t'} W_basis_inv[t, t'] * M[basis_indices[t'], ab]
    P = [[0] * ncols for _ in range(rank)]
    for t in range(rank):
        for ab in range(ncols):
            val = Fraction(0)
            for t2 in range(rank):
                val += W_basis_inv[t][t2] * M_frac[basis_indices[t2]][ab]
            P[t][ab] = val
    
    # Now decompose P[t, alpha*mp+beta] = U[alpha, t] * V[beta, t]
    # This requires P[t] to be rank-1 as a (nm x mp) matrix
    U = [[0] * rank for _ in range(nm_dim)]
    V = [[0] * rank for _ in range(mp_dim)]
    
    for t in range(rank):
        # Extract the (nm x mp) matrix from P[t]
        # P[t][alpha*mp+beta]
        # Check if it's rank 1: all 2x2 minors are 0
        # Find nonzero entry
        nz_alpha = -1
        nz_beta = -1
        for alpha in range(nm_dim):
            for beta in range(mp_dim):
                if P[t][alpha * mp_dim + beta] != 0:
                    nz_alpha = alpha
                    nz_beta = beta
                    break
            if nz_alpha >= 0:
                break
        
        if nz_alpha < 0:
            # All zeros, shouldn't happen
            continue
        
        pivot_val = P[t][nz_alpha * mp_dim + nz_beta]
        
        for alpha in range(nm_dim):
            for beta in range(mp_dim):
                # P[t][alpha*mp+beta] = U[alpha,t]*V[beta,t]
                # U[nz_alpha,t]*V[nz_beta,t] = pivot_val
                # Let U[nz_alpha,t] = 1, V[nz_beta,t] = pivot_val (or some factorization)
                pass
        
        # Set U[nz_alpha,t] = pivot_val, V[nz_beta,t] = 1
        # Then U[alpha,t] = P[t][alpha*mp+nz_beta] / V[nz_beta,t] = P[t][alpha*mp+nz_beta]
        # V[beta,t] = P[t][nz_alpha*mp+beta] / U[nz_alpha,t] = P[t][nz_alpha*mp+beta] / pivot_val
        U[nz_alpha][t] = pivot_val
        V[nz_beta][t] = Fraction(1)
        
        valid = True
        for alpha in range(nm_dim):
            for beta in range(mp_dim):
                expected = P[t][alpha * mp_dim + beta]
                pred = U[alpha][t] * V[beta][t] if False else None  # placeholder
                pass
        
        # Fill in U and V
        for alpha in range(nm_dim):
            U[alpha][t] = P[t][alpha * mp_dim + nz_beta]
        for beta in range(mp_dim):
            V[beta][t] = P[t][nz_alpha * mp_dim + beta] / pivot_val
        
        # Verify rank-1
        for alpha in range(nm_dim):
            for beta in range(mp_dim):
                if U[alpha][t] * V[beta][t] != P[t][alpha * mp_dim + beta]:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Not a valid rank-1 factorization for this slice
            return None
    
    # Verify the full decomposition
    # C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] should equal the true product
    # For all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = Fraction(0)
                for t in range(rank):
                    lhs += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if lhs != 1:
                    return None
    
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    DEADLINE = 130.0  # Leave 20s margin for the 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook as fallback
    def schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    try:
        M = _matmul_tensor_matricization(n, m, p)
        
        # Convert to list of lists of int for Fraction
        M_list = [[int(M[i][j]) for j in range(M.shape[1])] for i in range(M.shape[0])]
        
        rank = _compute_exact_rank(M_list)
        
        if time.time() > DEADLINE - 20:
            return best_result
        
        if rank < best_rank:
            decomp = _extract_decomposition_from_rank(n, m, p, M_list, rank)
            if decomp is not None:
                best_result = decomp
                best_rank = rank
        
    except Exception:
        pass
    
    if time.time() > DEADLINE - 10:
        return best_result
    
    # If rank didn't improve, try known lower bounds
    # For 2x5x5, known rank is at most 49 (Strassen-style improvement)
    # Let's try to construct a rank-49 decomposition
    
    # Try rank 49: use the standard Strassen idea adapted
    # Actually, let me just try to see if rank 48 or lower is possible
    # But computing exact rank for smaller candidates is expensive.
    
    # Given time constraints, return what we have
    return best_result

# EVOLVE-BLOCK-END
