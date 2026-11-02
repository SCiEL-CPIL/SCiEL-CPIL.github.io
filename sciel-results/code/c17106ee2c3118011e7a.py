import numpy as np
from scipy.optimize import least_squares
import time
from itertools import product
from scipy.linalg import null_space

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m  # 6
    size_V = m * p  # 15
    size_W = p * n  # 10
    
    # Build target tensor indices
    targets_alpha = []
    targets_beta = []
    targets_gamma = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets_alpha.append(a)
                targets_beta.append(b)
                targets_gamma.append(c)
                
    targets_alpha = np.array(targets_alpha, dtype=int)
    targets_beta = np.array(targets_beta, dtype=int)
    targets_gamma = np.array(targets_gamma, dtype=int)
    
    num_targets = len(targets_alpha)
    
    # Precompute all zero entries for penalty
    all_indices_a = np.arange(size_U)
    all_indices_b = np.arange(size_V)
    all_indices_c = np.arange(size_W)
    
    def compute_full_residuals(U, V, W):
        """Compute residuals for all entries (non-zero targets and zero entries)"""
        # Compute full tensor
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Build target tensor
        T_target = np.zeros((size_U, size_V, size_W))
        T_target[targets_alpha, targets_beta, targets_gamma] = 1.0
        
        # Flatten difference
        diff = (T_full - T_target).flatten()
        return diff
    
    def residuals_from_flat(x, R):
        """Compute residuals from flattened parameter vector"""
        U = x[:size_U*R].reshape(size_U, R)
        V = x[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
        W = x[(size_U+size_V)*R:].reshape(size_W, R)
        return compute_full_residuals(U, V, W)
    
    def verify_solution(U, V, W):
        """Verify if the solution is exact (within tolerance)"""
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((size_U, size_V, size_W))
        T_target[targets_alpha, targets_beta, targets_gamma] = 1.0
        max_err = np.max(np.abs(T_full - T_target))
        return max_err < 1e-9, max_err
    
    def try_rank_R(R, num_restarts=10, max_nfev=200):
        """Try to find a decomposition of rank R"""
        num_vars = (size_U + size_V + size_W) * R
        
        for restart in range(num_restarts):
            # Different initialization strategies
            if restart == 0:
                # Random small integers
                x0 = np.random.randint(-2, 3, num_vars).astype(float)
            elif restart == 1:
                # Random with smaller magnitude
                x0 = np.random.uniform(-1, 1, num_vars)
            elif restart == 2:
                # Structured: try to build from schoolbook and perturb
                np.random.seed(restart * 12345)
                x0 = np.random.randint(-3, 4, num_vars).astype(float)
            else:
                # More random restarts with varying scales
                scale = 0.5 + restart * 0.1
                x0 = np.random.uniform(-scale, scale, num_vars)
            
            try:
                res = least_squares(
                    lambda x: residuals_from_flat(x, R),
                    x0,
                    method='trf',
                    max_nfev=max_nfev,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10
                )
                
                if res.cost < 1e-8:
                    # Try to snap to half-integers
                    x_sol = res.x
                    x_snapped = np.round(x_sol * 2) / 2.0
                    
                    U = x_snapped[:size_U*R].reshape(size_U, R)
                    V = x_snapped[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
                    W = x_snapped[(size_U+size_V)*R:].reshape(size_W, R)
                    
                    is_valid, max_err = verify_solution(U, V, W)
                    if is_valid:
                        return U.tolist(), V.tolist(), W.tolist()
                        
                    # If snapping didn't work, try without snapping
                    is_valid, max_err = verify_solution(U, V, W)
                    if is_valid:
                        return U.tolist(), V.tolist(), W.tolist()
                        
            except Exception:
                pass
        
        return None
    
    def lattice_search(R, time_limit=30):
        """
        Lattice-reduction-based search for integer/half-integer tensor decompositions.
        
        The matrix multiplication tensor can be viewed as a system of quadratic equations.
        For a rank-R decomposition, we have:
          T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        This is bilinear in the factor matrices. We can fix one factor matrix and solve
        for the others, or use lattice reduction to find sparse integer solutions.
        
        Strategy:
        1. Enumerate possible support patterns for U (which entries are non-zero)
        2. For each pattern, set up linear constraints on V and W
        3. Use lattice reduction to find integer solutions
        """
        start_time = time.time()
        
        # The tensor has 30 non-zero entries (n*m*p = 2*3*5 = 30)
        # We need: sum_r U[a,r]*V[b,r]*W[c,r] = 1 for the 30 target entries
        # and = 0 for the other 870 entries
        
        # Approach: Fix W to be sparse (few non-zero entries per column),
        # then solve for U and V via linear algebra
        
        # For each column r, W[:,r] is a vector in R^10
        # The constraint is that for each (a,b) not in targets, 
        # sum_r U[a,r]*V[b,r]*W[c,r] = 0 for all c
        
        # Let's try a structured approach:
        # Assume each column of U, V, W has at most 2 non-zero entries
        # This gives us a combinatorial search over support patterns
        
        # Precompute target set
        target_set = set(zip(targets_alpha, targets_beta, targets_gamma))
        
        # For a rank-R decomposition with sparse factors:
        # Each column r defines a "rank-1 tensor" U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # The sum of these rank-1 tensors must equal the target tensor
        
        # Strategy: Try to find R rank-1 tensors whose sum equals the target
        # Each rank-1 tensor is determined by its support pattern
        
        # For simplicity, assume each factor has exactly 2 non-zero entries per column
        # (This is a heuristic; we can vary this)
        
        # Build list of possible "atoms" (rank-1 tensors with 2-nonzero support in each factor)
        # This is too large, so let's use a different approach
        
        # Alternative: Use the fact that the tensor is bilinear
        # Fix W, then the system becomes linear in (U*V)
        # Specifically, for fixed W, we need:
        #   (U ⊙ W) * V^T = T  (where ⊙ is some operation)
        
        # Let's try: enumerate sparse patterns for W, then solve linear system
        
        # For each column r of W, assume it has exactly k non-zero entries
        # Let's try k=1 first (W is a permutation matrix pattern)
        
        # If W[:,r] has only one non-zero entry at position c_r with value w_r,
        # then the r-th term contributes w_r * U[:,r] * V[:,r] to row c_r of the tensor
        
        # Actually, let's be more systematic. The tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        
        # For a given rank R, we can write this as a linear system in the variables
        # U[a,r], V[b,r], W[c,r]
        
        # But it's quadratic, not linear. However, if we fix one set of variables,
        # it becomes linear in the others.
        
        # Let's fix W and solve for U and V.
        # For each c, define W_c = W[c,:] (a row vector of length R)
        # Then T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        
        # This can be written as: T[:, :, c] = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
        
        # So for each c, T[:,:,c] is a linear combination of the R rank-1 matrices U[:,r]⊗V[:,r]
        # with coefficients W[c,r]
        
        # If we fix the rank-1 matrices M_r = U[:,r] ⊗ V[:,r], then
        # T[:,:,c] = sum_r W[c,r] * M_r
        # This is a linear system in W[c,r]
        
        # Conversely, if we fix W, then we need to find rank-1 matrices M_r such that
        # T[:,:,c] = sum_r W[c,r] * M_r for all c
        
        # This is equivalent to saying that the {T[:,:,c]} span a space of dimension ≤ R,
        # and the M_r form a basis with W giving the coordinates
        
        # So: compute the span of {T[:,:,c] : c=0..p-1}
        # If dim ≤ R, we can find a basis and express T[:,:,c] in that basis
        
        # Let's try this approach!
        
        # Build the tensor T
        T = np.zeros((size_U, size_V, size_W))
        T[targets_alpha, targets_beta, targets_gamma] = 1.0
        
        # For each c, T[:,:,c] is a size_U × size_V matrix
        # Stack them: T_slices = [T[:,:,0], T[:,:,1], ..., T[:,:,p-1]]
        # Each slice is size_U × size_V = 6 × 15 = 90 entries
        
        # The span of these p slices: what's the dimension?
        # If dim = d, then we need rank ≥ d
        
        # Let's compute the dimension of the span
        T_flat = T.reshape(size_U * size_V, size_W)  # 90 × 10
        rank_T = np.linalg.matrix_rank(T_flat)
        
        # The minimum rank is at least rank_T
        # Also at least max(size_U, size_V, size_W) = 15 for non-trivial tensors
        
        # Let's try to find a decomposition of rank = rank_T
        
        # If rank_T ≤ R, we can find R rank-1 matrices M_1, ..., M_R such that
        # T[:,:,c] = sum_r W[c,r] * M_r
        
        # Step 1: Find a basis for the column space of T_flat
        # Step 2: Express each T[:,:,c] in this basis
        # Step 3: The basis vectors are the M_r (reshaped)
        # Step 4: The coordinates are W[c,r]
        # Step 5: Decompose each M_r = U[:,r] ⊗ V[:,r] (rank-1 decomposition)
        
        # Let's implement this
        
        # Get basis for column space
        # Use SVD
        U_svd, s_vals, Vt_svd = np.linalg.svd(T_flat, full_matrices=False)
        
        # The rank is the number of non-zero singular values
        tol = 1e-10
        rank_svd = np.sum(s_vals > tol)
        
        if rank_svd > R:
            return None
        
        # Use the first R left singular vectors as basis (or rank_svd if smaller)
        num_basis = min(R, rank_svd)
        if num_basis == 0:
            return None
        
        # Basis vectors: U_svd[:, :num_basis], each of length size_U*size_V
        # Coordinates: Vt_svd[:num_basis, :] transposed, each of length size_W
        
        # M_r (for r=0..num_basis-1) = U_svd[:, r].reshape(size_U, size_V)
        # W[c, r] = Vt_svd[r, c]
        
        # Now we need to decompose each M_r = U[:,r] ⊗ V[:,r]
        # M_r is a size_U × size_V matrix. If it has rank 1, we can decompose it.
        
        # Check if each M_r has rank 1
        U_factors = np.zeros((size_U, R))
        V_factors = np.zeros((size_V, R))
        W_factors = np.zeros((size_W, R))
        
        for r in range(num_basis):
            M_r = U_svd[:, r].reshape(size_U, size_V)
            
            # Check rank of M_r
            # If rank 1, decompose
            U_r, s_r, Vt_r = np.linalg.svd(M_r, full_matrices=False)
            
            if len(s_r) > 1 and s_r[1] > tol:
                # M_r is not rank 1, this approach fails
                # We could try a different basis, but for now return None
                return None
            
            # M_r = U_r[:, 0] * s_r[0] * Vt_r[0, :]
            # We can absorb s_r[0] into either factor
            U_factors[:, r] = U_r[:, 0]
            V_factors[:, r] = s_r[0] * Vt_r[0, :]
            W_factors[:, r] = Vt_svd[r, :]
        
        # Now we have U_factors, V_factors, W_factors
        # But they might not be integer/half-integer
        # Try to snap to half-integers
        
        # Combine into flat vector
        x_sol = np.concatenate([U_factors.flatten(), V_factors.flatten(), W_factors.flatten()])
        x_snapped = np.round(x_sol * 2) / 2.0
        
        U = x_snapped[:size_U*R].reshape(size_U, R)
        V = x_snapped[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
        W = x_snapped[(size_U+size_V)*R:].reshape(size_W, R)
        
        # Verify
        is_valid, max_err = verify_solution(U, V, W)
        if is_valid:
            return U.tolist(), V.tolist(), W.tolist()
        
        # If snapping didn't work, try without snapping
        is_valid, max_err = verify_solution(U_factors, V_factors, W_factors)
        if is_valid:
            return U_factors.tolist(), V_factors.tolist(), W_factors.tolist()
        
        return None
    
    def lattice_search_enhanced(R, time_limit=30):
        """
        Enhanced lattice search with multiple strategies.
        """
        start_time = time.time()
        
        # Strategy 1: SVD-based approach (already implemented in lattice_search)
        result = lattice_search(R, time_limit)
        if result is not None:
            return result
        
        if time.time() - start_time > time_limit:
            return None
        
        # Strategy 2: Try different subsets of W columns
        # Instead of using all p slices, try subsets
        
        # Strategy 3: Randomized SVD
        for trial in range(5):
            if time.time() - start_time > time_limit:
                break
            
            # Add small random perturbation to break degeneracy
            T = np.zeros((size_U, size_V, size_W))
            T[targets_alpha, targets_beta, targets_gamma] = 1.0
            
            # Perturb
            T_perturbed = T + np.random.randn(*T.shape) * 1e-6
            
            T_flat = T_perturbed.reshape(size_U * size_V, size_W)
            U_svd, s_vals, Vt_svd = np.linalg.svd(T_flat, full_matrices=False)
            
            tol = 1e-8
            rank_svd = np.sum(s_vals > tol)
            
            if rank_svd > R:
                continue
            
            num_basis = min(R, rank_svd)
            if num_basis == 0:
                continue
            
            U_factors = np.zeros((size_U, R))
            V_factors = np.zeros((size_V, R))
            W_factors = np.zeros((size_W, R))
            
            success = True
            for r in range(num_basis):
                M_r = U_svd[:, r].reshape(size_U, size_V)
                
                U_r, s_r, Vt_r = np.linalg.svd(M_r, full_matrices=False)
                
                if len(s_r) > 1 and s_r[1] > tol:
                    success = False
                    break
                
                U_factors[:, r] = U_r[:, 0]
                V_factors[:, r] = s_r[0] * Vt_r[0, :]
                W_factors[:, r] = Vt_svd[r, :]
            
            if not success:
                continue
            
            # Snap to half-integers
            x_sol = np.concatenate([U_factors.flatten(), V_factors.flatten(), W_factors.flatten()])
            x_snapped = np.round(x_sol * 2) / 2.0
            
            U = x_snapped[:size_U*R].reshape(size_U, R)
            V = x_snapped[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
            W = x_snapped[(size_U+size_V)*R:].reshape(size_W, R)
            
            is_valid, max_err = verify_solution(U, V, W)
            if is_valid:
                return U.tolist(), V.tolist(), W.tolist()
            
            # Try without snapping
            is_valid, max_err = verify_solution(U_factors, V_factors, W_factors)
            if is_valid:
                return U_factors.tolist(), V_factors.tolist(), W_factors.tolist()
        
        return None
    
    start_time = time.time()
    deadline = 110  # Leave 10s safety margin
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = 15 for non-trivial
    # Upper bound: n*m*p = 30 (schoolbook)
    
    # First, compute the theoretical lower bound from SVD
    T = np.zeros((size_U, size_V, size_W))
    T[targets_alpha, targets_beta, targets_gamma] = 1.0
    T_flat = T.reshape(size_U * size_V, size_W)
    rank_T = np.linalg.matrix_rank(T_flat)
    
    # Also check the other two unfoldings
    T_flat2 = T.reshape(size_U, size_V * size_W)
    rank_T2 = np.linalg.matrix_rank(T_flat2)
    T_flat3 = T.reshape(size_U * size_W, size_V)
    rank_T3 = np.linalg.matrix_rank(T_flat3)
    
    lower_bound = max(rank_T, rank_T2, rank_T3)
    
    # Try ranks from lower_bound to 30
    for R in range(max(10, lower_bound), 31):
        if time.time() > deadline:
            break
        
        # Try lattice search first (faster for structured problems)
        result = lattice_search_enhanced(R, time_limit=20)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline:
            break
        
        # Try optimization-based approach
        result = try_rank_R(R, num_restarts=3, max_nfev=100)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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

# EVOLVE-BLOCK-END
