import numpy as np
import time
import itertools
from functools import lru_cache

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_max = n * m * p  # 72
    deadline = time.time() + 145  # safety margin

    # Schoolbook fallback
    U_fb = [[0] * rank_max for _ in range(n * m)]
    V_fb = [[0] * rank_max for _ in range(m * p)]
    W_fb = [[0] * rank_max for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_max

    # Precompute all possible indices
    U_indices = list(range(n * m))
    V_indices = list(range(m * p))
    W_indices = list(range(p * n))

    # Generate candidate subsets for each factor
    # We limit subset size to 3 to keep the search space manageable
    # A column in U is a vector of length n*m. We consider subsets of size 1, 2, 3.
    def gen_subsets(indices, max_size):
        subsets = []
        for r in range(1, max_size + 1):
            for comb in itertools.combinations(indices, r):
                subsets.append(comb)
        return subsets

    # We will use a greedy approach.
    # State: Residual tensor R[i,j,k]
    # Initially R[i,j,k] = 1 for all i,j,k.
    # We want to select columns (u, v, w) such that sum_c u_i v_j w_k = 1.
    # This is equivalent to minimizing ||R - sum_c u_i v_j w_k||_F^2 = 0.
    
    # However, the standard greedy for tensor decomposition (like CP) is complex.
    # Given the small dimensions (3x4x6 = 72 entries), we can try a different angle.
    
    # Let's try a "Combinatorial Search" with pruning.
    # We know the rank is likely around 24-30.
    # We will try to construct a valid decomposition of rank r by picking columns one by one.
    # At each step, we have a residual R.
    # We pick a column (u, v, w) from a candidate pool.
    # We subtract u_i v_j w_k from R.
    # We want R to become 0.
    
    # Candidate pool:
    # To keep it tractable, we restrict the support (non-zero entries) of u, v, w.
    # Let's try supports of size 1 for all (schoolbook-like but with sharing)
    # and size 2 for some.
    
    # Actually, a smarter approach for small dimensions:
    # Use a Mixed Integer Programming (MIP) solver? No, only standard lib + numpy/scipy.
    # scipy.optimize.milp is available in newer scipy. Let's check if we can use it.
    # But the variables are real numbers (or half-integers), not just integers.
    # The problem is: find min rank r such that there exist U, V, W with rank r columns satisfying the equations.
    # This is non-convex.
    
    # Let's stick to the Greedy Residual Reduction but with a smarter candidate set.
    # Candidate set: All possible vectors u, v, w with entries in {-1, -0.5, 0, 0.5, 1}
    # and sparsity <= 3.
    # Number of such vectors:
    # For U (size 12): 
    #   Support size 1: 12 * 4 (values) = 48
    #   Support size 2: C(12,2) * 16 = 66 * 16 = 1056
    #   Support size 3: C(12,3) * 64 = 220 * 64 = 14080
    # Total ~ 15k candidates for U.
    # Same for V (size 24) and W (size 18).
    # V candidates:
    #   Size 1: 24 * 4 = 96
    #   Size 2: C(24,2) * 16 = 276 * 16 = 4416
    #   Size 3: C(24,3) * 64 = 2024 * 64 = 129536 -> Too many.
    # Let's limit V and W to support size 2.
    # W candidates:
    #   Size 1: 18 * 4 = 72
    #   Size 2: C(18,2) * 16 = 153 * 16 = 2448
    
    # Total candidates:
    # U: ~15k
    # V: ~4.5k
    # W: ~2.5k
    # Total combinations per step: 15k * 4.5k * 2.5k = 1.7 billion. Too many.
    
    # We need to reduce the search space.
    # Observation: The tensor is 3x4x6.
    # Let's try a different decomposition strategy: **Strassen-like**.
    # Strassen works for 2x2. Can we adapt it?
    # For 3x3, rank is 23 (or less?).
    # For 3x4x6, we can pad to 4x4x4 or 8x8x8?
    # Padding increases the rank usually.
    
    # Let's try a **Genetic Algorithm** with a fitness function based on the residual.
    # Representation: A set of columns.
    # Fitness: -||Residual||_F^2.
    # We start with the schoolbook decomposition (rank 72).
    # We try to merge columns or replace them to reduce rank.
    
    # Alternative: **Local Search on Rank**.
    # Start with rank 24.
    # Try to find a solution.
    # If fail, increase rank.
    
    # Let's try a **Randomized Local Search** with a specific initialization.
    # Initialize U, V, W with random values from {-1, 0, 1}.
    # Compute Residual R.
    # Pick a random column c.
    # Randomly perturb U[:, c], V[:, c], W[:, c].
    # If ||R_new|| < ||R_old||, accept.
    # Repeat until R is 0 or time runs out.
    
    # This is essentially Simulated Annealing or Hill Climbing.
    
    np.random.seed(42)
    
    def compute_residual(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # R[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want R to be all 1s.
        # Residual = 1 - R.
        
        # Vectorized computation
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        
        # Product U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Sum over c.
        # This is a tensor contraction.
        # Term = U_t * V_t[None, :, :, :] * W_t[:, None, None, :] ... wait, indices don't match directly.
        
        # Let's use einsum
        # 'ijc,jkc,kic -> ijk'
        try:
            R = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        except Exception:
            # Fallback to loops if einsum fails (shouldn't)
            R = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        
        return np.ones((n, m, p)) - R

    def norm_sq(res):
        return np.sum(res * res)

    # Start with schoolbook
    U = np.zeros((n*m, rank_max))
    V = np.zeros((m*p, rank_max))
    W = np.zeros((p*n, rank_max))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # We want to reduce rank.
    # We will try to zero out columns by merging them.
    # Actually, let's just do a random search for lower ranks as before, but with a better heuristic.
    # Heuristic: Start with random sparse matrices.
    
    best_rank_found = rank_max
    best_U = U
    best_V = V
    best_W = W
    
    # Try ranks from 24 to 40 (likely range)
    for r in range(24, 41):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Initialize
            # Use a mix of dense and sparse
            if attempt % 2 == 0:
                # Dense random
                U_c = np.random.uniform(-1, 1, size=(n*m, r))
                V_c = np.random.uniform(-1, 1, size=(m*p, r))
                W_c = np.random.uniform(-1, 1, size=(p*n, r))
            else:
                # Sparse random with values from {-1, -0.5, 0.5, 1}
                U_c = np.zeros((n*m, r))
                V_c = np.zeros((m*p, r))
                W_c = np.zeros((p*n, r))
                
                for c in range(r):
                    # Pick 2-3 non-zeros
                    nz_u = np.random.choice(n*m, size=np.random.randint(2, 5), replace=False)
                    nz_v = np.random.choice(m*p, size=np.random.randint(2, 5), replace=False)
                    nz_w = np.random.choice(p*n, size=np.random.randint(2, 5), replace=False)
                    
                    U_c[nz_u, c] = np.random.choice([-1, -0.5, 0.5, 1], size=len(nz_u))
                    V_c[nz_v, c] = np.random.choice([-1, -0.5, 0.5, 1], size=len(nz_v))
                    W_c[nz_w, c] = np.random.choice([-1, -0.5, 0.5, 1], size=len(nz_w))
            
            # Local search / Hill Climbing
            current_res = compute_residual(U_c, V_c, W_c)
            current_norm = norm_sq(current_res)
            
            # If already good, check for exact
            if current_norm < 1e-9:
                best_rank_found = r
                best_U = U_c
                best_V = V_c
                best_W = W_c
                break
                
            # Try to improve
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                # Pick a random column to perturb
                c = np.random.randint(r)
                
                # Save old
                old_U = U_c[:, c].copy()
                old_V = V_c[:, c].copy()
                old_W = W_c[:, c].copy()
                
                # Perturb
                # Strategy: Replace with new random values
                U_c[:, c] = np.random.uniform(-1, 1, size=n*m)
                V_c[:, c] = np.random.uniform(-1, 1, size=m*p)
                W_c[:, c] = np.random.uniform(-1, 1, size=p*n)
                
                new_res = compute_residual(U_c, V_c, W_c)
                new_norm = norm_sq(new_res)
                
                if new_norm < current_norm:
                    current_norm = new_norm
                    current_res = new_res
                    improved = True
                else:
                    # Revert
                    U_c[:, c] = old_U
                    V_c[:, c] = old_V
                    W_c[:, c] = old_W
                
                if current_norm < 1e-9:
                    best_rank_found = r
                    best_U = U_c
                    best_V = V_c
                    best_W = W_c
                    break
        
        if best_rank_found < rank_max:
            break
            
    if best_rank_found < rank_max:
        best_result = {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
        
    return best_result
# EVOLVE-BLOCK-END
