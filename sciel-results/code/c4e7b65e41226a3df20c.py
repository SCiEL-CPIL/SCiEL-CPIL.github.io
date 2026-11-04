"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Build the multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if the (i,k) entry of A@B depends on A[i,j]*B[j,k]
    # Actually, the tensor for matrix multiplication: T[i,j,k] = 1 for all i,j,k
    # The multiplication tensor is the identity-like tensor where
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid indices (it's the all-ones tensor? No!)
    # 
    # Wait. The matrix multiplication tensor is defined by:
    # T_{i,j,k} = 1 if we multiply A[i,j] * B[j,k] to contribute to C[i,k]
    # Actually the standard tensor: for multiplying (n×m) by (m×p),
    # the tensor T is in R^{n × m × p} with T[i,j,k] = 1 for all i,j,k.
    # No wait - that's not right either.
    #
    # The matrix multiplication tensor: we have bilinear form
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has entries T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # This is the all-ones tensor of shape (n,m,p).
    #
    # A rank-r decomposition: T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c in R^n, v_c in R^m, w_c in R^p
    #
    # In the interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m + j, c] = u_c[i]
    # V[j*p + k, c] = v_c[j]  ... wait, let me re-read
    #
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So U rows indexed by (i,j) -> i*m+j, V rows by (j,k) -> j*p+k, W rows by (k,i) -> k*n+i
    #
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    #
    # This is a tensor completion / decomposition problem.

    # Flattened tensor
    # T[i,j,k] = 1 for all i,j,k
    # Shape: (n, m, p)
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Reshape to matrix for ALS: T_mat[i*m+j, k] doesn't quite work for 3-way
    # Better: use mode-n unfolding
    
    # We'll do ALS on the 3-way tensor
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # u_c is length n, v_c length m, w_c length p
    
    # U_flat: (n*m, rank), row i*m+j = u_c[i]
    # V_flat: (m*p, rank), row j*p+k = v_c[j]  
    # W_flat: (p*n, rank), row k*n+i = w_c[i]
    
    # For ALS: fix two, solve for third.
    
    def flatten_U(u_list):
        """u_list: list of rank vectors of length n -> (n*m, rank)"""
        U = np.zeros((n*m, len(u_list)), dtype=np.float64)
        for c, u in enumerate(u_list):
            for i in range(n):
                U[i*m:(i+1)*m, c] = u[i]  # wait, this is wrong
        # Actually U[i*m+j, c] = u_c[i] for all j
        # So for fixed c, U[:, c] has n*m entries, grouped by i, each group of m has same value u_c[i]
        U = np.zeros((n*m, len(u_list)), dtype=np.float64)
        for c in range(len(u_list)):
            for i in range(n):
                U[i*m:(i+1)*m, c] = u_list[c][i]
        return U
    
    def flatten_V(v_list):
        """v_list: list of rank vectors of length m -> (m*p, rank)"""
        V = np.zeros((m*p, len(v_list)), dtype=np.float64)
        for c in range(len(v_list)):
            for j in range(m):
                V[j*p:(j+1)*p, c] = v_list[c][j]
        return V
    
    def flatten_W(w_list):
        """w_list: list of rank vectors of length p -> (p*n, rank)"""
        W = np.zeros((p*n, len(w_list)), dtype=np.float64)
        for c in range(len(w_list)):
            for k in range(p):
                W[k*n:(k+1)*n, c] = w_list[c][k]
        return W
    
    def unflatten_U(U):
        """U: (n*m, rank) -> list of rank vectors of length n"""
        rank = U.shape[1]
        u_list = []
        for c in range(rank):
            u = np.zeros(n, dtype=np.float64)
            for i in range(n):
                u[i] = U[i*m, c]  # all m entries in row group are same
            u_list.append(u)
        return u_list
    
    def unflatten_V(V):
        rank = V.shape[1]
        v_list = []
        for c in range(rank):
            v = np.zeros(m, dtype=np.float64)
            for j in range(m):
                v[j] = V[j*p, c]
            v_list.append(v)
        return v_list
    
    def unflatten_W(W):
        rank = W.shape[1]
        w_list = []
        for c in range(rank):
            w = np.zeros(p, dtype=np.float64)
            for k in range(p):
                w[k] = W[k*n, c]
            w_list.append(w)
        return w_list
    
    def compute_error(u_list, v_list, w_list):
        """Compute ||T - sum_c u_c ⊗ v_c ⊗ w_c||_F"""
        # T[i,j,k] = 1
        # Approx[i,j,k] = sum_c u_c[i]*v_c[j]*w_c[k]
        # Build approx tensor
        approx = np.zeros((n, m, p), dtype=np.float64)
        for c in range(len(u_list)):
            approx += np.outer(np.outer(u_list[c], v_list[c]), w_list[c])
        return np.linalg.norm(T - approx)
    
    def round_to_half(x):
        return np.round(2*x) / 2.0
    
    def try_rank(r, max_iters=50, seed=0):
        """Try to find rank-r decomposition via ALS + rounding"""
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(seed)
        
        # Initialize
        u_list = [rng.randn(n) for _ in range(r)]
        v_list = [rng.randn(m) for _ in range(r)]
        w_list = [rng.randn(p) for _ in range(r)]
        
        best_err = compute_error(u_list, v_list, w_list)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update u: for each c, minimize over u_c
            # Fix v, w. The term involving u_c is:
            # sum_{i,j,k} (T[i,j,k] - sum_{c'} u_{c'}[i]v_{c'}[j]w_{c'}[k]) * u_c[i] v_c[j] w_c[k]
            # = sum_i u_c[i] * sum_{j,k} v_c[j] w_c[k] * (T[i,j,k] - sum_{c'≠c} u_{c'}[i]v_{c'}[j]w_{c'}[k])
            # For each c, solve linear system for u_c
            # Actually, let's do gradient-free: just do a few Gauss-Seidel-like updates
            
            # Simpler: use ALS with mode-n unfolding
            # Mode-1 unfolding: T_(1) of shape (n, m*p)
            # T_(1)[i, j*p+k] = T[i,j,k] = 1
            T1 = np.ones((n, m*p), dtype=np.float64)
            
            # Mode-2 unfolding: T_(2) of shape (m, n*p)
            T2 = np.ones((m, n*p), dtype=np.float64)
            
            # Mode-3 unfolding: T_(3) of shape (p, n*m)
            T3 = np.ones((p, n*m), dtype=np.float64)
            
            # Build V_mat: (m*p, rank) from v_list
            V_mat = flatten_V(v_list)
            # But V_mat has shape (m*p, rank), and for mode-1 ALS we need to reshape
            # Actually for ALS on 3-way tensor:
            # T_(1) ≈ U_mat @ (W_mat ⊙ V_mat)^T where ⊙ is Khatri-Rao
            # This is getting complex. Let me just do simple iterative refinement.
            
            # Simple approach: for each component c, update u_c, v_c, w_c by
            # projecting the residual onto the subspace spanned by the other two factors.
            
            # Compute current approximation
            approx = np.zeros((n, m, p), dtype=np.float64)
            for c in range(r):
                approx += np.outer(np.outer(u_list[c], v_list[c]), w_list[c])
            
            residual = T - approx
            
            for c in range(r):
                if time.time() > deadline:
                    break
                # Update u_c: minimize ||residual - u_c ⊗ v_c ⊗ w_c||_F
                # = minimize sum_{i,j,k} (residual[i,j,k] - u_c[i]v_c[j]w_c[k])^2
                # Optimal u_c[i] = sum_{j,k} residual[i,j,k] v_c[j] w_c[k] / sum_{j,k} (v_c[j]w_c[k])^2
                denom_u = np.sum((v_list[c] * w_list[c].reshape(1, p)) ** 2)
                # Actually: sum_{j,k} v_c[j]^2 w_c[k]^2
                denom_u = np.sum(v_list[c]**2) * np.sum(w_list[c]**2)
                if abs(denom_u) < 1e-12:
                    continue
                for i in range(n):
                    num = np.sum(residual[i, :, :] * np.outer(v_list[c], w_list[c]))
                    u_list[c][i] = num / denom_u
                
                # Recompute residual contribution of component c
                approx_c = np.outer(np.outer(u_list[c], v_list[c]), w_list[c])
                residual = T - approx + approx_c  # remove old, will add new below
                # Actually let's just recompute fully each iteration for simplicity
                # But that's O(r*n*m*p) per iteration which is fine for small dims
            
            # Recompute approximation and error
            approx = np.zeros((n, m, p), dtype=np.float64)
            for c in range(r):
                approx += np.outer(np.outer(u_list[c], v_list[c]), w_list[c])
            err = np.linalg.norm(T - approx)
            if err < best_err:
                best_err = err
            
            # Also update v and w similarly
            for c in range(r):
                if time.time() > deadline:
                    break
                # Update v_c
                denom_v = np.sum(u_list[c]**2) * np.sum(w_list[c]**2)
                if abs(denom_v) < 1e-12:
                    continue
                for j in range(m):
                    num = np.sum(residual[:, j, :] * np.outer(u_list[c], w_list[c]))
                    v_list[c][j] = num / denom_v
                
                # Update w_c
                denom_w = np.sum(u_list[c]**2) * np.sum(v_list[c]**2)
                if abs(denom_w) < 1e-12:
                    continue
                for k in range(p):
                    num = np.sum(residual[:, :, k] * np.outer(u_list[c], v_list[c]))
                    w_list[c][k] = num / denom_w
            
            # Recompute
            approx = np.zeros((n, m, p), dtype=np.float64)
            for c in range(r):
                approx += np.outer(np.outer(u_list[c], v_list[c]), w_list[c])
            err = np.linalg.norm(T - approx)
            if err < best_err:
                best_err = err
            
            if best_err < 1e-6:
                break
        
        # Round to half-integers and check
        u_rounded = [round_to_half(u) for u in u_list]
        v_rounded = [round_to_half(v) for v in v_list]
        w_rounded = [round_to_half(w) for w in w_list]
        
        # Check exactness
        approx = np.zeros((n, m, p), dtype=np.float64)
        for c in range(r):
            approx += np.outer(np.outer(u_rounded[c], v_rounded[c]), w_rounded[c])
        
        if np.allclose(approx, T, atol=1e-9):
            # Valid! Build output
            U = flatten_U(u_rounded)
            V = flatten_V(v_rounded)
            W = flatten_W(w_rounded)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = n*m + m*p + n*p - 2  # 69
    schoolbook = n * m * p  # 105
    
    # Try a few ranks
    ranks_to_try = list(range(lower_bound, min(lower_bound + 20, schoolbook + 1)))
    # Also try some specific promising ranks
    extra_ranks = [70, 72, 75, 80, 85, 90, 95, 100, 104, 105]
    ranks_to_try = list(set(ranks_to_try + extra_ranks))
    ranks_to_try.sort()
    
    # Remove schoolbook (we know it works)
    if schoolbook in ranks_to_try:
        ranks_to_try.remove(schoolbook)
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        # Try a few seeds
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            result = try_rank(r, max_iters=30, seed=seed)
            if result is not None:
                return result
    
    # Fallback: schoolbook
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
