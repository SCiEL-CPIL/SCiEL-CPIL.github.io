"""Search for a low-rank exact tensor decomposition of 3x3 * 3x4 matrix multiplication."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
DEADLINE = 110.0  # safety margin before 120s / 150s kill

# The multiplication tensor T has entries T[i,j,k] = 1 for all (i,j,k) in {0..2}x{0..2}x{0..3}
# A rank-r decomposition is a set of r triples (u_c, v_c, w_c) such that
#   T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
# with u_c in R^3, v_c in R^3, w_c in R^4.
# The output factor W follows trace convention W[k*n+i, r] = w_c[k] * u_c[i] (product), but actually
# the task says W[k*n+i, r] is the factor matrix entry. Let me re-read.
#
# "output factor follows the official trace convention W[k*n+i,r]"
# U has shape (n*m, rank) = (9, rank), indexed U[i*m+j, c]
# V has shape (m*p, rank) = (12, rank), indexed V[j*p+k, c]
# W has shape (p*n, rank) = (12, rank), indexed W[k*n+i, c]
#
# The decomposition is: for the product C = A*B where A is 3x3, B is 3x4,
# C[i,k] = sum_{j} A[i,j] * B[j,k]
#
# In tensor form: T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k].
# So T[i,j,k] = 1 for all valid (i,j,k).
#
# A rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
#
# This is the standard matrix multiplication tensor decomposition.

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook rank = n*m*p = 36
    # We know from literature that M(3,3,4) has rank at most... 
    # Let's think about known bounds.
    # The standard schoolbook is 36.
    # There are known results: for 2x2x2 it's 7 (Strassen).
    # For 3x3x3, the best known is 23 (Landsberg).
    # For 3x3x4, we need to figure out.
    
    # Strategy: try to find a decomposition with rank < 36 using randomized search + local optimization.
    # But first, let's try a structured approach.
    
    # Actually, let me think about this differently. The tensor T[i,j,k] = 1 for all i,j,k.
    # This is a "constant" tensor. 
    # 
    # A decomposition of rank r means we can write 1 = sum_c u_c[i] * v_c[j] * w_c[k] for all i,j,k.
    # This is equivalent to: the 3x3 matrix (u_c[i] * v_c[j]) for each c, and w_c[k]...
    # Actually, let's think of it as: for each k, the 3x3 matrix M_k[i,j] = T[i,j,k] = 1 must equal
    # sum_c (u_c[i] * w_c[k]) * v_c[j]. So M_k = sum_c (u_c * w_c[k]) ⊗ v_c^T... 
    # Hmm, this is getting complicated.
    
    # Let me try a different approach: randomized rank-r search with gradient-free methods.
    # For small dimensions, we can try ranks from lower to upper bound.
    
    # Known: the rank of the 3x3x4 multiplication tensor.
    # Let me just try to find a good decomposition via random initialization + local search.
    
    # First, let's establish the schoolbook as fallback.
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_school, V_school, W_school, rank_school = schoolbook()
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions.
    # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # A rank-r decomposition: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try random search for various ranks.
    # For each rank r, initialize randomly and try to optimize.
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from a reasonable lower bound up to schoolbook - 1
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 12, 12) = 12? No, that's not a valid lower bound.
    # The rank must be at least the dimension of any matricization.
    # Matricization along first index: 3 x (3*4) = 3x12, rank <= 3. So rank >= 3.
    # Matricization along second: 3 x 12, rank <= 3.
    # Matricization along third: 4 x 9, rank <= 4.
    # So rank >= max(3, 3, 4) = 4. But that's a very weak bound.
    
    # Actually, the rank must satisfy that the tensor fits in a rank-r decomposition.
    # Let's just try ranks from, say, 15 up to 35.
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This should equal 1 for all i,j,k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def residual(U, V, W):
        """Compute the maximum residual."""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    max_res = max(max_res, abs(val - 1.0))
        return max_res
    
    # Try randomized search with local refinement.
    # For each rank r, try multiple random initializations.
    
    def random_decomposition(r, scale=1.0):
        U = np.random.randn(n * m, r) * scale
        V = np.random.randn(m * p, r) * scale
        W = np.random.randn(p * n, r) * scale
        return U, V, W
    
    def local_search(U, V, W, max_iters=200, tol=1e-10):
        """Simple coordinate descent to minimize the residual."""
        r = U.shape[1]
        prev_res = residual(U, V, W)
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            # Try updating each coordinate
            for c in range(r):
                # Update U[:, c]
                old_Uc = U[:, c].copy()
                # Optimal U[:,c] given V and W: minimize sum_{i,j,k} (1 - sum_{c'} U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'])^2
                # This is a linear least squares problem for U[:,c].
                # Let R = 1 - sum_{c' != c} U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c']
                # Then we want to minimize sum_{i,j,k} (R[i,j,k] - U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
                # This is not linear in U[:,c] because of the V and W factors.
                # Actually, for fixed V and W, the problem is:
                # min_Uc sum_{i,j,k} (R[i,j,k] - Uc[i*m+j] * V[j*p+k,c] * W[k*n+i,c])^2
                # This is a weighted least squares problem.
                
                # Let's construct the linear system.
                # For each (i,j,k), the term is Uc[i*m+j] * (V[j*p+k,c] * W[k*n+i,c])
                # So we have a system: A * Uc = b where A is (n*m*p) x (n*m) and b is (n*m*p)
                
                # Build A and b
                A = np.zeros((n * m * p, n * m))
                b = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Residual without c
                            res = 0.0
                            for c2 in range(r):
                                if c2 == c:
                                    continue
                                res += U[i * m + j, c2] * V[j * p + k, c2] * W[k * n + i, c2]
                            target = 1.0
                            b[idx] = target - res
                            weight = V[j * p + k, c] * W[k * n + i, c]
                            A[idx, i * m + j] = weight
                            idx += 1
                
                # Solve least squares
                try:
                    Uc_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[:, c] = Uc_new
                except:
                    pass
                
                # Update V[:, c]
                old_Vc = V[:, c].copy()
                A = np.zeros((n * m * p, m * p))
                b = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res = 0.0
                            for c2 in range(r):
                                if c2 == c:
                                    continue
                                res += U[i * m + j, c2] * V[j * p + k, c2] * W[k * n + i, c2]
                            target = 1.0
                            b[idx] = target - res
                            weight = U[i * m + j, c] * W[k * n + i, c]
                            A[idx, j * p + k] = weight
                            idx += 1
                try:
                    Vc_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[:, c] = Vc_new
                except:
                    pass
                
                # Update W[:, c]
                old_Wc = W[:, c].copy()
                A = np.zeros((n * m * p, p * n))
                b = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res = 0.0
                            for c2 in range(r):
                                if c2 == c:
                                    continue
                                res += U[i * m + j, c2] * V[j * p + k, c2] * W[k * n + i, c2]
                            target = 1.0
                            b[idx] = target - res
                            weight = U[i * m + j, c] * V[j * p + k, c]
                            A[idx, k * n + i] = weight
                            idx += 1
                try:
                    Wc_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[:, c] = Wc_new
                except:
                    pass
                
                # Check if we improved
                cur_res = residual(U, V, W)
                if cur_res < prev_res - 1e-12:
                    improved = True
                    prev_res = cur_res
                
                if cur_res < tol:
                    return U, V, W, cur_res, True
            if not improved and it > 10:
                break
        return U, V, W, prev_res, False
    
    # Try ranks from 15 to 35 (schoolbook is 36)
    for r in range(15, 36):
        if time.time() - start_time > DEADLINE:
            break
        # Try multiple random initializations
        n_trials = 5
        for trial in range(n_trials):
            if time.time() - start_time > DEADLINE:
                break
            U, V, W = random_decomposition(r, scale=1.0)
            U, V, W, res, converged = local_search(U, V, W, max_iters=100)
            if converged and res < 1e-9:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        if best_rank == r:
            break
    
    # Convert to list format with half-integer entries if possible
    def to_half_integer_matrix(M):
        """Convert a numpy matrix to a list of lists with half-integer entries."""
        result = []
        for row in M:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                new_row.append(rounded)
            result.append(new_row)
        return result
    
    # Verify the best solution
    if verify(best_U, best_V, best_W, tol=1e-9):
        U_out = to_half_integer_matrix(best_U)
        V_out = to_half_integer_matrix(best_V)
        W_out = to_half_integer_matrix(best_W)
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback to schoolbook
        U_out = to_half_integer_matrix(U_school)
        V_out = to_half_integer_matrix(V_school)
        W_out = to_half_integer_matrix(W_school)
        return {"U": U_out, "V": V_out, "W": W_out}
