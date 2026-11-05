"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import time
import random
import numpy as np

DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45

    # Build the target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The tensor is of shape (n, m, p) and represents the matrix multiplication tensor.
    # For a decomposition of rank r, we have matrices U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all (i,j,k): T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Initialize with the schoolbook decomposition
    def make_schoolbook():
        r = rank_schoolbook
        U = np.zeros((n*m, r), dtype=np.float64)
        V = np.zeros((m*p, r), dtype=np.float64)
        W = np.zeros((p*n, r), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    U, V, W = make_schoolbook()
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # The tensor T is all ones for valid (i,j,k) indices
    # T[i,j,k] = 1 if 0<=i<n, 0<=j<m, 0<=k<p
    
    # We want to find a low-rank decomposition.
    # Strategy: Use ALS (Alternating Least Squares) to refine and then try to reduce rank.
    
    # First, let's try to find a good rank-40 or lower decomposition by random initialization
    # and ALS, then check if we can get exact.
    
    def tensor_residual(U_mat, V_mat, W_mat, r):
        """Compute the Frobenius norm of the residual between the tensor and the decomposition."""
        # The decomposition gives T_decomp[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # We want this to equal 1 for all valid (i,j,k).
        total_sq = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    total_sq += (val - 1.0)**2
        return np.sqrt(total_sq)

    def als_step(U_mat, V_mat, W_mat, r, iters=200):
        """Run ALS to minimize the residual. Returns updated U, V, W."""
        U_mat = U_mat.copy()
        V_mat = V_mat.copy()
        W_mat = W_mat.copy()
        
        for it in range(iters):
            if time.time() > DEADLINE:
                break
            # Update U: for each (i,j), solve for U[i*m+j, :] given V, W
            for i in range(n):
                for j in range(m):
                    # We need U[i*m+j, c] such that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
                    # This is a linear system: A * u = b where A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # and b[k] = 1 for all k in range(p)
                    A = np.zeros((p, r))
                    b = np.ones(p)
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    # Solve A * u = b in least squares sense
                    try:
                        u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    U_mat[i*m+j, :] = u
            
            # Update V: for each (j,k), solve for V[j*p+k, :] given U, W
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    b = np.ones(n)
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = U_mat[i*m+j, c] * W_mat[k*n+i, c]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    V_mat[j*p+k, :] = v
            
            # Update W: for each (k,i), solve for W[k*n+i, :] given U, V
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U_mat[i*m+j, c] * V_mat[j*p+k, c]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    W_mat[k*n+i, :] = w
        
        return U_mat, V_mat, W_mat

    # Try various ranks from low to high
    # Theoretical lower bound: for 3x3x5, the rank is at least 12 (from 3x3x3 case) but could be higher.
    # Known: rank(3x3x3) = 19 (over reals, actually 19 is for complex, over reals it might differ)
    # Actually, the rank of 3x3x3 matrix multiplication tensor is 19 over any field of characteristic 0? 
    # No, it's known to be 19 for 3x3x3. For 3x3x5, we need to figure out.
    
    # Let's try ranks from 15 up to 45
    random.seed(42)
    
    for target_rank in range(15, rank_schoolbook + 1):
        if time.time() > DEADLINE:
            break
        
        # Try a few random initializations for this rank
        attempts = 3 if target_rank <= 30 else 1
        for attempt in range(attempts):
            if time.time() > DEADLINE:
                break
            
            # Initialize with random small values
            U_r = np.random.randn(n*m, target_rank) * 0.1
            V_r = np.random.randn(m*p, target_rank) * 0.1
            W_r = np.random.randn(p*n, target_rank) * 0.1
            
            # Run ALS
            U_r, V_r, W_r = als_step(U_r, V_r, W_r, target_rank, iters=100)
            
            # Check residual
            res = tensor_residual(U_r, V_r, W_r, target_rank)
            if res < 1e-10:
                # Found an exact decomposition!
                best_rank = target_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                break
        
        if best_rank <= target_rank:
            break
    
    # Convert to lists of integers or half-integers if possible
    # For now, just return the best we found
    # The problem says entries should be integer/half-integer, but let's first get a valid decomposition
    # We can round to nearest half-integer and check if it's still exact
    
    def to_half_integer(mat):
        """Round to nearest half-integer and return as list of lists."""
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2.0
                new_row.append(rounded)
            result.append(new_row)
        return result

    # Check if the best decomposition is valid (residual close to 0)
    final_res = tensor_residual(best_U, best_V, best_W, best_rank)
    
    if final_res < 1e-8:
        # Convert to lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = make_schoolbook()
        U_list = [[float(x) for x in row] for row in U_sb]
        V_list = [[float(x) for x in row] for row in V_sb]
        W_list = [[float(x) for x in row] for row in W_sb]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
