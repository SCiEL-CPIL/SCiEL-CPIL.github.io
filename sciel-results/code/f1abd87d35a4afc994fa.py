"""Improved program: attempts a lower-rank exact decomposition for 3x3*3x6."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# Known optimal rank for M_{3,3,3} is 23 (Bürgisser).
# For M_{3,3,6}, the known lower bound is 44. We aim for rank 44.
# A direct construction is non-trivial. We will use a randomized search
# to find a decomposition of rank 44. If it fails, we fallback to 54.

def construct():
    deadline = time.time() + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1 if (i, j, k) is a valid multiplication index
    # Here i is row of A (0..n-1), j is col of A (0..m-1), k is col of B (0..p-1)
    # Wait, the tensor for matrix multiplication A(n,m) * B(m,p) -> C(n,p)
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices (i, j, k) such that A[i,j], B[j,k], C[i,k] are involved.
    # T[i, j, k] = 1.
    
    # Factor matrices:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # Index mapping:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]  <-- Note: task says W[k*n+i, r]. This is (k, i) index.
    # So W is indexed by (k, i) where k is col of B (and C) and i is row of A (and C).
    # This matches C[i, k].
    
    def check_decomp(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Reconstruct T from factors:
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want T[i, j, k] == 1 for all valid i,j,k.
        # Valid i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # Since dimensions are small, we can just check all.
        r = U.shape[1]
        # Efficient check using einsum?
        # Let's just loop or vectorize.
        # T_calc[i, j, k] = einsum('ac,bc,dc->', ...) no, it's per (i,j,k).
        # T_calc is a tensor of size n x m x p.
        # T_calc[i, j, k] = U[i*m+j, :] @ V[j*p+k, :] * W[k*n+i, :]? No.
        # It is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # We can build the tensor by iterating c?
        # Or using broadcasting.
        # Let's create a 3D array.
        # This is a rank-1 sum of products of slices.
        # For each c, we add U[:, c] * V[:, c] * W[:, c] but the indices are mixed.
        # U row idx: u_idx = i*m + j
        # V row idx: v_idx = j*p + k
        # W row idx: w_idx = k*n + i
        
        # This is complex to vectorize directly without loops if indices are not aligned.
        # But n,m,p are small (3,3,6). 54 entries.
        # We can just loop i, j, k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=1e-8):
                        return False
        return True

    # Try rank 44
    # Random search with restarts
    # The space of rank-r decompositions is non-convex.
    # We can try to find a solution by random initialization and gradient descent?
    # Or random sampling? Random sampling in high dimensions is unlikely to hit the exact solution.
    # However, for r=44, the degrees of freedom are 44 * (9+18+18) = 44*45 = 1980.
    # Constraints: 54 equations.
    # So it is highly underdetermined. There should be many solutions.
    # But they are sparse/structured? No, the tensor is dense (all 1s).
    
    # Let's try a simple heuristic:
    # Start with a random rank-44 decomposition and use a solver?
    # Since we can't use symbolic solvers easily, we can use `scipy.optimize.least_squares`?
    # Minimize sum (sum_c U... - 1)^2.
    
    from scipy.optimize import least_squares
    
    def residuals(params, r):
        # params: flattened U, V, W
        # U: 9*r, V: 18*r, W: 18*r
        total_size = (9 + 18 + 18) * r
        if len(params) != total_size:
            return None
        
        U = params[0:9*r].reshape(9, r)
        V = params[9*r:9*r+18*r].reshape(18, r)
        W = params[9*r+18*r:9*r+18*r+18*r].reshape(18, r)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def find_decomp(rank, max_iter=100, timeout=100):
        start_time = time.time()
        for attempt in range(max_iter):
            if time.time() > start_time + timeout:
                return None
            
            # Random init
            # Scale matters. If we start with 0, gradient is 0.
            # Start with small random numbers.
            r = rank
            size_U = 9 * r
            size_V = 18 * r
            size_W = 18 * r
            params = np.random.randn(size_U + size_V + size_W) * 0.1
            
            # Use least_squares to minimize sum of squares
            try:
                result = least_squares(residuals, params, args=(rank,), method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12)
            except:
                continue
                
            if result.success and result.cost < 1e-10:
                # Verify
                U = result.x[0:size_U].reshape(9, rank)
                V = result.x[size_U:size_U+size_V].reshape(18, rank)
                W = result.x[size_U+size_V:size_U+size_V+size_W].reshape(18, rank)
                
                if check_decomp(U, V, W):
                    # Round to rational? The task allows integer/half-integer.
                    # "exactly decompose" usually implies exact match in evaluation.
                    # If we return floats, will it pass?
                    # "with integer/half-integer entries" -> This is a constraint on the OUTPUT.
                    # We must return integers or half-integers.
                    # So we need to find a solution with entries in {0.5 * Z}.
                    # This is a Diophantine-like problem.
                    # Least squares finds real solutions.
                    # We might need to search for rational solutions specifically.
                    # Or, the solution found might be close to rational?
                    # Given the small rank and simple tensor, rational solutions likely exist.
                    
                    # Let's try to round to nearest 0.5 and check if it still works.
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if check_decomp(U_r, V_r, W_r):
                        # Convert to lists of floats (or ints if whole)
                        # The interface expects lists of lists.
                        U_list = U_r.tolist()
                        V_list = V_r.tolist()
                        W_list = W_r.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                    
                    # If rounding didn't work, maybe the solution is not close to half-integers?
                    # But we can try to solve for integer/half-integer directly?
                    # That's harder.
                    # For now, if we find a real solution, it's a good sign.
                    # But we MUST return valid types.
                    
        return None

    # Try rank 44
    decomp = find_decomp(44, max_iter=50, timeout=80)
    if decomp:
        return decomp

    # Fallback to schoolbook 54
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
