import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110
    
    def schoolbook_decomposition():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, r):
        """Verify that U, V, W give exact decomposition of rank r."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U_arr[i * m + j, r_idx] * V_arr[j * p + k, r_idx] * W_arr[k * n + i, r_idx]
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def try_rank_bilinear(r, num_attempts=200):
        """
        Try to find a rank-r decomposition by fixing U and V randomly,
        then solving for W linearly.
        
        For fixed U and V, the equations are linear in W:
        For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        But W[k*n+i,r] appears in multiple equations (for different j).
        Let's collect all unknowns: W has p*n*r entries.
        We have n*m*p equations.
        
        For each equation (i,j,k), the coefficient of W[k*n+i,r] is U[i*m+j,r]*V[j*p+k,r].
        """
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Create random sparse U and V with integer entries
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            
            # Each row of U and V gets a few non-zero entries
            for row in range(n * m):
                nnz = random.randint(1, min(4, r))
                cols = random.sample(range(r), nnz)
                for c in cols:
                    U[row, c] = random.choice([-2, -1, 1, 2])
            
            for row in range(m * p):
                nnz = random.randint(1, min(4, r))
                cols = random.sample(range(r), nnz)
                for c in cols:
                    V[row, c] = random.choice([-2, -1, 1, 2])
            
            # Build linear system A @ w_vec = b
            # Unknowns: W flattened to (p*n*r,)
            # Equation (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            num_eqs = n * m * p
            num_unk = p * n * r
            
            A = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            col = (k * n + i) * r + r_idx
                            A[eq_idx, col] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
                        eq_idx += 1
            
            # Solve the linear system
            try:
                # Use least squares
                w_vec, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if residual is small
                if residuals.size > 0:
                    if residuals[0] > 1e-10:
                        continue
                
                # Verify exactness
                w_vec = w_vec.reshape(p * n, r)
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r_idx in range(r):
                                val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * w_vec[k * n + i, r_idx]
                            if abs(val - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Try to round to nice values
                    w_rounded = np.round(w_vec * 2) / 2  # half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    
                    # Verify rounded version
                    valid_rounded = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for r_idx in range(r):
                                    val += U_rounded[i * m + j, r_idx] * V_rounded[j * p + k, r_idx] * w_rounded[k * n + i, r_idx]
                                if abs(val - 1.0) > 1e-8:
                                    valid_rounded = False
                                    break
                            if not valid_rounded:
                                break
                        if not valid_rounded:
                            break
                    
                    if valid_rounded:
                        U_list = [[float(U_rounded[i, r_idx]) for r_idx in range(r)] for i in range(n * m)]
                        V_list = [[float(V_rounded[i, r_idx]) for r_idx in range(r)] for i in range(m * p)]
                        W_list = [[float(w_rounded[i, r_idx]) for r_idx in range(r)] for i in range(p * n)]
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Use unrounded
                        U_list = [[float(U[i, r_idx]) for r_idx in range(r)] for i in range(n * m)]
                        V_list = [[float(V[i, r_idx]) for r_idx in range(r)] for i in range(m * p)]
                        W_list = [[float(w_vec[i, r_idx]) for r_idx in range(r)] for i in range(p * n)]
                        return {"U": U_list, "V": V_list, "W": W_list}
            except:
                continue
        
        return None
    
    def try_rank_symmetric(r, num_attempts=100):
        """
        Try with U=V=W structure for symmetry, or use Strassen-like patterns.
        """
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Try a structured approach: use the fact that for matrix multiplication,
            # we can sometimes share computation.
            
            # Create U, V, W with structured patterns
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Fill with random values but with some structure
            for r_idx in range(r):
                # Each rank term picks random indices
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                
                # Sometimes use the same i,j,k (schoolbook-like)
                if random.random() < 0.5:
                    U[i * m + j, r_idx] = 1
                    V[j * p + k, r_idx] = 1
                    W[k * n + i, r_idx] = 1
                else:
                    # Try different combinations
                    U[i * m + j, r_idx] = random.choice([-1, 1])
                    V[j * p + k, r_idx] = random.choice([-1, 1])
                    W[k * n + i, r_idx] = random.choice([-1, 1])
            
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist(), r):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from 44 down
    best_result = None
    for r in range(rank_schoolbook - 1, 10, -1):
        if time.time() > deadline:
            break
        
        # Try bilinear approach
        result = try_rank_bilinear(r, num_attempts=100)
        if result is not None:
            best_result = result
            break
        
        # Try symmetric approach
        result = try_rank_symmetric(r, num_attempts=50)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
