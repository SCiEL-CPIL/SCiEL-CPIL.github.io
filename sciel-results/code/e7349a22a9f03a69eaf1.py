import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    """
    Finds a low-rank exact tensor decomposition for 3x3x6 matrix multiplication.
    
    Strategy:
    1. Generate the schoolbook rank-54 solution as a safe fallback.
    2. Search for a rank-13 or 14 decomposition of the 3x3x2 tensor.
       - If rank 13 is found, total rank is 3 * 13 = 39.
       - If rank 14 is found, total rank is 3 * 14 = 42.
       - If rank 15 is found, total rank is 3 * 15 = 45.
    3. The search exploits the structure: fix U and V randomly, solve for W,
       then check if W entries are half-integers.
    
    The 3x3x6 tensor is a direct sum of three 3x3x2 tensors (splitting B and C into blocks of width 2).
    Thus, if we have a rank-R decomposition for 3x3x2, we get a rank 3R decomposition for 3x3x6.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = 3, 3, 6

    def generate_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = generate_schoolbook()
    best_rank = n * m * p

    def check_half_integer(arr, tol=1e-8):
        """
        Checks if array entries are close to half-integers.
        Returns rounded half-integer array if valid, else None.
        """
        doubled = arr * 2.0
        rounded = np.round(doubled)
        if np.max(np.abs(doubled - rounded)) < tol:
            return rounded / 2.0
        return None

    def find_3x3x2_decomposition(R, max_attempts=2000, rand_range=2):
        """
        Search for rank-R decomposition of 3x3x2.
        Returns (U, V, W) as lists or None.
        U: 9xR, V: 6xR, W: 6xR
        """
        n2, m2, p2 = 3, 3, 2
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization with small integers
            # Using a wider range might help find non-trivial solutions
            U_r = np.random.randint(-rand_range, rand_range + 1, size=(n2 * m2, R))
            V_r = np.random.randint(-rand_range, rand_range + 1, size=(m2 * p2, R))
            
            W_r = np.zeros((p2 * n2, R))
            valid = True
            
            # Solve for W
            # Equation: sum_r U[i,j,r] V[j',k,r] W[k',i',r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
            # For fixed i, k:
            # sum_r U[i,j,r] V[j',k,r] W[i,k,r] = delta_{j,j'}  (Note: W index is k*n+i, but here k is fixed for this block? No.)
            
            # Let's re-verify indices for 3x3x2:
            # U[i, j, r] where i in 0..2, j in 0..2. Index: i*3+j
            # V[j, k, r] where j in 0..2, k in 0..1. Index: j*2+k
            # W[k, i, r] where k in 0..1, i in 0..2. Index: k*3+i
            
            # Tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            
            # For fixed i, k:
            # We have equations for j, j'.
            # sum_r U[i,j,r] V[j',k,r] W[k,i,r] = delta_{j,j'}
            # Let W_vec = W[k, :, :] which is a vector of length R? No, W[k,i,r] has fixed k, variable i and r.
            # Wait, the equation for fixed i, k involves W[k, i, r].
            # The LHS is a sum over r of (U[i,j,r] * V[j',k,r]) * W[k,i,r].
            # This is a linear system for W[k,i,:] (which is a vector of length R).
            # Let M be a matrix of size (m2*m2) x R.
            # M[j, j', :] = U[i,j,:] * V[j',k,:]
            # RHS is delta_{j,j'}
            
            for i in range(n2):
                for k in range(p2):
                    M = np.zeros((m2 * m2, R))
                    for j in range(m2):
                        for jp in range(m2):
                            M[j * m2 + jp, :] = U_r[i * m2 + j, :] * V_r[jp * p2 + k, :]
                    
                    rhs = np.zeros(m2 * m2)
                    for j in range(m2):
                        rhs[j * m2 + j] = 1.0
                    
                    try:
                        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if w_vec is half-integer
                    w_check = check_half_integer(w_vec)
                    if w_check is None:
                        valid = False
                        break
                    else:
                        W_r[k * n2 + i, :] = w_check
            
            if valid:
                # Verify full tensor
                U_np = U_r.astype(float)
                V_np = V_r.astype(float)
                W_np = W_r.astype(float)
                
                correct = True
                for i in range(n2):
                    for ip in range(n2):
                        for k in range(p2):
                            for kp in range(p2):
                                for j in range(m2):
                                    for jp in range(m2):
                                        expected = 1.0 if (i == ip and k == kp and j == jp) else 0.0
                                        actual = 0.0
                                        for r in range(R):
                                            actual += U_np[i * m2 + j, r] * V_np[jp * p2 + k, r] * W_np[kp * n2 + ip, r]
                                        if abs(actual - expected) > 1e-6:
                                            correct = False
                                            break
                                    if not correct: break
                                if not correct: break
                            if not correct: break
                        if not correct: break
                    if not correct: break
                if correct:
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        return None

    # Try to find 3x3x2 decomposition
    # Prioritize the known optimal rank 15, then try lower ranks
    # Using a wider range for random integers might help find the solution
    for R in [15, 14, 13]:
        if time.time() > deadline - 10:
            break
        
        # Use more attempts for rank 15, and a wider range of integers
        if R == 15:
            max_attempts = 2000
            rand_range = 3  # -2 to 2
        else:
            max_attempts = 500
            rand_range = 2  # -1 to 1
            
        # Modify the search function call to use the new range
        # We need to update find_3x3x2_decomposition to accept rand_range
        decomp = find_3x3x2_decomposition(R, max_attempts=max_attempts, rand_range=rand_range)
        
        if decomp is not None:
            U_core, V_core, W_core = decomp
            total_rank = 3 * R
            
            if total_rank < best_rank:
                # Extend to 3x3x6
                U_tot = [[0] * total_rank for _ in range(n * m)]
                V_tot = [[0] * total_rank for _ in range(m * p)]
                W_tot = [[0] * total_rank for _ in range(p * n)]
                
                # 3 blocks of 3x3x2
                for block in range(3):
                    offset = block * R
                    for i in range(3):
                        for j in range(3):
                            for r in range(R):
                                U_tot[i * 3 + j][offset + r] = U_core[i * 3 + j][r]
                    
                    for j in range(3):
                        for k in range(2):
                            for r in range(R):
                                V_tot[j * 6 + block * 2 + k][offset + r] = V_core[j * 2 + k][r]
                    
                    for i in range(3):
                        for k in range(2):
                            for r in range(R):
                                W_tot[(block * 2 + k) * 3 + i][offset + r] = W_core[k * 3 + i][r]
                
                best_result = {"U": U_tot, "V": V_tot, "W": W_tot}
                best_rank = total_rank
                break

    return best_result
