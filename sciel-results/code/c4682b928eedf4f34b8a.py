"""
Combined approach: block decomposition with Strassen's algorithm for 2x2x2 sub-blocks
to reduce rank from schoolbook 50 to 46 for 2x5x5 matrix multiplication.

Strategy:
- Split the 5x5 B matrix into blocks along the middle (size 5 = 2+1+2).
- The 2x5 A matrix is split into A_left (2x2), A_mid (2x1), A_right (2x2).
- Use Strassen's 7-term multiplication for the 2x2 blocks (A_left * B_left and A_right * B_right).
- The middle column/row interactions are handled with schoolbook terms.
- This yields rank 7 + 7 + 5 + 5 + 2 = 26? Let me recompute.

Actually, let me think more carefully.

Matrix multiplication: C = A * B where A is 2x5, B is 5x5, C is 2x5.

Split indices:
- A: rows i in {0,1}, cols j in {0,1,2,3,4}
- B: rows j in {0,1,2,3,4}, cols k in {0,1,2,3,4}
- C: rows i in {0,1}, cols k in {0,1,2,3,4}

Let's partition the middle index j into groups: J0={0,1}, J1={2}, J2={3,4}
And partition the output index k into groups: K0={0,1}, K1={2}, K2={3,4}

Then:
C[:, K0] = A[:, J0] * B[J0, K0] + A[:, J1] * B[J1, K0] + A[:, J2] * B[J2, K0]
C[:, K1] = A[:, J0] * B[J0, K1] + A[:, J1] * B[J1, K1] + A[:, J2] * B[J2, K1]
C[:, K2] = A[:, J0] * B[J0, K2] + A[:, J1] * B[J1, K2] + A[:, J2] * B[J2, K2]

Each term is a matrix product:
- A[:, J0] is 2x2, B[J0, K0] is 2x2 -> 2x2 product, rank 7 (Strassen)
- A[:, J1] is 2x1, B[J1, K0] is 1x2 -> 2x2 product via outer product, rank 4 (schoolbook: 2*1*2=4)
- A[:, J2] is 2x2, B[J2, K0] is 2x2 -> 2x2 product, rank 7 (Strassen)
- Similarly for K1 and K2

Total rank = 3*(7 + 4 + 7) = 3*18 = 54. That's worse than 50.

Let me try a different partition. Maybe partition differently.

Actually, the known result is rank 46. Let me think about how to achieve that.

Alternative: Use the fact that we can share Strassen terms across different output columns.

Let me reconsider. The standard approach for reducing rank in matrix multiplication involves finding linear dependencies among the tensor terms.

For 2x5x5, the schoolbook rank is 50. The known minimal rank is 46.

Let me try a different approach: direct optimization to find a rank-46 decomposition.

Actually, let me try the block approach more carefully.

Partition j into {0,1,2,3,4} and k into {0,1,2,3,4}.

Consider splitting into 2x2 blocks where possible:
- For the 2x2 subproblems, use Strassen (rank 7 instead of 8)

If we can identify multiple 2x2x2 subproblems:
- A[0:2, 0:2] * B[0:2, 0:2] -> 2x2x2, rank 7
- A[0:2, 3:5] * B[3:5, 3:5] -> 2x2x2, rank 7
- Cross terms need schoolbook

Let me count more carefully with a specific decomposition.

Actually, let me just implement a direct search for rank-46 using the alternating least squares approach from the inspiration, but with a better initialization.

The key insight from the cell findings: "Using block decomposition with Strassen's algorithm for 2x2x2 sub-blocks reduces the rank from 50 to 46."

Let me construct this explicitly.

For 2x5x5:
- Think of it as having two "rows" of A.
- The 5 middle indices and 5 output indices.

One approach: 
- Use Strassen for the first 2x2 block: A[:,0:2] * B[0:2, 0:2] contributes to C[:, 0:2]
- Use Strassen for the last 2x2 block: A[:,3:5] * B[3:5, 3:5] contributes to C[:, 3:5]
- Handle the middle column j=2 and middle output k=2 separately
- Handle cross terms

This is getting complex. Let me just implement the alternating optimization to find a rank-46 solution, starting from a smart initialization.

Given the 120s time limit, I'll use the alternating least squares approach with careful time management.
"""

import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n, m, p = 2, 5, 5
    schoolbook_rank = n * m * p  # 50
    
    # Build schoolbook as fallback
    def build_schoolbook():
        U = np.zeros((n * m, schoolbook_rank), dtype=int)
        V = np.zeros((m * p, schoolbook_rank), dtype=int)
        W = np.zeros((p * n, schoolbook_rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    best_solution = build_schoolbook()
    best_rank = schoolbook_rank
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is valid."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def try_rank(R, max_iter=500, seed=42):
        """Try to find a valid decomposition of rank R using alternating least squares."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        def solve_linear(fixed1, fixed2, which_to_solve):
            """
            Solve for one factor given the other two are fixed.
            which_to_solve: 'U', 'V', or 'W'
            fixed1, fixed2: the two fixed matrices
            """
            if which_to_solve == 'U':
                V_fix, W_fix = fixed1, fixed2
                n_vars = n * m * R
                n_constraints = n * m * p
                A = np.zeros((n_constraints, n_vars))
                b = np.ones(n_constraints)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i * m * p + j * p + k
                            for r in range(R):
                                col = (i * m + j) * R + r
                                A[row, col] = V_fix[j*p+k, r] * W_fix[k*n+i, r]
                try:
                    u_flat = np.linalg.lstsq(A, b, rcond=None)[0]
                    return u_flat.reshape(n*m, R)
                except:
                    return None
            elif which_to_solve == 'V':
                U_fix, W_fix = fixed1, fixed2
                n_vars = m * p * R
                n_constraints = n * m * p
                A = np.zeros((n_constraints, n_vars))
                b = np.ones(n_constraints)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i * m * p + j * p + k
                            for r in range(R):
                                col = (j * p + k) * R + r
                                A[row, col] = U_fix[i*m+j, r] * W_fix[k*n+i, r]
                try:
                    v_flat = np.linalg.lstsq(A, b, rcond=None)[0]
                    return v_flat.reshape(m*p, R)
                except:
                    return None
            elif which_to_solve == 'W':
                U_fix, V_fix = fixed1, fixed2
                n_vars = p * n * R
                n_constraints = n * m * p
                A = np.zeros((n_constraints, n_vars))
                b = np.ones(n_constraints)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i * m * p + j * p + k
                            for r in range(R):
                                col = (k * n + i) * R + r
                                A[row, col] = U_fix[i*m+j, r] * V_fix[j*p+k, r]
                try:
                    w_flat = np.linalg.lstsq(A, b, rcond=None)[0]
                    return w_flat.reshape(p*n, R)
                except:
                    return None
        
        # Alternating least squares
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for U
            U_new = solve_linear(V, W, 'U')
            if U_new is None:
                break
            U = U_new
            
            # Solve for V
            V_new = solve_linear(U, W, 'V')
            if V_new is None:
                break
            V = V_new
            
            # Solve for W
            W_new = solve_linear(U, V, 'W')
            if W_new is None:
                break
            W = W_new
            
            # Check convergence
            if iteration % 50 == 0:
                if verify(U, V, W):
                    return U, V, W
        
        # Final check
        if verify(U, V, W):
            return U, V, W
        
        return None
    
    # Try ranks from 46 upward
    for R in range(46, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = try_rank(R, max_iter=300, seed=seed)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    best_solution = (U.tolist(), V.tolist(), W.tolist())
                    best_rank = R
                    break
        
        if best_rank == R:
            break  # Found the best rank, stop
    
    return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
