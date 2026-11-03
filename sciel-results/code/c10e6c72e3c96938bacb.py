"""Tensor decomposition search for matrix multiplication 4x4 * 4x8."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # n=4, m=4, p=8
    # U: 16 x r
    # V: 32 x r
    # W: 32 x r
    
    # The tensor T[i, j, k] = delta_{i, i'} delta_{j, j'} delta_{k, k'} ... no, standard matmul.
    # (A B)[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T has indices (i, j, k) where A is (i, j), B is (j, k).
    # Wait, the problem says: "multiplying a 4 by 4 matrix by a 4 by 8 matrix"
    # A is 4x4, B is 4x8.
    # Result C is 4x8.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Indices: i in [0,3], j in [0,3], k in [0,7].
    # U index: i*m + j ? No, U shape is (n*m, rank).
    # Let's check the seed code:
    # U[i * m + j][column] = 1
    # V[j * p + k][column] = 1
    # W[k * n + i][column] = 1
    # n=4, m=4, p=8.
    # U row: i*m+j. i in 0..3, j in 0..3. Total 16 rows. Correct.
    # V row: j*p+k. j in 0..3, k in 0..7. Total 32 rows. Correct.
    # W row: k*n+i. k in 0..7, i in 0..3. Total 32 rows. Correct.
    # The tensor entry T[i, j, k] is 1 if it contributes to C[i, k] via A[i, j] B[j, k].
    # Yes, T[i, j, k] = 1 for all valid i, j, k in the standard decomposition.
    # Actually, the tensor for matmul is T[i, j, k] = 1.
    # The decomposition is: T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c] ?
    # No, U is (n*m, rank). So U[row_u, c]. row_u = i*m + j.
    # V is (m*p, rank). row_v = j*p + k.
    # W is (p*n, rank). row_w = k*n + i.
    
    # Let's verify the product.
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # In schoolbook: c = (i, j, k).
    # U[i*m+j, (i,j,k)] = 1
    # V[j*p+k, (i,j,k)] = 1
    # W[k*n+i, (i,j,k)] = 1
    # Product is 1.
    # Are there other c' that contribute?
    # U[i*m+j, c'] is non-zero only if the first part of c' matches i*m+j?
    # In schoolbook, each column c is unique for a triple (i, j, k).
    # So yes, this is the standard tensor.
    
    # We want to minimize rank r.
    # Current best known for 4x4x4 is 49 (Landsberg).
    # For 4x4x8, it's related.
    # We can try to find a decomposition with rank < 128 (schoolbook).
    # Even rank 100 would be a huge improvement.
    
    # Strategy:
    # 1. Start with a random sparse initialization or a structured one.
    # 2. Use a local search / gradient-free optimization to minimize the rank?
    #    No, rank is discrete. We fix a target rank R and try to find U, V, W of that rank.
    #    If we fail, increase R.
    #    But how to find U, V, W?
    #    This is a hard NP-hard problem in general.
    #    However, for small dimensions, we can use heuristics.
    #    
    #    Heuristic:
    #    Initialize U, V, W with small random integers.
    #    Compute the tensor approximation.
    #    Use a "peeling" or "matching" approach?
    #    
    #    Alternative: Use the known structure of matrix multiplication tensors.
    #    They are symmetric in some ways (if square). Here not square.
    #    
    #    Let's try a simpler approach:
    #    We know that the rank of matmul (n x m) * (m x p) is at most n*m*p.
    #    It is also known to be at least n*m + m*p + p*n - 2 (for n,m,p >= 2? No, that's for something else).
    #    Lower bound: max(n*m, m*p, p*n). Here max(16, 32, 32) = 32.
    #    So rank is at least 32.
    #    Schoolbook is 128.
    #    Can we get close to 32? Unlikely.
    #    Can we get close to 64?
    #    
    #    Let's try to construct a decomposition of rank R by solving a system of equations?
    #    For each (i, j, k), sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
    #    This is a system of n*m*p = 128 equations.
    #    Variables: 16*R + 32*R + 32*R = 80*R.
    #    If R=16, 1280 variables. 128 equations. Underdetermined.
    #    But it's non-linear (product of 3 variables).
    #    
    #    We can use a randomized local search:
    #    1. Pick R.
    #    2. Initialize U, V, W randomly (e.g., from {-1, 0, 1}).
    #    3. Calculate error: for each (i, j, k), err = sum_c U...V...W... - 1.
    #    4. If error is 0, we found a solution.
    #    5. Otherwise, modify one entry to reduce error.
    #    
    #    This is similar to finding a solution to a 3-XOR or similar constraint satisfaction, but with products.
    #    
    #    Let's try R=100 first.
    #    We have plenty of time (140s).
    #    
    #    Better idea: Use the fact that we can scale.
    #    If we find a rational solution, we can clear denominators.
    #    
    #    Let's implement a simple hill-climbing / local search.
    #    
    #    Objective: Minimize sum_{i,j,k} (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1)^2
    #    
    #    We can use gradient descent if we treat entries as continuous, then round?
    #    But rounding might break it.
    #    
    #    Let's stick to discrete search on a subset of variables.
    #    
    #    Actually, there is a known trick:
    #    The tensor is sparse.
    #    
    #    Let's try a specific structured ansatz.
    #    For square matrices, the rank is related to the tensor rank of matrix multiplication.
    #    For 4x4x4, rank 49 is known.
    #    For 4x4x8, we can view it as two 4x4x4 multiplications?
    #    A (4x4) * B (4x8).
    #    Split B into B1 (4x4) and B2 (4x4).
    #    C1 = A * B1, C2 = A * B2.
    #    Rank(A*B1) + Rank(A*B2) <= Rank(A*B).
    #    If Rank(4x4x4) = 49, then Rank(4x4x8) <= 98.
    #    This is a valid upper bound!
    #    Let's verify if we can construct it.
    #    
    #    We need a decomposition for 4x4x4.
    #    The seed program gives schoolbook (rank 64 for 4x4x4).
    #    We don't have the rank-49 decomposition hardcoded.
    #    Can we find it?
    #    
    #    If we just use the sum of two schoolbook decompositions:
    #    Rank = 64 + 64 = 128. No improvement.
    #    
    #    We need a better decomposition for 4x4x4.
    #    
    #    Let's try to find a low-rank decomposition for 4x4x4 using local search.
    #    Dimensions: n=4, m=4, p=4.
    #    U: 16 x r, V: 16 x r, W: 16 x r.
    #    Target rank: 50? 49?
    #    
    #    If we find a rank 49 decomposition for 4x4x4, we can combine two to get rank 98 for 4x4x8.
    #    Score = 98/128 = 0.765.
    #    Current score 1.0 (rank 128).
    #    This would be a win.
    #    
    #    How to find rank 49 for 4x4x4?
    #    This is a known result (Landsberg 1994).
    #    Can we search for it?
    #    128 equations, 3*16*49 = 2352 variables.
    #    Very underdetermined.
    #    
    #    Let's try a random search for a solution to the 4x4x4 tensor equation with rank R.
    #    We'll start with R=50 and try to satisfy the equations.
    #    
    #    Algorithm for 4x4x4:
    #    1. Initialize U, V, W with random integers in {-1, 0, 1}.
    #    2. Compute residual for each (i, j, k).
    #    3. If any residual is non-zero, pick one.
    #    4. Try to change one variable to reduce the error.
    #    5. Repeat.
    #    
    #    This is a CSP.
    #    
    #    Let's write a function to solve for a given n, m, p, R.
    
    def solve_tensor(n, m, p, R, max_iters=100000, timeout=10):
        # Returns U, V, W if found, else None
        start = time.time()
        # Initialize
        U = np.zeros((n*m, R), dtype=int)
        V = np.zeros((m*p, R), dtype=int)
        W = np.zeros((p*n, R), dtype=int)
        
        # Random init
        U[:] = np.random.choice([-1, 0, 1], size=(n*m, R))
        V[:] = np.random.choice([-1, 0, 1], size=(m*p, R))
        W[:] = np.random.choice([-1, 0, 1], size=(p*n, R))
        
        # Precompute indices
        # For each (i, j, k), we need:
        # u_idx = i*m + j
        # v_idx = j*p + k
        # w_idx = k*n + i
        
        # We want sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c] == 1
        
        # To speed up, we can maintain the current sum for each (i, j, k).
        # S[i, j, k] = sum_c ...
        
        S = np.zeros((n, m, p), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    # S[i, j, k] = dot(U[u], V[v] * W[w]) ? No, element-wise product then sum.
                    # S[i, j, k] = sum_c U[u, c] * V[v, c] * W[w, c]
                    # This is slow to compute from scratch.
                    # We can update it incrementally.
        
        # Recompute S initially
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    S[i, j, k] = np.sum(U[u] * V[v] * W[w])
        
        # Count errors
        errors = np.sum(S != 1)
        
        if errors == 0:
            return U, V, W
        
        # Local search
        # We pick a random (i, j, k) with error, and try to fix it by changing one variable.
        # Changing U[u, c] affects S[i', j, k'] where u = i'*m + j.
        # i' is fixed? No, u = i*m+j. For a fixed u, i is fixed?
        # u = i*m + j. Since m=4, i = u // 4, j = u % 4.
        # So changing U[u, c] affects all (i, j, k) where i = u//4, j = u%4.
        # Specifically, for all k.
        # So it affects p values of S[i, j, k].
        
        # Similarly for V and W.
        
        # Let's implement a simple hill climbing.
        # We will try to reduce the number of errors.
        
        # To make it faster, we can use a "simulated annealing" style or just greedy.
        # Given the underdetermined nature, greedy might get stuck.
        
        # Let's try a different approach:
        # Fix U and V, solve for W?
        # W is the only unknown?
        # S[i, j, k] = sum_c U[u, c] V[v, c] W[w, c]
        # Let A_c = U[u, c] V[v, c].
        # Then S[i, j, k] = sum_c A_c W[w, c].
        # For a fixed (i, j), w depends on k.
        # w = k*n + i.
        # So for fixed i, j, we have a system for k=0..p-1.
        # S[i, j, k] = sum_c A_c W[k*n + i, c]
        # This is a linear system in W!
        # If we fix U and V, we can solve for W.
        
        # Let's try this:
        # 1. Randomly initialize U and V.
        # 2. For each (i, j), set up the linear system for W.
        #    Variables: W[0*n+i, c], W[1*n+i, c], ..., W[(p-1)*n+i, c] for c=0..R-1.
        #    Total p*R variables for this (i, j) block?
        #    No, W is shared across different (i, j) for the same k?
        #    W index is k*n + i.
        #    For a fixed i, the rows of W involved are i, n+i, 2n+i, ..., (p-1)n+i.
        #    These are distinct for different k.
        #    So for each i, we have a set of W rows.
        #    The equations for S[i, j, k] involve W rows for that i.
        #    So we can solve for W for each i independently?
        #    Yes!
        #    For each i in 0..n-1:
        #       Variables: W[k*n + i, c] for k in 0..p-1, c in 0..R-1.
        #       Equations: For each j in 0..m-1, k in 0..p-1:
        #           sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        #       This is a linear system with p*R unknowns and m*p equations.
        #       If p*R > m*p, i.e., R > m, it is underdetermined.
        #       We can find a solution if the system is consistent.
        #       We can use least squares or find a particular solution.
        
        # This is a great structural insight!
        # We can iterate:
        # 1. Init U, V randomly.
        # 2. Solve for W (linear).
        # 3. Init U, W randomly.
        # 4. Solve for V (linear).
        # 5. Init V, W randomly.
        # 6. Solve for U (linear).
        # Repeat until convergence or error is 0.
        
        # Let's implement this.
        
        def solve_linear_block(A, b, R):
            # A is (num_eq, p*R)
            # We want x such that A x = b
            # x is the flattened W block for a specific i.
            # We can use np.linalg.lstsq
            # But we want integer solutions?
            # If we use floating point, we might get non-integers.
            # We can round and check.
            # Or we can use rational arithmetic?
            # Let's try float and rounding first.
            
            sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Round to nearest integer
            sol_int = np.round(sol).astype(int)
            # Check error
            err = np.sum(np.abs(A @ sol_int - b))
            return sol_int, err
        
        # Main loop
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for iter in range(50):
            if time.time() - start > timeout:
                break
            
            # Initialize U, V
            if iter == 0:
                U[:] = np.random.choice([-1, 0, 1], size=(n*m, R))
                V[:] = np.random.choice([-1, 0, 1], size=(m*p, R))
            
            # Solve for W
            W_new = np.zeros_like(W)
            total_err = 0
            for i in range(n):
                # Variables: W[k*n + i, c] for k=0..p-1, c=0..R-1
                # Flatten: index = k*R + c
                # Equations: for each j, k
                A_block = np.zeros((m*p, p*R))
                b_block = np.ones(m*p)
                eq_idx = 0
                for j in range(m):
                    for k in range(p):
                        u = i*m + j
                        v = j*p + k
                        # Coefficients for W[k*n + i, c]
                        # A_block[eq_idx, k*R + c] = U[u, c] * V[v, c]
                        coeff = U[u] * V[v]
                        A_block[eq_idx, k*R:(k+1)*R] = coeff
                        eq_idx += 1
                
                sol, err = solve_linear_block(A_block, b_block, R)
                total_err += err
                # Unflatten sol into W
                for k in range(p):
                    W_new[k*n + i] = sol[k*R:(k+1)*R]
            
            W[:] = W_new
            if total_err == 0:
                return U, V, W
            
            # Solve for V
            V_new = np.zeros_like(V)
            total_err = 0
            for j in range(m):
                # Variables: V[j*p + k, c] for k=0..p-1, c=0..R-1
                # Equations: for each i, k
                A_block = np.zeros((n*p, p*R))
                b_block = np.ones(n*p)
                eq_idx = 0
                for i in range(n):
                    for k in range(p):
                        u = i*m + j
                        w = k*n + i
                        # Coefficients for V[j*p + k, c]
                        # A_block[eq_idx, k*R + c] = U[u, c] * W[w, c]
                        coeff = U[u] * W[w]
                        A_block[eq_idx, k*R:(k+1)*R] = coeff
                        eq_idx += 1
                
                sol, err = solve_linear_block(A_block, b_block, R)
                total_err += err
                for k in range(p):
                    V_new[j*p + k] = sol[k*R:(k+1)*R]
            
            V[:] = V_new
            if total_err == 0:
                return U, V, W
            
            # Solve for U
            U_new = np.zeros_like(U)
            total_err = 0
            for i in range(n):
                for j in range(m):
                    # Variables: U[i*m + j, c] for c=0..R-1
                    # Equations: for each k
                    A_block = np.zeros((p, R))
                    b_block = np.ones(p)
                    for k in range(p):
                        v = j*p + k
                        w = k*n + i
                        # A_block[k, c] = V[v, c] * W[w, c]
                        A_block[k] = V[v] * W[w]
                    
                    sol, err = solve_linear_block(A_block, b_block, R)
                    total_err += err
                    U_new[i*m + j] = sol
            
            U[:] = U_new
            if total_err == 0:
                return U, V, W
        
        return None, None, None

    # We want to solve for 4x4x8 directly?
    # Or solve for 4x4x4 and combine?
    # Direct 4x4x8:
    # n=4, m=4, p=8.
    # Solve for W:
    # For each i (4), variables W[k*4+i, c] for k=0..7, c=0..R-1.
    # 8*R variables.
    # Equations: m*p = 4*8 = 32.
    # If R > 4, underdetermined.
    
    # Let's try direct search for 4x4x8 with R=100.
    # If we find a solution, rank is 100.
    # Score 100/128 = 0.78.
    
    # Let's try R=64.
    # Score 0.5.
    
    # We'll iterate R from 64 upwards.
    # Start with R=64.
    
    # Note: The solver above uses np.linalg.lstsq which is float.
    # Rounding might fail.
    # But with underdetermined systems, there are many solutions.
    # We might need to try multiple random inits.
    
    # Let's wrap the solver in a retry loop.
    
    def find_decomposition(n, m, p, R, max_retries=20, timeout_per_retry=5):
        for retry in range(max_retries):
            if time.time() > deadline - 10:
                break
            res = solve_tensor(n, m, p, R, timeout=timeout_per_retry)
            if res[0] is not None:
                return res
        return None, None, None

    # Try ranks
    for R in [64, 80, 96, 112, 128]:
        if time.time() > deadline - 20:
            break
        print(f"Trying rank {R} for {n}x{m}x{p}")
        U, V, W = find_decomposition(n, m, p, R)
        if U is not None:
            # Verify
            # We should verify the solution is exact.
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i*m + j
                        v = j*p + k
                        w = k*n + i
                        val = np.sum(U[u] * V[v] * W[w])
                        if val != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                # Convert to lists of ints
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Schoolbook
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
