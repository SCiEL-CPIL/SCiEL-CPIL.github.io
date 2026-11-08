"""Find a rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # We need to find a low-rank exact decomposition for the matrix multiplication tensor.
    # The seed program provides the schoolbook decomposition of rank n*m*p = 24.
    # We will attempt to find a decomposition of rank 23 or lower using random search.
    # If no lower rank decomposition is found within the time limit, we return the seed.
    
    # Dimensions: n=2, m=3, p=4
    n, m, p = 2, 3, 4
    seed_rank = n * m * p  # 24
    
    # Deadline guard
    deadline = time.time() + 110.0
    
    # Seed the random number generator for reproducibility within the run
    random.seed(42)
    
    # The matrix multiplication tensor T has components T[i, j, k] which is 1 if A[i,j] * B[j,k] contributes to C[i,k].
    # The decomposition is sum_{c=1}^R U[i, c] * V[j, c] * W[k, c] = T[i, j, k].
    # Indices:
    # U has shape (n*m, R) -> rows indexed by (i, j)
    # V has shape (m*p, R) -> rows indexed by (j, k)
    # W has shape (p*n, R) -> rows indexed by (k, i) -- Note: Trace convention W[k*n+i, r]
    
    # Let's check the tensor entries.
    # T[i, j, k] = 1 if the product A[i,j] * B[j,k] adds to C[i,k].
    # Since matrix multiplication is C[i,k] = sum_j A[i,j] B[j,k], the term A[i,j]B[j,k] is present for all valid i,j,k.
    # So T[i, j, k] = 1 for all i in {0..1}, j in {0..2}, k in {0..3}.
    # Wait, this is the structure tensor of matrix multiplication.
    # The tensor is M[i, j, k] = 1.
    # The rank of this tensor is the matrix multiplication rank.
    
    # Let's formulate the condition for rank R.
    # We need U (n*m x R), V (m*p x R), W (p*n x R) such that:
    # for all i, j, k: sum_{c=0}^{R-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of equations in the entries of U, V, W.
    # Number of equations: n*m*p = 24.
    # Number of variables: 3 * (R * (n*m or m*p or p*n)).
    
    # We can try to solve this by fixing some entries and solving for others, or by using a randomized approach.
    # Given the small size, we can try to search for integer/half-integer solutions.
    
    # Let's first verify the schoolbook solution to be sure.
    # U[i*m+j, c] = 1 if c == i*m*p + j*p + k? No, seed uses column index.
    
    # Strategy:
    # 1. Try to find a rank 23 solution with entries in {0, 1/2, 1, 3/2, 2} or similar small rationals.
    # 2. If time runs out, return rank 24.
    
    def check_solution(U, V, W, n, m, p):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(R):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank(R, max_iters=10000, time_limit=None):
        # Randomly initialize U, V, W with small integers or halves
        # and try to adjust?
        # This is hard. Instead, let's use the property that if we can set one column to be "free" 
        # and solve for it given the others, we might reduce the rank.
        
        # Actually, a simpler approach for small tensors:
        # We can use the "border rank" or "rank" algorithms from tensor algebra, 
        # but implementing them from scratch is risky.
        
        # Let's try a specific known result or a simple perturbation.
        # For 2x2x2, rank is 7. 
        # For 2x3x3, is it 17?
        # For 2x3x4, is it 23?
        
        # Let's try to construct a rank 23 solution by "merging" two columns of the rank 24 solution.
        # If we can express one column as a linear combination of the others, we reduce rank.
        # But the equations are nonlinear.
        
        # Alternative: Random search over the manifold of solutions.
        # This is computationally expensive.
        
        # Given the constraints and the risk of finding an invalid solution (score 0), 
        # and the lack of a guaranteed method to find a lower rank, 
        # the safest bet is to return the schoolbook decomposition.
        
        # However, the prompt asks for a "stronger algorithm".
        # Is there a known rank for 2x3x4?
        # The rank of the matrix multiplication tensor M_{n,m,p} is known to be n*m*p.
        # Strassen showed M_{2,2,2} has rank 7.
        # For M_{2,3,4}, the rank is not widely published as a small integer < 24 in standard introductory texts, 
        # but it is likely less than 24.
        
        # Let's try a randomized local search around the schoolbook solution.
        # We start with the rank 24 solution.
        # We try to set one entry in U to 0, and see if we can adjust the rest to satisfy the equations.
        
        # This is effectively solving a system of polynomial equations.
        # We can use sympy? No, standard library, numpy, scipy only.
        # Scipy has fsolve.
        
        # Let's try to use scipy.optimize.least_squares to find a solution of rank R.
        # We will flatten U, V, W into a single vector x.
        # We want to minimize sum (sum_c U_ic V_jc W_kc - 1)^2.
        
        # This is a non-convex optimization problem.
        # With rank 23, we have fewer variables than degrees of freedom?
        # Variables: 23 * (6 + 12 + 8) = 23 * 26 = 598.
        # Equations: 24.
        # It is highly underdetermined. We should be able to find a solution.
        
        # Let's implement a solver using scipy.
        
        def solve_rank(R, time_limit_remaining):
            n, m, p = 2, 3, 4
            dim_u = n * m
            dim_v = m * p
            dim_w = p * n
            total_vars = R * (dim_u + dim_v + dim_w)
            
            # Objective function
            def objective(x):
                # x is a vector of length total_vars
                U = x[0:R*dim_u].reshape(R, dim_u).T  # (dim_u, R)
                V = x[R*dim_u:R*dim_u+R*dim_v].reshape(R, dim_v).T  # (dim_v, R)
                W = x[R*dim_u+R*dim_v:R*dim_u+R*dim_v+R*dim_w].reshape(R, dim_w).T  # (dim_w, R)
                
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            prod = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                            errors.append(prod - 1.0)
                return np.array(errors)
            
            # Initial guess: Scale down the schoolbook solution to fit R columns?
            # This is tricky. Let's just use a random initialization with small values.
            # Or, we can try to initialize with the first R columns of the schoolbook solution,
            # and set the last column to be a linear combination?
            
            # Let's try a random initialization first.
            # Since the problem is underdetermined, random init might work.
            
            # But we need integer/half-integer entries.
            # The solution from least_squares will be floating point.
            # We can try to round it to the nearest half-integer and check if it's still valid.
            
            # However, there is a risk that the solution is not close to half-integers.
            
            # Let's try to run this solver.
            
            # Set up the solver
            x0 = np.random.rand(total_vars) * 0.1  # Small random init
            
            # Use scipy.optimize.least_squares
            # method='lm' (Levenberg-Marquardt) is good for small problems.
            
            # Time check
            if time_limit_remaining < 1.0:
                return None
            
            try:
                result = scipy.optimize.least_squares(objective, x0, method='lm', xtol=1e-10, ftol=1e-10, gtol=1e-10)
            except Exception:
                return None
            
            if result.cost < 1e-8:
                # We found a solution. Now we need to check if it can be converted to half-integers.
                # This is a separate problem.
                # If the solution is generic, it might not be half-integers.
                # However, if the rank is minimal, the solution space might have discrete components.
                
                # Let's extract U, V, W
                U = result.x[0:R*dim_u].reshape(R, dim_u).T
                V = result.x[R*dim_u:R*dim_u+R*dim_v].reshape(R, dim_v).T
                W = result.x[R*dim_u+R*dim_v:R*dim_u+R*dim_v+R*dim_w].reshape(R, dim_w).T
                
                # Round to nearest half-integer
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                # Check if the rounded solution is valid
                if check_solution(U_half, V_half, W_half, n, m, p):
                    return U_half, V_half, W_half
            
            return None
        
        import scipy.optimize
        
        # Try rank 23
        if time.time() < deadline - 5.0:
            sol = solve_rank(23, deadline - time.time())
            if sol is not None:
                U, V, W = sol
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If we reach here, we couldn't find a better solution quickly.
        # Return the schoolbook solution.
        U = [[0] * seed_rank for _ in range(n * m)]
        V = [[0] * seed_rank for _ in range(m * p)]
        W = [[0] * seed_rank for _ in range(p * n)]
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
