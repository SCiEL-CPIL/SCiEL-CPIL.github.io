import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
        n, m, p = DIMENSIONS
        # Target ranks to try, from high (easier) to low (harder)
        # We know 60 works. Let's try 45, 40, 35...
        # The lower bound is max(nm, mp, pn) = 20.
        
        start_time = time.time()
        deadline = 145  # 145 seconds
        
        # Pre-compute indices
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # We will try to find a decomposition for a specific rank R
        # by minimizing the error.
        
        def try_rank(R):
            # Random initialization
            # To help convergence, we can use small integers or random floats
            U = np.random.uniform(-1, 1, (n*m, R))
            V = np.random.uniform(-1, 1, (m*p, R))
            W = np.random.uniform(-1, 1, (p*n, R))
            
            # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a non-linear system.
            # We can use a simple gradient descent or just check if a random integer solution exists (unlikely for low R).
            # Better: Use the fact that we can fix two matrices and solve for the third?
            # No, it's bilinear in each matrix if the others are fixed.
            
            # Let's try a heuristic search:
            # 1. Randomly assign integers to U, V, W.
            # 2. Calculate error.
            # 3. If error is 0, return.
            # This is too slow.
            
            # Alternative: Use a known constructive method for rectangular matrices if possible.
            # For 3x4x5, we can use a block decomposition.
            # Split 4 into 2+2.
            # A = [A1 A2], B = [B1; B2]
            # C = A1 B1 + A2 B2.
            # This doesn't reduce rank below schoolbook for the blocks if we do it naively.
            
            # Let's try to use the "Strassen" trick for the 4x4 part?
            # No, 4 is the middle dimension.
            
            # Let's stick to a randomized local search on the continuous domain, then round.
            # We will run a few iterations of a simple optimizer.
            
            # To make it faster, we can use a "coordinate descent" style update.
            # But 150s is short for heavy optimization.
            
            # Let's try a specific known lower rank if we can recall one.
            # For 2x2x2, rank 7.
            # For 3x3x3, rank 19? (Actually 19 is for 4x4? No. 3x3 is 19? No, 3x3 is 19? 
            # Coppersmith-Winograd is asymptotic.
            # For small matrices, exact ranks are known.
            # 3x3x3 rank is 19? No, 3x3x3 rank is 19 is false. It is 19 for 4x4?
            # Let's assume we can find a rank < 60 via search.
            
            pass

        # Since we don't have a guaranteed constructive algorithm for 3x4x5 < 60 in memory,
        # and random search for exact integer solutions in high dimensions is hard,
        # we will implement a **relaxed** search that might find a solution.
        
        # However, there is a simpler structural improvement:
        # The schoolbook rank is 60.
        # We can use a "block" algorithm that treats the 4x4 multiplication as two 2x2 multiplications?
        # No, 4 is the inner dimension.
        
        # Let's try to use a **randomized integer search** for small R.
        # If we assume the entries are -1, 0, 1, the space is huge.
        
        # Given the constraints, the safest "improvement" that is likely to work is 
        # using a **pre-computed** decomposition if one is known, or a very fast heuristic.
        
        # Actually, for 3x4x5, we can use the following:
        # Rank 45 is a common target for 3x3x3 (if 3x3x3 is 19, then 3x4x5 is roughly 19*4/3? No).
        
        # Let's try a different approach: **Linear Algebra Reduction**.
        # We want $T = \sum U_r \otimes V_r \otimes W_r$.
        # This is equivalent to finding a low-rank factorization of the matrix unfolding of T.
        # The matrix $T_{(ij), k}$ has size $(nm) \times p = 12 \times 5$.
        # The rank of this matrix is at most 5.
        # But the tensor rank is higher.
        
        # I will implement a **randomized local search** that tries to satisfy the equations.
        # We will use `numpy` to compute the error.
        # We will try ranks 45, 40, 35.
        
        for R in [45, 40, 35, 30]:
            if time.time() > deadline - 10:
                break
                
            # Try a few random restarts for this R
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                    
                # Initialize with small integers to encourage integer solutions
                # Use -1, 0, 1
                U = np.random.randint(-1, 2, (n*m, R))
                V = np.random.randint(-1, 2, (m*p, R))
                W = np.random.randint(-1, 2, (p*n, R))
                
                # Check if this random guess works (very unlikely)
                # Instead, we can try to "fix" the matrices.
                
                # A better heuristic:
                # Set U and V randomly, then solve for W?
                # The equation is linear in W for fixed U, V.
                # $\sum_r W_{k \cdot n + i, r} ( \sum_j U_{i \cdot m + j, r} V_{j \cdot p + k, r} A_{ij} B_{jk} \dots )$
                # No, the equation is:
                # For all i, j, k:
                # $\sum_r U_{i \cdot m + j, r} V_{j \cdot p + k, r} W_{k \cdot n + i, r} = 1$
                
                # This is a system of linear equations in W if U and V are fixed.
                # Variables: $W_{k \cdot n + i, r}$ for $k \in [0,p), i \in [0,n), r \in [0,R)$.
                # Total variables: $p \cdot n \cdot R$.
                # Equations: $n \cdot m \cdot p$.
                # For $n=3, m=4, p=5, R=45$:
                # Vars: $5 \cdot 3 \cdot 45 = 675$.
                # Eqs: $3 \cdot 4 \cdot 5 = 60$.
                # Underdetermined! We can solve for W.
                
                # Let's do this:
                # 1. Randomly initialize U and V with integers (e.g., -1, 0, 1).
                # 2. Construct the linear system $A_{eq} W_{vec} = b$.
                # 3. Solve for $W_{vec}$.
                # 4. Check if $W_{vec}$ is close to integers/half-integers.
                # 5. If yes, round and verify.
                
                # Build the linear system
                # Eq index: (i, j, k) -> idx = i * m * p + j * p + k
                # Var index: (k, i, r) -> w_idx = (k * n + i) * R + r
                
                num_eqs = n * m * p
                num_vars = p * n * R
                
                A_mat = np.zeros((num_eqs, num_vars))
                b_vec = np.ones(num_eqs)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                # Coefficient for W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
                                coeff = U[i * m + j, r] * V[j * p + k, r]
                                var_idx = (k * n + i) * R + r
                                A_mat[eq_idx, var_idx] = coeff
                
                # Solve A_mat W = b_vec
                # Use least squares (pinv) since it's underdetermined
                # We want the minimum norm solution
                W_vec, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Reshape W_vec
                W_mat = W_vec.reshape((p * n, R))
                
                # Check if W_mat is close to integers or half-integers
                # We can round to nearest 0.5
                W_rounded = np.round(W_mat * 2) / 2.0
                
                # Verify if the rounded solution is exact
                # Reconstruct tensor
                # T_recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W_rounded[k*n+i, r]
                
                # Efficient check
                # We can compute the error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U[i * m + j, r] * V[j * p + k, r] * W_rounded[k * n + i, r]
                            if abs(val - 1.0) > 1e-9:
                                error += 1
                                break
                        if error > 0:
                            break
                    if error > 0:
                        break
                
                if error == 0:
                    # Found a valid decomposition!
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_rounded.tolist()
                    }

        # Fallback to schoolbook
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
