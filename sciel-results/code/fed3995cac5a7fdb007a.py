import time
import numpy as np

DIMENSIONS = [2, 4, 5]
TIME_LIMIT = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Schoolbook rank: n*m*p = 40
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p

    # Verify function
    def verify(res, tol=1e-9):
        try:
            U = np.array(res["U"], dtype=float)
            V = np.array(res["V"], dtype=float)
            W = np.array(res["W"], dtype=float)
            
            # Check dimensions
            if U.shape != (n*m, best_rank): return False
            if V.shape != (m*p, best_rank): return False
            if W.shape != (p*n, best_rank): return False
            
            # The tensor is T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # We want T to match the matrix multiplication tensor.
            # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
            # T[a,b,c] = 0 otherwise
            
            # Efficient check using einsum or loops
            # Since dimensions are small, we can just loop over all possible a,b,c
            # But it's faster to check the non-zeros and a sample of zeros, 
            # or just all 1600 entries. 1600 * 36 multiplications is trivial.
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                        if abs(val - 1.0) > tol:
                            return False
            
            # Check zeros
            # We can iterate over all a, b, c
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        # Check if this is a "valid" index
                        i = a // m
                        j = a % m
                        j_b = b // p
                        k_b = b % p
                        k_c = c // n
                        i_c = c % n
                        
                        is_valid = (j == j_b) and (k_b == k_c) and (i == i_c)
                        
                        val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                        if is_valid:
                            if abs(val - 1.0) > tol:
                                return False
                        else:
                            if abs(val) > tol:
                                return False
            return True
        except Exception:
            return False

    # Groebner Basis / Algebraic Elimination Strategy
    # We want to find U, V, W of rank R < 36.
    # The equations are:
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # This is a system of quadratic equations in the variables U, V, W.
    # We can fix one matrix (say W) to a random generic matrix to break symmetry,
    # then the equations become linear in U and V? No, they are bilinear in U and V.
    # sum_r (U[a,r] V[b,r]) W[c,r] = T[a,b,c]
    # Let M[a,b] = sum_r U[a,r] V[b,r]. Then sum_r M[a,b] W[c,r] = T[a,b,c].
    # This is a linear system for M given W.
    # M = T * W^{-1} (if W is invertible and dimensions match, which they don't exactly).
    # Actually, T is (n*m, m*p, p*n). W is (p*n, R).
    # Let T_{ab} be the (n*m x m*p) matrix for fixed c? No.
    # The equation is: For each (a,b,c), sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    # Let X = U V^T? No, U is (n*m, R), V is (m*p, R).
    # Let M = U V^T. M is (n*m, m*p).
    # Then sum_r M[a,b] W[c,r] ... wait.
    # sum_r U[a,r] V[b,r] W[c,r] is not M[a,b] dot W[c,:].
    # It is sum_r (U[a,r] V[b,r]) W[c,r].
    # Let Y[a,b,r] = U[a,r] V[b,r].
    # Then sum_r Y[a,b,r] W[c,r] = T[a,b,c].
    # This is linear in Y given W.
    # But Y must be rank-1 in the (a,b) sense? No, Y[a,b,r] is just a product.
    # Actually, for a fixed r, U[:,r] V[:,r]^T is a rank-1 matrix.
    # So M_r = U[:,r] V[:,r]^T.
    # The tensor is sum_r M_r \otimes W[:,r]^T? No.
    # T = sum_r (U[:,r] \otimes V[:,r] \otimes W[:,r]).
    
    # Strategy:
    # 1. Guess R.
    # 2. Randomly initialize W (p*n, R) with Gaussian entries.
    # 3. Solve for U and V.
    #    The condition is: For all a,b,c, sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    #    This is a system of equations in U and V.
    #    Let's fix U and solve for V?
    #    sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    #    For fixed a,c, this is linear in V[b,:].
    #    Let A_{a,c,b,r} = U[a,r] W[c,r].
    #    Then sum_r A_{a,c,b,r} V[b,r] = T[a,b,c].
    #    This is a linear system for V[b,:] given U.
    #    But V[b,:] is shared across a and c.
    #    So we have a large linear system.
    #    Variables: V[b,r] for b in 0..m*p-1, r in 0..R-1. Total R*m*p variables.
    #    Equations: For each (a,b,c), one equation. Total n*m*m*p*p*n equations.
    #    This is overdetermined if R is small.
    #    We can use Least Squares to find V given U.
    #    Then check if the residual is small.
    #    If yes, we have a solution.
    #    If no, randomize U and try again.
    
    # This is a "Randomized Linearization" approach.
    # It is structurally different from block decomposition.
    
    def solve_rank_R(R, num_trials=100):
        # Target Tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Reshape T to (n*m, m*p, p*n) -> (n*m, m*p*p*n)?
        # Equation: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # Fix W randomly.
        # Then for each (a,b), we have:
        # sum_r (U[a,r] W[c,r]) V[b,r] = T[a,b,c] for all c.
        # Let M_{a,c} = sum_r U[a,r] W[c,r]. (Matrix of size n*m x p*n)
        # Then M_{a,c} is a scalar? No.
        # Wait. U[a,r] is a scalar. W[c,r] is a scalar.
        # M_{a,c} = sum_r U[a,r] W[c,r] is a scalar.
        # The equation is: M_{a,c} V[b,r]?? No.
        # The sum is over r.
        # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
        # This is NOT linear in V if U and W are fixed?
        # Let's look at the structure.
        # For a fixed r, the term is U[a,r] V[b,r] W[c,r].
        # This is a rank-1 tensor in (a,b,c).
        # The sum of R such tensors equals T.
        
        # If we fix U and W, the equation is linear in V?
        # No, V[b,r] is multiplied by U[a,r] and W[c,r].
        # For a specific b, let v_b be the vector V[b, :] of length R.
        # The equation for a specific (a,b,c) is:
        # dot( (U[a,:] * W[c,:]), v_b ) = T[a,b,c]
        # where * is element-wise product.
        # Let Z_{a,c} = U[a,:] * W[c,:] (element-wise). Z_{a,c} is a vector of length R.
        # Then dot(Z_{a,c}, v_b) = T[a,b,c].
        # This is a linear system for v_b!
        # For each b, we have a system of equations:
        # For all a in 0..n*m-1 and c in 0..p*n-1:
        # dot(Z_{a,c}, v_b) = T[a,b,c]
        # Number of equations: n*m * p*n = 2*4 * 5*2 = 8 * 10 = 80.
        # Number of unknowns in v_b: R.
        # If R < 80, we can solve this least squares problem for each b.
        # Then we check if the resulting V satisfies the original tensor equation.
        # Since we solved for V given U and W, it will satisfy the equations for the specific (a,b,c) used.
        # But wait, the equations are exactly the tensor equations.
        # So if we find v_b that satisfies dot(Z_{a,c}, v_b) = T[a,b,c] for ALL a,c,
        # then the tensor equation is satisfied.
        # Is it possible?
        # The system is:
        # [Z_{0,0}, Z_{0,1}, ..., Z_{0,9}]
        # [Z_{1,0}, Z_{1,1}, ..., Z_{1,9}]
        # ...
        # [Z_{7,0}, Z_{7,1}, ..., Z_{7,9}]
        # * v_b = T[:, b, :] flattened.
        
        # So the algorithm is:
        # 1. Randomize U (n*m, R) and W (p*n, R).
        # 2. For each b in 0..m*p-1:
        #    Construct matrix A_b of size (n*m*p*n) x R.
        #    A_b[ (a*n_p + c), r ] = U[a, r] * W[c, r]
        #    Construct vector t_b of size (n*m*p*n).
        #    t_b[ (a*n_p + c) ] = T[a, b, c]
        #    Solve A_b v_b = t_b using least squares.
        # 3. Assemble V from v_b.
        # 4. Verify the full tensor.
        
        # Why would this work?
        # If a solution exists, then for the correct U and W, the system A_b v_b = t_b is consistent.
        # If we pick random U and W, the system is likely inconsistent (no exact solution).
        # However, if R is large enough, the least squares solution might be "close" to a solution?
        # No, we need an EXACT solution.
        # This approach only works if we can find U and W such that the systems are consistent.
        # This is a nonlinear constraint on U and W.
        
        # Alternative:
        # Use the fact that if we fix V, we can solve for U and W?
        # No, it's symmetric.
        
        # Let's try a different structural change:
        # "Groebner basis" suggests solving the polynomial system directly.
        # But that's hard.
        
        # Let's stick to the Randomized Linearization but add a twist:
        # Instead of random U, W, we can use the structure of the schoolbook decomposition
        # and try to "merge" terms.
        
        # Actually, the "Randomized Linearization" is a valid different strategy.
        # It might fail to find a solution if R is too small, but it's worth a try.
        # We can also try to optimize U and W to minimize the residual of the least squares solutions.
        
        # Let's implement the least squares approach.
        
        for trial in range(num_trials):
            if time.time() - start_time > TIME_LIMIT - 5:
                return best_result
            
            U = np.random.randn(n*m, R)
            W = np.random.randn(p*n, R)
            
            # Solve for V
            V = np.zeros((m*p, R))
            success = True
            for b in range(m*p):
                # Construct A_b
                # A_b has rows indexed by (a, c)
                # A_b[idx, r] = U[a, r] * W[c, r]
                A_b = np.zeros((n*m * p*n, R))
                t_b = np.zeros(n*m * p*n)
                
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A_b[idx, :] = U[a, :] * W[c, :]
                        t_b[idx] = T[a, b, c]
                        idx += 1
                
                # Solve A_b v_b = t_b
                # Use least squares
                try:
                    v_b, residuals, rank, sv = np.linalg.lstsq(A_b, t_b, rcond=None)
                    V[b, :] = v_b
                except Exception:
                    success = False
                    break
            
            if not success:
                continue
            
            # Construct result
            res = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Verify
            if verify(res):
                best_result = res
                best_rank = R
                return best_result
                
    # Try ranks from 35 down to 20
    for R in range(35, 19, -1):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        solve_rank_R(R, num_trials=20)
        
    return best_result
