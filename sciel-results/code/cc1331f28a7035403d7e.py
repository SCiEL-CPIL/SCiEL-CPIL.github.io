import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target ranks to try, starting from schoolbook - 1
    # We limit the search to a reasonable number of ranks to stay within time
    max_search_rank = rank_schoolbook - 1
    min_search_rank = 20 # Arbitrary lower bound
    
    # Precompute the multiplication tensor M
    # M[i, j, k] = 1 if A[i,j] and B[j,k] contribute to C[k,i]
    # Wait, the tensor T_{i,j,k} for matrix mult (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The standard tensor decomposition form is:
    # (A*B)_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) * (sum_j V_{j,k,c} B_{j,k}) * W_{k,i,c} ??
    # No, the standard form is:
    # C_{i,k} = sum_c W_{k,i,c} * (sum_j U_{i,j,c} A_{i,j}) * (sum_j V_{j,k,c} B_{j,k})
    # This is not quite right. The bilinear form is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Decomposition:
    # C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_j V_{j,k,c} B_{j,k}) W_{k,i,c} is wrong.
    # The correct tensor decomposition for matrix multiplication $C = AB$ is:
    # $C_{i,k} = \sum_{c=1}^R (\sum_{j} U_{i,j,c} A_{i,j}) (\sum_{j} V_{j,k,c} B_{j,k})$ 
    # This requires $W_{k,i,c}$ to be absorbed or the form is:
    # $C_{i,k} = \sum_c W_{c} ( \sum_j U_{i,j,c} A_{i,j} ) ( \sum_j V_{j,k,c} B_{j,k} )$
    # Here $W_{c}$ is a scalar. But the interface requires $W$ to be a matrix of shape $(p*n, rank)$.
    # The contract says: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the term is $W_{k*n+i, c}$.
    # So the equation is:
    # $\delta_{i,k} \text{ is not involved. }$
    # $C_{i,k} = \sum_c W_{k*n+i, c} ( \sum_j U_{i*m+j, c} A_{i,j} ) ( \sum_j V_{j*p+k, c} B_{j,k} )$
    # Let's verify the indices.
    # $U$ shape: $(n*m, R)$. Index $i*m+j$.
    # $V$ shape: $(m*p, R)$. Index $j*p+k$.
    # $W$ shape: $(p*n, R)$. Index $k*n+i$.
    # This matches the standard tensor definition for matrix multiplication.
    
    M = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i, j, k] = 1.0
                
    # Reshape M to match the linear system
    # The equation is: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = M[i,j,k]
    # This is a system of equations for the variables U, V, W.
    
    def try_rank(R):
        start_time = time.time()
        timeout = 10 # seconds per rank attempt? Total budget 120s.
        # We have ~25 ranks to try. 120/25 = 4.8s per rank.
        # Let's use 2s per rank for safety, or adjust dynamically.
        
        for restart in range(50): # Max 50 restarts per rank
            if time.time() - start_time > 2.0:
                break
            
            # Initialize W randomly with -1, 0, 1
            W = np.random.choice([-1, 0, 1], size=(p*n, R))
            # Ensure W is not all zeros
            if np.all(W == 0):
                W[0, 0] = 1
                
            # We can solve for U and V given W?
            # The equation is linear in U if V and W are fixed? No, it's U*V.
            # It is linear in U if we fix V and W.
            # It is linear in V if we fix U and W.
            
            # Let's fix V and W and solve for U?
            # Or fix U and W and solve for V?
            
            # Better: Fix W. Then the equation is:
            # sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] = M[i,j,k]
            # This is still bilinear in U and V.
            
            # We can use a random initialization for U and V and alternate.
            U = np.random.choice([-1, 0, 1], size=(n*m, R))
            V = np.random.choice([-1, 0, 1], size=(m*p, R))
            
            for step in range(20):
                # Solve for U
                # For each (i, j, k), sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] = M[i,j,k]
                # This is not a simple linear system for U because V is also a variable.
                # However, if we fix V, we can solve for U?
                # The term is U[i*m+j, c] * (sum over k? No, k is an index in the output).
                
                # Let's vectorize.
                # Let $A_c$ be the matrix formed by $U_{:,c}$ and $V_{:,c}$.
                # This is complex.
                
                # Alternative: Use the fact that we can solve for one matrix if the other two are fixed.
                # If we fix V and W, can we solve for U?
                # Equation: sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] = M[i,j,k]
                # Group by c: sum_c ( W[k*n+i, c] * V[j*p+k, c] ) * U[i*m+j, c] = M[i,j,k]
                # This is linear in U_{i*m+j, c} for fixed i,j?
                # No, the sum is over c. And the coefficient of U_{i*m+j, c} depends on k.
                # But for a fixed i,j, the equation must hold for ALL k.
                # So for fixed i,j:
                # For all k: sum_c U[i*m+j, c] * ( W[k*n+i, c] * V[j*p+k, c] ) = M[i,j,k]
                # Let $Z_{j,k,c} = W[k*n+i, c] * V[j*p+k, c]$. This depends on i!
                # So for fixed i, we have a system for U[i*m+0..m-1, :].
                # For each j, we have a vector $u_j = U[i*m+j, :]$.
                # The equation is: $u_j \cdot Z_{j,k,:} = M[i,j,k]$ for all k.
                # This is a linear system for $u_j$ (size R) with p equations.
                # If p < R, it's underdetermined. If p > R, overdetermined.
                # Here p=5, R >= 20. So underdetermined.
                # We can find the minimum norm solution.
                
                # So, algorithm:
                # 1. Fix V, W.
                # 2. For each i, j:
                #    Construct matrix $Z$ of size (p, R).
                #    $Z[k, c] = W[k*n+i, c] * V[j*p+k, c]$
                #    Solve $Z u_j = M[i,j,:]$ for $u_j$ (size R).
                #    Set $U[i*m+j, :] = u_j$.
                # 3. Repeat for V, fixing U, W.
                #    For each j, k:
                #    Construct matrix $Y$ of size (m, R).
                #    $Y[i, c] = W[k*n+i, c] * U[i*m+j, c]$  <-- Wait, i is the index for U.
                #    Equation: sum_c U[i*m+j, c] W[k*n+i, c] V[j*p+k, c] = M[i,j,k]
                #    For fixed j, k:
                #    Let $v_k = V[j*p+k, :]$.
                #    Equation: sum_i ( U[i*m+j, c] W[k*n+i, c] ) v_k[c] ... No.
                #    Sum over c: $v_k[c] * ( U[i*m+j, c] W[k*n+i, c] )$.
                #    This is not linear in $v_k$ if U and W are fixed?
                #    Yes it is!
                #    For fixed j, k:
                #    Let $Y_{i, c} = U[i*m+j, c] * W[k*n+i, c]$.
                #    Then sum_i? No, the sum is over c.
                #    Equation: sum_c $Y_{i, c} v_k[c]$? No, $v_k[c]$ is multiplied by $U[i*m+j, c] W[k*n+i, c]$.
                #    But the equation is for a specific i? No, the equation is for all i, j, k.
                #    So for fixed j, k, the equation must hold for all i.
                #    So: For all i: sum_c $Y_{i, c} v_k[c] = M[i,j,k]$.
                #    This is a linear system for $v_k$ (size R) with n equations.
                #    Here n=3, R >= 20. Underdetermined.
                #    Solve for minimum norm $v_k$.
                
                # 4. Check error.
                
                pass # Placeholder for implementation
            
            # Implementation of the alternating solver
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.random.choice([-1, 0, 1], size=(p*n, R))
            if np.all(W == 0): W[0,0] = 1
            
            for step in range(10):
                # Update U
                for i in range(n):
                    for j in range(m):
                        Z = np.zeros((p, R))
                        for k in range(p):
                            Z[k, :] = W[k*n+i, :] * V[j*p+k, :]
                        # Solve Z u = M[i,j,:]
                        # Z is (p, R), M[i,j,:] is (p,)
                        # Use lstsq
                        u, residuals, rank, sv = np.linalg.lstsq(Z, M[i,j,:], rcond=None)
                        U[i*m+j, :] = u
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        Y = np.zeros((n, R))
                        for i in range(n):
                            Y[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        # Solve Y v = M[:,j,k]
                        # Y is (n, R), M[:,j,k] is (n,)
                        v, residuals, rank, sv = np.linalg.lstsq(Y, M[:,j,k], rcond=None)
                        V[j*p+k, :] = v
                        
                # Check Error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(R):
                                val += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                            err += (val - M[i,j,k])**2
                if err < 1e-6:
                    # Found a solution!
                    # Try to round to integers/half-integers
                    # If it's already close to integers, return it.
                    return U, V, W
                    
        return None

    # Search for better rank
    for R in range(rank_schoolbook - 1, min_search_rank - 1, -1):
        start = time.time()
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists as required
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
        if time.time() - start > 110:
            break
            
    # Fallback to schoolbook
    rank = rank_schoolbook
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
# EVOLVE-BLOCK-END
