import time
import numpy as np

DIMENSIONS = [3, 5, 6]
DEADLINE_SECONDS = 120

def construct():
    """
    Uses Adaptive Greedy Coordinate Descent (AGCD) to find a low-rank exact 
    tensor decomposition for 3x5 * 5x6 matrix multiplication.
    
    Strategy:
    1. Initialize with schoolbook (rank 90) as a safe fallback.
    2. Iteratively attempt to reduce rank by finding the best rank-1 update 
       that minimizes the residual error in the tensor space.
    3. Use a greedy heuristic to select the next component: 
       - Sample random directions or use local search on the current factors.
       - Update U, V, W to incorporate the new component.
       - If the residual decreases significantly, accept the step.
    4. Return the best valid decomposition found before the deadline.
    """
    n, m, p = DIMENSIONS
    current_time = time.time()
    deadline = current_time + DEADLINE_SECONDS
    
    # Initialize with schoolbook
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    best_residual = 0.0

    # Target: Try to find a rank < 90 decomposition
    # We will try to build a decomposition from scratch using greedy steps
    # but since exact integer/half-integer constraints are hard for greedy,
    # we will use the greedy approach to find a *structure* and then 
    # verify if it can be exact.
    
    # However, AGCD on the full tensor is expensive. 
    # Alternative: Use the fact that matrix multiplication tensor has a specific structure.
    # We will try to find a decomposition of rank R by solving a system of equations 
    # for random R, but that's NP-hard.
    
    # Let's try a "Greedy Rank Reduction" approach starting from Schoolbook.
    # We try to merge or remove components? No, that's hard.
    
    # Let's try a different angle: Randomized Numerical Linear Algebra / Sketching?
    # No, must be exact.
    
    # Let's stick to the AGCD idea but applied to *finding* a low rank candidate 
    # that might be exact.
    
    # Actually, for small dimensions, we can use an integer programming solver 
    # or a specialized search if we restrict the coefficients to {-1, 0, 1, 0.5}.
    # But 90 variables * 3 matrices is too big for brute force.
    
    # Let's use the "Block Strassen" idea but generalize it.
    # The inspiration mentioned Block Strassen reduced rank to ~84.
    # Let's try to implement a more general block decomposition search.
    
    # We can view the 3x5 and 5x6 matrices as blocks.
    # 3x5 can be seen as [1x5; 1x5; 1x5] or [2x5; 1x5] etc.
    # 5x6 can be seen as [5x2 5x4] etc.
    
    # Let's try to find a decomposition by partitioning the indices.
    # This is essentially searching for a "tensor rank" via block partitions.
    
    # For now, let's implement a simple local search around the schoolbook solution 
    # to see if we can drop rank by 1 or 2.
    
    # Define a function to check if a given U, V, W is valid
    def is_valid(U, V, W):
        # Check if U @ (V^T) matches the multiplication tensor
        # Tensor T[i,j,k] = 1 if i,j,k are the indices of A[i,j] * B[j,k]
        # The decomposition is sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We need to check if this sum equals 1 for valid (i,j,k) and 0 otherwise.
        
        # For efficiency, we can check a few random indices or all indices.
        # n*m*p = 90 indices. 90 checks is cheap.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The term in the product C[i,k] = sum_j A[i,j] B[j,k]
                    # The tensor entry T[i,j,k] is 1 if this j contributes, 0 otherwise?
                    # No, the tensor is defined such that:
                    # C[i,k] = sum_{j} A[i,j] B[j,k]
                    # In tensor notation: C[i,k] = sum_{j,r} T[i,j,k,r] A[i,j] B[j,k]
                    # Wait, the standard decomposition is:
                    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_k V[j,k,r] B[j,k]) * W[k,i,r]?
                    # No, the standard form is:
                    # C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] A[i,j]) * (sum_l V[j,l,r] B[j,l])?
                    # Let's stick to the trace convention: W[k*n+i, r].
                    # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This sum must equal 1 if we are computing the specific term A[i,j]B[j,k] for C[i,k]?
                    # No, the matrix multiplication tensor T has T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
                    # But in the decomposition, we are decomposing the *multiplication map*.
                    # The map is (A,B) -> C.
                    # C[i,k] = sum_j A[i,j] B[j,k].
                    # The tensor T_{i,j,k} is 1 for all i,j,k? No.
                    # The tensor T is defined on indices (i,j,k) where i is row of A, j is col of A/row of B, k is col of B.
                    # T[i,j,k] = 1.
                    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
                    # So for all i in 0..n-1, j in 0..m-1, k in 0..p-1, the sum must be 1.
                    # Wait, is that right?
                    # If T[i,j,k] = 1 for all i,j,k, then yes.
                    # Let's verify with schoolbook.
                    # U[i*m+j, r] = 1 if r corresponds to (i,j,k).
                    # V[j*p+k, r] = 1 if r corresponds to (i,j,k).
                    # W[k*n+i, r] = 1 if r corresponds to (i,j,k).
                    # Sum_r U V W = 1 for the matching r, 0 for others.
                    # So yes, for every (i,j,k), the sum must be 1.
                    
                    val = 0
                    for r in range(len(U[0])):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if not np.isclose(val, 1.0):
                        return False
        return True

    # We will try to reduce the rank by removing columns that are linear combinations of others?
    # Or by finding a lower rank approximation and checking exactness.
    
    # Given the small size, we can try a randomized search for a rank R decomposition.
    # We fix R and try to solve for U, V, W.
    # This is a system of polynomial equations.
    
    # Let's try R = 80, 85, 89.
    # For a given R, we can use a simple iterative method:
    # 1. Initialize U, V, W randomly.
    # 2. Update U: Fix V, W. Solve for U to minimize ||T - sum U V W||.
    #    This is a linear system for each column of U? No, it's coupled.
    #    Actually, if we fix V and W, the term for a specific (i,j,k) is:
    #    sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
    #    This is linear in U[i*m+j, :].
    #    So we can solve for each row of U independently given V, W.
    #    Similarly for V and W.
    #    This is Coordinate Descent!
    
    def coordinate_descent_step(U, V, W, steps=10):
        for _ in range(steps):
            if time.time() > deadline - 5:
                return U, V, W
            
            # Update U
            for i in range(n * m):
                # For each row i of U, we need to satisfy:
                # For all (j,k) such that this row is involved?
                # Row i of U corresponds to A[i//m, i%m]. Let i_A = i//m, j_A = i%m.
                # The equations involving U[i, :] are for all k in 0..p-1:
                # sum_r U[i, r] * V[j_A*p+k, r] * W[k*n+i_A, r] = 1
                # This is a linear system A_x * U[i, :]^T = b_x
                # Where A_x[r, r'] = V[j_A*p+r, r'] * W[r*n+i_A, r']? No.
                # The sum is over r. The unknowns are U[i, 0..R-1].
                # The equation for a specific k is:
                # sum_{r=0}^{R-1} U[i, r] * (V[j_A*p+k, r] * W[k*n+i_A, r]) = 1
                # This gives p equations for R unknowns.
                # If R > p, it's underdetermined. We can use least squares or pick a simple solution.
                # If R <= p, we can solve exactly if the matrix is invertible.
                
                A_mat = np.zeros((p, len(U[0])))
                for k in range(p):
                    for r in range(len(U[0])):
                        A_mat[k, r] = V[j_A*p+k, r] * W[k*n+i_A, r]
                
                b_vec = np.ones(p)
                
                try:
                    # Use least squares
                    U[i, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
            
            # Update V
            for j in range(m * p):
                j_B = j // p
                k_B = j % p
                # Equations for all i in 0..n-1:
                # sum_r V[j, r] * (U[i*m+j_B, r] * W[k_B*n+i, r]) = 1
                A_mat = np.zeros((n, len(V[0])))
                for i in range(n):
                    for r in range(len(V[0])):
                        A_mat[i, r] = U[i*m+j_B, r] * W[k_B*n+i, r]
                b_vec = np.ones(n)
                try:
                    V[j, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
            
            # Update W
            for k in range(p * n):
                k_C = k // n
                i_C = k % n
                # Equations for all j in 0..m-1:
                # sum_r W[k, r] * (U[i_C*m+j, r] * V[j*p+k_C, r]) = 1
                A_mat = np.zeros((m, len(W[0])))
                for j in range(m):
                    for r in range(len(W[0])):
                        A_mat[j, r] = U[i_C*m+j, r] * V[j*p+k_C, r]
                b_vec = np.ones(m)
                try:
                    W[k, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        return U, V, W

    # Try ranks from 80 to 89
    for R in range(80, 90):
        if time.time() > deadline - 10:
            break
            
        # Initialize randomly
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Run coordinate descent
        U, V, W = coordinate_descent_step(U, V, W, steps=50)
        
        # Check validity
        if is_valid(U, V, W):
            best_U, best_V, best_W = U, V, W
            best_rank = R
            break # Found a lower rank, stop searching higher ranks
            
    # Return the best found
    # Convert to lists as required by interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
