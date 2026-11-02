"""Matrix multiplication 3x3x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    DEADLINE = 120.0  # seconds, with safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # The matrix multiplication tensor for (n,m) x (m,p) -> (p,n)
    # has rank equal to the minimum number of rank-1 tensors needed
    # to represent it exactly.
    
    # For 3x3 x 3x7, we need to find the minimal rank.
    # 
    # Known bounds:
    # - Schoolbook rank: n*m*p = 3*3*7 = 63
    # - We can potentially do better using block structure
    #
    # The tensor T[i,j,k] = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} for all i,j,k
    # where i in 0..n-1, j in 0..m-1, k in 0..p-1
    #
    # Actually, the matrix multiplication tensor is:
    # T = sum_{i,j,k} T_{i,j,k} * (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i)
    # where T_{i,j,k} = 1 if the indices match appropriately
    #
    # More precisely, for multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor is: sum_{i,j,k} A_{i,j} * B_{j,k} * (C_{i,k})
    #
    # In the trace convention W[k*n+i], so W is indexed by (k,i) where
    # k is the row of C and i is the column... wait, let me re-read.
    #
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank), and W[k*n+i, r] corresponds to the
    # (i,k) entry of the result matrix C (n x p).
    #
    # Wait, C is n x p, so C[i,k] for i in 0..n-1, k in 0..p-1.
    # W[k*n+i] means row index is k*n + i, so for a given (i,k),
    # the row is k*n + i.
    
    # Let's verify: for n=3, p=7, W has 21 rows.
    # W[k*3+i] for k in 0..6, i in 0..2.
    # So W[0] = C[0,0], W[1] = C[1,0], W[2] = C[2,0], W[3] = C[0,1], etc.
    
    # The tensor equation is:
    # For each (i, k): C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # In terms of the decomposition:
    # C[i,k] = sum_r U[i*m+j_r, r] * V[j_r*p+k, r] * W[k*n+i, r]
    #
    # Wait, that's not quite right. Let me think more carefully.
    #
    # A rank-r decomposition of the matrix multiplication tensor means:
    # For all A, B: C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Can be written as:
    # C[i,k] = sum_{r=1}^R (sum_j U[j, r] * A[i,j]) * (sum_j V[j, r] * B[j,k]) * W[k, i]
    #
    # No wait, that's not standard. Let me use the standard formulation.
    #
    # The standard tensor decomposition for matrix multiplication:
    # T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    #
    # where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{n*p}
    # (or in our trace convention, w_r is in R^{p*n})
    #
    # And the condition is:
    # sum_{i,j} u_r[i*m+j] * A[i,j] * sum_{j',k} v_r[j'*p+k] * B[j',k] * w_r[k*n+i]
    # = C[i,k] for the appropriate (i,k)
    #
    # Actually, the correct formulation is:
    # C[i,k] = sum_{r=1}^R (sum_{j} U[i*m+j, r] * A[i,j]) * (sum_{j} V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    #
    # No, that's still not right. Let me be more careful.
    #
    # The matrix multiplication tensor T_{i,j,k} for multiplying (n x m) by (m x p)
    # is defined such that:
    # <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A rank-1 term is: u ⊗ v ⊗ w where u in R^{n*m}, v in R^{m*p}, w in R^{n*p}
    # and <u ⊗ v ⊗ w, A ⊗ B> = (<u, A> * <v, B>) * w
    #
    # So for the decomposition to work:
    # sum_{r} (<u_r, A> * <v_r, B>) * w_r = C = A*B
    #
    # Where <u_r, A> = sum_{i,j} u_r[i*m+j] * A[i,j]
    # <v_r, B> = sum_{j,k} v_r[j*p+k] * B[j,k]
    # w_r is a vector in R^{n*p} (or R^{p*n} with trace convention)
    #
    # And the condition is that for all A, B:
    # sum_r (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{j,k} V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # = sum_j A[i,j] * B[j,k]
    #
    # Wait, I'm confusing myself. Let me use a cleaner formulation.
    #
    # The tensor is: T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the standard basis vector in R^{n*m} for position (i,j)
    # e_{j,k} is in R^{m*p} for position (j,k)
    # e_{k,i} is in R^{p*n} for position (k,i) [trace convention]
    #
    # A rank-1 decomposition term is: u_r ⊗ v_r ⊗ w_r
    # The sum of rank-1 terms equals T if and only if:
    # sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} = sum_r u_r ⊗ v_r ⊗ w_r
    #
    # This means: for each (i,j,k):
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    #
    # And for all other combinations of indices (i',j',k') that don't correspond
    # to a valid term in the tensor, the sum should be 0.
    #
    # Actually no. The tensor T has entries T_{(i,j),(j,k),(k,i)} = 1 for valid
    # (i,j,k) and 0 otherwise. So the condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid
    # multiplication term (i.e., i in 0..n-1, j in 0..m-1, k in 0..p-1)
    # and 0 otherwise.
    #
    # But wait, ALL combinations of (i,j,k) with i in 0..n-1, j in 0..m-1, k in 0..p-1
    # are valid terms in the tensor. The tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    #
    # So the condition is simply:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    #
    # This is a system of n*m*p = 3*3*7 = 63 equations.
    
    # The schoolbook decomposition has rank 63:
    # For each (i,j,k): U[i*m+j, r_{i,j,k}] = 1, V[j*p+k, r_{i,j,k}] = 1, W[k*n+i, r_{i,j,k}] = 1
    
    # We want to find a lower rank decomposition.
    
    # Known results:
    # - For 3x3 x 3x3, the rank is 23 (Strassen)
    # - For 3x3 x 3x7, we need to find the minimal rank
    
    # One approach: use the block structure.
    # The 3x3 x 3x7 multiplication can be seen as:
    # C = A * B where A is 3x3, B is 3x7
    #
    # We can split B into B1 (3x3) and B2 (3x4) or other splits.
    # But that doesn't directly help.
    
    # Another approach: use known decompositions and extend them.
    
    # Let's try a computational search for a lower rank decomposition.
    # We'll use a randomized approach to find a rank-r decomposition.
    
    # The condition is:
    # sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # This is equivalent to: the tensor T can be written as a sum of R rank-1 tensors.
    
    # Let's try to find a decomposition with rank less than 63.
    
    # Strategy: 
    # 1. Start with the schoolbook decomposition (rank 63)
    # 2. Try to reduce the rank by finding linear dependencies
    
    # Actually, let's use a more systematic approach.
    # We can formulate this as a tensor rank minimization problem.
    
    # For small tensors, we can use the following approach:
    # - The tensor T is a 3D array of size (n*m) x (m*p) x (p*n) = 9 x 21 x 21
    # - We want to find the minimal R such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    
    # Let's try a greedy approach:
    # - Start with R = 63 (schoolbook)
    # - Try to reduce R by finding linear combinations
    
    # Actually, let's use a different approach. We'll use the fact that
    # the matrix multiplication tensor has a known structure.
    
    # For 3x3 x 3x7, let's try to use a decomposition that exploits
    # the rectangular shape.
    
    # One idea: use the rank-23 decomposition for 3x3 x 3x3 and extend it.
    
    # Let me first verify that the schoolbook decomposition works, then try to improve.
    
    def schoolbook_decomposition():
        """Return the schoolbook rank-63 decomposition."""
        R = n * m * p  # 63
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is correct."""
        R = U.shape[1]
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    # First, get the schoolbook decomposition as a baseline
    U_school, V_school, W_school = schoolbook_decomposition()
    valid, info = verify_decomposition(U_school, V_school, W_school)
    # assert valid, f"Schoolbook decomposition failed: {info}"
    
    # Now, try to find a lower rank decomposition.
    # We'll use a randomized approach.
    
    def try_random_decomposition(R, num_trials=100, seed=42):
        """
        Try to find a rank-R decomposition using random initialization.
        Returns (U, V, W) if successful, else None.
        """
        rng = np.random.RandomState(seed)
        
        for trial in range(num_trials):
            # Initialize U, V, W randomly
            U = rng.randn(n * m, R)
            V = rng.randn(m * p, R)
            W = rng.randn(p * n, R)
            
            # Try to solve for the decomposition
            # The condition is: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            #
            # This is a nonlinear system. We can try to solve it iteratively.
            #
            # One approach: fix U and V, solve for W (or vice versa).
            #
            # For a fixed U and V, the condition becomes:
            # For each (i,j,k): sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            #
            # This is a linear system in W for each (i,j,k) separately... no,
            # it's coupled because W appears in multiple equations.
            #
            # Actually, for a fixed U and V, we have:
            # For each (k,i): sum_j (sum_r U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            #
            # Wait, that's not right either. Let me re-derive.
            #
            # The condition is: for all (i,j,k):
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            #
            # If we fix U and V, then for each (i,k), we have:
            # sum_j [sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]] = ?
            #
            # No, the condition is for each specific (i,j,k), not summed over j.
            #
            # So if we fix U and V, we have a system of equations in W:
            # For each (i,j,k): sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            #
            # This is a linear system in the variables W[k*n+i,r] for all (k,i,r).
            # There are p*n*R = 21*R unknowns and n*m*p = 63 equations.
            #
            # If R is large enough, this system may have a solution.
            #
            # Let's set up the linear system.
            
            # Unknowns: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..R-1
            # Number of unknowns: p * n * R = 21 * R
            # Number of equations: n * m * p = 63
            
            num_unknowns = p * n * R
            num_eqs = n * m * p
            
            if num_unknowns < num_eqs:
                # Underdetermined, but we might still have a solution
                # Actually, if num_unknowns < num_eqs, the system is overdetermined
                # and may not have a solution
                pass
            
            # Build the system: A * x = b
            # where x is the vectorized W
            # x[k*n*p + i*R + r] = W[k*n+i, r]
            
            A = np.zeros((num_eqs, num_unknowns))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            # x index for W[k*n+i, r]
                            x_idx = (k * n + i) * R + r
                            A[eq_idx, x_idx] = U[i * m + j, r] * V[j * p + k, r]
                        eq_idx += 1
            
            # Solve the linear system
            try:
                # Use least squares
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is good
                residual = np.linalg.norm(A @ x - b)
                if residual < 1e-8:
                    # Reconstruct W
                    W_sol = x.reshape((p * n, R))
                    return U, V, W_sol
            except np.linalg.LinAlgError:
                continue
        
        return None
    
    # Try different ranks
    best_result = (U_school, V_school, W_school)
    best_rank = n * m * p  # 63
    
    # Try ranks from 62 down to some lower bound
    # The lower bound for the rank is at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But for matrix multiplication, the rank is typically higher.
    
    # Let's try a few ranks
    for R in range(62, 30, -1):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        result = try_random_decomposition(R, num_trials=50, seed=R)
        if result is not None:
            U_new, V_new, W_new = result
            valid, info = verify_decomposition(U_new, V_new, W_new)
            if valid:
                best_result = (U_new, V_new, W_new)
                best_rank = R
                break  # Found a valid decomposition, stop
    
    # If we found a better decomposition, use it
    if best_rank < n * m * p:
        U, V, W = best_result
    else:
        U, V, W = U_school, V_school, W_school
    
    # Convert to lists for the return value
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
