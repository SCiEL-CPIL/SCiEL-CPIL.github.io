import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute decomposed tensor D[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    D = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
    
    return np.allclose(D, T, atol=1e-9, rtol=1e-9)

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 40).
    2. Attempt to reduce rank by checking if any term is linearly dependent
       on the others in the tensor space, or by using algebraic structure.
    3. Use time-bounded search with a safety margin before the 150s limit.
    
    The key insight for small matrices: the multiplication tensor for n x m x p
    has a specific structure. For 2x4x5, we can attempt to find a lower rank
    decomposition by exploiting the fact that some entries may be expressible
    as linear combinations of others.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 40
    
    # Get the schoolbook solution as fallback
    schoolbook = _schoolbook()
    
    # Verify schoolbook solution
    if not _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_solution = schoolbook
    best_rank = schoolbook_rank
    
    # Strategy 1: Try to reduce rank by checking linear dependencies
    # among the rank-1 tensors in the schoolbook decomposition.
    # 
    # Each term in the schoolbook decomposition is a rank-1 tensor:
    #   e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # 
    # If we can find a subset of these that are linearly dependent,
    # we might be able to reduce the rank. However, for the standard
    # multiplication tensor, these are typically linearly independent.
    # 
    # A more promising approach: use the structure of the problem.
    # For matrix multiplication, there are known lower bounds and
    # upper bounds on the rank. For 2x4x5, the schoolbook rank is 40.
    # 
    # Known results:
    # - Strassen's algorithm for 2x2x2 has rank 7 (vs schoolbook 8)
    # - For larger matrices, the rank is generally close to n*m*p
    # 
    # Let's try a randomized approach: perturb the schoolbook decomposition
    # and see if we can find a lower rank solution.
    
    # Strategy 2: Try rank = schoolbook_rank - 1, schoolbook_rank - 2, etc.
    # For each candidate rank, try to solve for U, V, W using a nonlinear
    # least-squares approach or by exploiting the structure.
    
    # For small ranks, we can use a greedy approach:
    # Start with the schoolbook terms and try to merge or eliminate terms.
    
    # Actually, let's try a more systematic approach:
    # The multiplication tensor T can be viewed as a 3-way array.
    # We want to find the minimum number of rank-1 tensors that sum to T.
    # This is the tensor rank problem, which is NP-hard in general.
    
    # For this specific case (2x4x5), let's try to use the following insight:
    # The tensor has a "banded" structure due to the matrix multiplication
    # pattern. Specifically, T[i*m+j, j*p+k, k*n+i] = 1 only when the indices
    # are compatible (i.e., the middle index j connects the first and second
    # dimensions, and k connects the second and third).
    
    # Let's try to see if we can find a rank-39 decomposition by checking
    # if one of the 40 schoolbook terms can be expressed as a linear combination
    # of the others. This is unlikely for the standard multiplication tensor,
    # but worth checking.
    
    # Build the matrix of all 40 rank-1 tensors as vectors
    tensor_size = (n * m) * (m * p) * (p * n)
    tensor_matrix = np.zeros((schoolbook_rank, tensor_size), dtype=np.float64)
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * (m * p) * (p * n) + \
                      (j * p + k) * (p * n) + \
                      (k * n + i)
                tensor_matrix[col_idx, idx] = 1.0
                col_idx += 1
    
    # Check the rank of this matrix
    tensor_rank = np.linalg.matrix_rank(tensor_matrix, tol=1e-10)
    
    # If the tensor rank is less than schoolbook_rank, we might be able
    # to reduce the decomposition rank.
    if tensor_rank < schoolbook_rank:
        # This would mean there's a linear dependency among the schoolbook terms
        # For now, we'll just note this and continue with the schoolbook solution
        pass
    
    # Strategy 3: Try to use algebraic structure to find a lower rank decomposition
    # 
    # For matrix multiplication, one can sometimes use the following trick:
    # If we can find a set of coefficients such that a linear combination of
    # the rank-1 tensors equals zero, we can reduce the rank.
    # 
    # This is equivalent to finding a non-trivial solution to:
    #   sum_c alpha_c * (e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}) = 0
    # for all (i,j,k) not in the support of T.
    # 
    # This is a complex constraint satisfaction problem. For 2x4x5, let's
    # try a few heuristic approaches.
    
    # Heuristic: Try to pair up terms that share the same i or j or k
    # and see if they can be combined.
    
    # For example, fix i=0 and try to combine terms with different j,k.
    # This is not straightforward, so let's try a different approach.
    
    # Strategy 4: Use the fact that the problem is small enough to try
    # a brute-force search over possible rank values.
    # 
    # For each candidate rank r from schoolbook_rank-1 down to some minimum,
    # try to find U, V, W with that rank. This is computationally expensive,
    # so we'll limit the search.
    
    # Given the time constraints, let's try a few specific ranks and see
    # if we can find a valid decomposition.
    
    # For now, let's just return the schoolbook solution. Finding a lower
    # rank decomposition for 2x4x5 is a hard open problem, and the time
    # limit is 120s.
    
    # However, let's try one more thing: check if the known lower bounds
    # for matrix multiplication rank apply here.
    
    # Lower bound for n x m x p matrix multiplication:
    #   rank >= max(n*m, m*p, p*n)
    # For 2x4x5: max(8, 20, 10) = 20
    # 
    # Upper bound: n*m*p = 40
    # 
    # So the rank is between 20 and 40. The schoolbook rank is 40.
    # 
    # There are known algorithms that achieve ranks below n*m*p for some
    # dimensions, but for 2x4x5, I'm not aware of a specific algorithm
    # that beats the schoolbook rank.
    
    # Let's try a randomized search for a lower rank decomposition.
    # We'll try ranks from 39 down to 30, and for each rank, we'll try
    # a few random initializations and use gradient descent to minimize
    # the reconstruction error.
    
    # This is a heuristic approach and may not find the optimal solution,
    # but it's worth trying given the time limit.
    
    for candidate_rank in range(schoolbook_rank - 1, max(19, schoolbook_rank - 10), -1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of rank candidate_rank
        # using a randomized approach.
        
        # Initialize U, V, W with random values
        np.random.seed(42)  # For reproducibility
        U_init = np.random.randn(n * m, candidate_rank)
        V_init = np.random.randn(m * p, candidate_rank)
        W_init = np.random.randn(p * n, candidate_rank)
        
        # Define the objective function: reconstruction error
        def objective(params):
            # Unpack parameters
            U = params[:n * m * candidate_rank].reshape(n * m, candidate_rank)
            V = params[n * m * candidate_rank : n * m * candidate_rank + m * p * candidate_rank].reshape(m * p, candidate_rank)
            W = params[n * m * candidate_rank + m * p * candidate_rank :].reshape(p * n, candidate_rank)
            
            # Compute decomposed tensor
            D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            
            # Compute error
            error = np.sum((D - T) ** 2)
            return error
        
        # Use scipy's minimize to try to find a solution
        from scipy.optimize import minimize
        
        params_init = np.concatenate([
            U_init.flatten(),
            V_init.flatten(),
            W_init.flatten()
        ])
        
        # Try a few iterations of gradient descent
        try:
            result = minimize(objective, params_init, method='L-BFGS-B', 
                            options={'maxiter': 100, 'maxfun': 500})
            
            if result.success or result.fun < 1e-6:
                # Check if we found a valid solution
                U_sol = result.x[:n * m * candidate_rank].reshape(n * m, candidate_rank)
                V_sol = result.x[n * m * candidate_rank : n * m * candidate_rank + m * p * candidate_rank].reshape(m * p, candidate_rank)
                W_sol = result.x[n * m * candidate_rank + m * p * candidate_rank :].reshape(p * n, candidate_rank)
                
                # Verify the solution
                D_sol = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol, optimize=True)
                if np.allclose(D_sol, T, atol=1e-6, rtol=1e-6):
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_sol]
                    V_list = [[float(x) for x in row] for row in V_sol]
                    W_list = [[float(x) for x in row] for row in W_sol]
                    
                    best_solution = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = candidate_rank
                    break  # Found a valid lower rank solution
        except Exception:
            pass
    
    # Return the best solution found
    return best_solution
# EVOLVE-BLOCK-END
