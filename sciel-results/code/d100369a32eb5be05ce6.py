"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import sys
import itertools

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Theoretical lower bound
    lower_bound = max(n * p, m * p, n * m)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Try ranks from lower bound to schoolbook
    for r in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
            
        # Attempt to find a rank-r decomposition using systematic backtracking
        result = _solve_csp(n, m, p, r, deadline)
        if result is not None:
            return result
            
    # Fallback to schoolbook if no better solution found
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _solve_csp(n, m, p, r, deadline):
    """
    Systematic backtracking search for a rank-r decomposition.
    Variables: U[i*m+j][c], V[j*p+k][c], W[k*n+i][c]
    Constraints: Sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] == 1 for all i,j,k
    
    To make this tractable, we restrict the domain of values to a small set
    and use symmetry breaking.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Domain of possible values
    # We assume integer or half-integer values
    domain = [0, 1, -1, 0.5, -0.5]
    
    # Initialize matrices with None to indicate unassigned
    U = [[None] * r for _ in range(nm)]
    V = [[None] * r for _ in range(mp)]
    W = [[None] * r for _ in range(pn)]
    
    # We will assign variables column by column, but interleave U, V, W for a column?
    # Actually, the constraints couple all columns.
    # A better variable ordering: Assign column by column.
    # For each column c, assign all U, V, W entries for that column.
    # Then check partial sums? No, the sum is over c.
    
    # Alternative: Assign individual variables.
    # Total variables = r * (nm + mp + pn)
    # For r=15, nm=6, mp=15, pn=10 -> 15 * 31 = 465 variables.
    # This is too many for naive backtracking.
    
    # Optimization: Exploit sparsity.
    # In rank-minimal decompositions, factors are often sparse.
    # We will search for solutions where each column has limited support.
    
    # Strategy:
    # 1. Enumerate possible "patterns" for each column? Too many.
    # 2. Use a heuristic: Start with a known good structure?
    
    # Let's try a different structural approach: 
    # Since we cannot use external solvers, we will use a "Guess and Check" 
    # with aggressive pruning based on the remaining capacity of the sum.
    
    # However, for 2x3x5, the rank is likely 15 (the lower bound).
    # Let's try to verify if rank 15 is possible with a specific structured guess.
    
    # Known result: The rank of matrix multiplication M_{2,3,5} is 15.
    # A constructive proof exists.
    # We can try to construct it explicitly using algebraic identities.
    
    # If we can hardcode the structure or find it via a smaller search space, it's better.
    
    # Let's try a randomized search but with a much tighter constraint check and 
    # a different representation: We only store non-zeros.
    
    # Actually, the most robust "different" strategy in pure Python for this 
    # small instance is to use the fact that the tensor is small and use 
    # **Exact Arithmetic Search with Memoization** on the residual tensor.
    
    # State: (current_residual_tensor, remaining_columns)
    # This is still huge.
    
    # Let's fall back to a highly optimized **Genetic Algorithm** with 
    # **Elitism** and **Crossover**, which is structurally different from 
    # simple local search (which is 1-point mutation).
    # Or **Simulated Annealing**.
    
    # Given the stagnation, let's try **Simulated Annealing** with 
    # **Vectorized NumPy Operations** for speed, allowing many more 
    # evaluations per second.
    
    return _try_simulated_annealing(n, m, p, r, deadline)

def _try_simulated_annealing(n, m, p, r, deadline):
    """
    Simulated Annealing with NumPy vectorization.
    This allows evaluating thousands of candidates per second.
    """
    import numpy as np
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute index mapping for vectorized dot products
    # U shape: (nm, r)
    # V shape: (mp, r)
    # W shape: (pn, r)
    
    # We need to check: Sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a tensor contraction.
    
    # Let's represent U, V, W as 2D arrays.
    # To check validity, we can compute the resulting tensor T of shape (n, m, p).
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We can vectorize this check.
    
    def _evaluate(U, V, W):
        """
        Returns the number of (i,j,k) entries that equal 1.
        Uses NumPy for speed.
        """
        # Reshape to 3D tensors for broadcasting?
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # We want T[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
        
        # This is not a standard dot product due to index permutations.
        # We can loop over c (rank) which is small (15-30).
        
        T = np.zeros((n, m, p), dtype=np.float64)
        
        for c in range(r):
            # u_c: (n, m)
            # v_c: (m, p)
            # w_c: (p, n)
            u_c = U_3d[:, :, c]
            v_c = V_3d[:, :, c]
            w_c = W_3d[:, :, c]
            
            # Term for this c: 
            # We need to multiply U[i,j], V[j,k], W[k,i] and sum over c.
            # This is like a tensor product.
            
            # For each i, j, k:
            # We can do: 
            # temp[i, j, k] = u_c[i, j] * v_c[j, k] * w_c[k, i]
            
            # Broadcasting:
            # u_c[:, :, None] (n, m, 1)
            # v_c[:, :, None] ? No, v_c is (m, p). We need to align j and k.
            # v_c[j, k] -> (1, m, p)
            # w_c[k, i] -> (p, n, 1)
            
            # Let's construct:
            # u_term = u_c[:, :, np.newaxis]  # (n, m, 1)
            # v_term = v_c[np.newaxis, :, :]  # (1, m, p)
            # w_term = w_c.T[:, :, np.newaxis] # (n, p, 1) -> Need (p, n) -> (1, p, n)?
            
            # Wait, w_c is (p, n). We need w_c[k, i].
            # So w_term should be (1, p, n) if we align k with p-axis and i with n-axis?
            # But the axes are (i, j, k).
            # u_c[i, j] -> axis 0 (i), axis 1 (j)
            # v_c[j, k] -> axis 1 (j), axis 2 (k)
            # w_c[k, i] -> axis 2 (k), axis 0 (i)
            
            # This is a diagonal extraction from a 3-way product.
            
            # Efficient way:
            # For each c, compute the contribution.
            # contrib = np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
            
            T += np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
            
        # Count how many are exactly 1 (with tolerance)
        # Since we use floats, we check abs(T - 1) < 1e-9
        return np.sum(np.abs(T - 1) < 1e-9)

    def _score(U, V, W):
        """
        Score = number of correct entries.
        """
        return _evaluate(U, V, W)

    # Initialize with random small integers
    def _random_init():
        U = np.random.choice([0, 1, -1, 0.5, -0.5], size=(nm, r))
        V = np.random.choice([0, 1, -1, 0.5, -0.5], size=(mp, r))
        W = np.random.choice([0, 1, -1, 0.5, -0.5], size=(pn, r))
        return U, V, W

    best_U, best_V, best_W = _random_init()
    best_score = _score(best_U, best_V, best_W)
    
    # Simulated Annealing
    current_U, current_V, current_W = best_U.copy(), best_V.copy(), best_W.copy()
    current_score = best_score
    
    T_start = 100.0
    T_end = 1.0
    steps = 50000
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (step / steps)
        
        # Generate a neighbor
        # Pick a random matrix and a random entry
        mat_choice = np.random.randint(0, 3)
        if mat_choice == 0:
            idx = np.random.randint(0, nm)
            col = np.random.randint(0, r)
            old_val = current_U[idx, col]
            new_val = np.random.choice([0, 1, -1, 0.5, -0.5])
            current_U[idx, col] = new_val
        elif mat_choice == 1:
            idx = np.random.randint(0, mp)
            col = np.random.randint(0, r)
            old_val = current_V[idx, col]
            new_val = np.random.choice([0, 1, -1, 0.5, -0.5])
            current_V[idx, col] = new_val
        else:
            idx = np.random.randint(0, pn)
            col = np.random.randint(0, r)
            old_val = current_W[idx, col]
            new_val = np.random.choice([0, 1, -1, 0.5, -0.5])
            current_W[idx, col] = new_val
            
        new_score = _score(current_U, current_V, current_W)
        
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = current_U.copy()
                best_V = current_V.copy()
                best_W = current_W.copy()
                
                if best_score == n * m * p:
                    # Found a perfect solution
                    # Convert back to list of lists for the interface
                    return {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
        else:
            # Revert
            if mat_choice == 0:
                current_U[idx, col] = old_val
            elif mat_choice == 1:
                current_V[idx, col] = old_val
            else:
                current_W[idx, col] = old_val

    # If we didn't find a perfect solution, return None to try next rank
    return None

# EVOLVE-BLOCK-END
