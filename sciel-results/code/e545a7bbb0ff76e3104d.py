import numpy as np
import time
from fractions import Fraction
import itertools
import random

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    # Target Tensor T[i, j, k] = 1 if (i, j, k) is a valid multiplication term
    # Indices:
    # U index: a = i * m + j  (0..3)
    # V index: b = j * p + k  (0..5)
    # W index: c = k * n + i  (0..5)
    
    nm, mp, pn = n*m, m*p, p*n
    
    # The target tensor has 12 non-zero entries.
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    # Let's list these 12 index triples (a, b, c)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    # We want to find U, V, W of rank R such that sum_r U[:,r] V[:,r] W[:,r] == T_target
    # This is equivalent to finding R rank-1 tensors that sum to T_target.
    
    # Strategy: 
    # The schoolbook decomposition uses R=12 terms, each being e_a e_b e_c for the 12 triples above.
    # Can we do better? 
    # A rank-1 tensor is u v^T w^T (outer product).
    # If we restrict u, v, w to be vectors with entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}, 
    # we can enumerate possible rank-1 tensors.
    
    # However, enumerating all possible u, v, w is large.
    # Instead, let's look for a decomposition of the form:
    # T = sum_{r=1}^R t_r
    # where each t_r is a rank-1 tensor.
    
    # A key insight for small matrices: 
    # If we fix the "skeleton" of the decomposition (which indices are non-zero in U, V, W for each rank r),
    # the problem becomes solving a linear system for the values.
    
    # Let's try to find a rank 10 or 11 solution.
    # We will generate a pool of candidate rank-1 tensors.
    # A candidate is defined by (u, v, w) where u, v, w are sparse vectors.
    
    # To make this feasible, we will restrict the non-zero positions of u, v, w to subsets of the indices
    # that appear in the target tensor.
    
    # Indices appearing in target:
    # A_indices = set(a for a,b,c in target_indices) -> {0,1,2,3}
    # B_indices = set(b for a,b,c in target_indices) -> {0,1,2,3,4,5}
    # C_indices = set(c for a,b,c in target_indices) -> {0,1,2,3,4,5}
    
    # Actually, for the tensor to match T_target, any non-zero entry in T must be covered.
    # And any entry in T that is 0 must be cancelled out.
    
    # Let's use a more direct algebraic approach:
    # We know that for 2x2x3, the rank is likely 10 or 11.
    # Let's attempt to construct a rank 11 solution by modifying the schoolbook.
    # Schoolbook: 12 terms T_{ijk}.
    # If we can find 3 schoolbook terms that sum to a rank-2 tensor, we can replace them with 2 terms, reducing rank by 1.
    # Or 4 terms to rank 3, etc.
    
    # Let's check if any subset of schoolbook terms has rank < size.
    # This is equivalent to finding linear dependencies among the schoolbook tensors.
    # But they are distinct basis vectors in the space of tensors? No, they are specific rank-1 tensors.
    # The space of 2x2x3 tensors has dimension 4*6*6 = 144.
    # The 12 schoolbook tensors are linearly independent? 
    # If we view them as vectors in R^144, are they independent?
    # Yes, because they have disjoint supports? No, they don't have disjoint supports in the flattened tensor?
    # T_{000} has support at (0,0,0). T_{001} has support at (0,1,1). 
    # Wait, the indices are (a,b,c).
    # (0,0,0) corresponds to i=0,j=0,k=0.
    # (0,1,1) corresponds to i=0,j=0,k=1.
    # The supports are disjoint in the sense that each (a,b,c) triple is unique for each (i,j,k).
    # So the 12 schoolbook tensors are linearly independent.
    # Therefore, we cannot simply remove one. We must replace a set with a smaller set of *different* rank-1 tensors.
    
    # Let's try a "Guess and Check" for Rank 10 using a specific ansatz.
    # Many efficient decompositions for matrix multiplication use "Strassen-like" structures.
    # For 2x2, Strassen reduces 8 mults to 7.
    # For 2x2x3, we have 12 mults.
    # Can we reduce to 11?
    
    # Let's try to find a rank 11 solution by solving a linear system with fixed sparsity patterns.
    # We will iterate over all ways to assign "active" indices for U, V, W for each of the 11 ranks.
    # This is too large to iterate exhaustively.
    
    # Alternative: Randomized Local Search on the Discrete Lattice with Stronger Initialization.
    # The previous program did this but maybe the initialization was poor or the search space too constrained.
    
    # New Idea: **Use the fact that the target tensor is 0-1.**
    # If we restrict U, V, W to have entries in {0, 1}, the sum of rank-1 tensors is an integer tensor.
    # We want the sum to be exactly the 0-1 target tensor.
    # This means we are looking for a "disjoint" or "overlapping" cover of the target tensor by rank-1 tensors 
    # such that overlaps cancel out (if we allow negative numbers) or don't happen (if positive only).
    # Since we allow half-integers, we can have cancellations.
    
    # Let's try a **Genetic Algorithm** specifically optimized for the structure.
    # Represent a solution as a list of R tuples (u_r, v_r, w_r).
    # u_r, v_r, w_r are vectors of length nm, mp, pn with entries in {-1, -0.5, 0, 0.5, 1}.
    
    # To make it faster, we will fix the "sparsity pattern" to be similar to schoolbook but with some mixing.
    
    # Let's try a simpler approach: **Direct Linear Solve with Randomized Sparsity Patterns**.
    # For a fixed rank R, we choose R random sparse patterns for U, V, W.
    # Then we solve the linear system for the values.
    # If the solution is close to half-integers, we round and verify.
    
    target_ranks = [10, 11, 12]
    
    best_U, best_V, best_W = None, None, None
    best_rank = 12 # Fallback
    
    # Precompute target tensor as a flat vector for speed
    T_flat = np.zeros(nm * mp * pn)
    for a, b, c in target_indices:
        idx = (a * mp + b) * pn + c
        T_flat[idx] = 1.0
        
    def flatten_tensor(U, V, W, R):
        # Compute tensor T = sum_r u_r v_r w_r^T
        # Returns flat array
        T = np.zeros((nm, mp, pn))
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T += np.einsum('i,j,k->ijk', u, v, w)
        return T.flatten()
    
    def compute_error(U, V, W, R):
        T_calc = flatten_tensor(U, V, W, R)
        return np.linalg.norm(T_calc - T_flat)
    
    def try_solve_with_pattern(R, pattern_seed):
        """
        Try to solve for values given a fixed sparsity pattern.
        Pattern is defined by which entries in U, V, W are non-zero.
        We use a randomized pattern based on the seed.
        """
        rng = random.Random(pattern_seed)
        
        # Define sparsity: for each rank r, pick a small subset of indices for U, V, W
        # To ensure we can cover the target, we should bias towards indices present in the target.
        
        # Let's try a pattern where each rank r is associated with a "center" (i,j,k) 
        # and allows variations.
        
        # Actually, let's just pick random non-zero positions.
        # For rank R, we have R columns in U, V, W.
        # Let's assume each column in U has ~2 non-zeros, V has ~2, W has ~2.
        # Total variables = R * (2+2+2) = 6R.
        # Constraints: 144 (but many are 0=0).
        # The system is under-determined if we pick random patterns?
        # No, we have 12 non-zero targets. We need to satisfy 12 equations (plus 132 zero equations).
        
        # Let's construct the linear system A x = b.
        # x is the vector of non-zero values.
        # A is the Jacobian of the tensor flattening with respect to x.
        
        # This is complex to build dynamically.
        
        # Simpler: Use `scipy.optimize.least_squares` on the values, fixing the zeros.
        # But we want exact integer/half-integer solutions.
        
        # Let's go back to **Exhaustive Search on Small Integer Coefficients** for a Reduced Subspace.
        # We know the schoolbook works. 
        # Let's try to find a rank 11 solution by replacing 2 schoolbook terms with 1 new term + 1 modified term?
        
        # Let's try a specific known structure for 2x2x3.
        # Reference: "Fast matrix multiplication" literature.
        # For 2x2, rank is 7 (Strassen).
        # For 2x2x3, rank is likely 10 or 11.
        
        # Let's try to construct a rank 10 solution manually using a "tensor train" like decomposition 
        # or by solving a specific system.
        
        # I will implement a **Simulated Annealing** on the continuous values, 
        # but with a **hard projection** to the grid {-1, -0.5, 0, 0.5, 1} at the end of each cooling step,
        # and a **restart** mechanism if stuck.
        
        # To make it structurally different: I will use **Nelder-Mead** (simplex) which is derivative-free 
        # and can handle discrete-like landscapes better if initialized well, 
        # combined with **multi-start**.
        
        pass

    # Let's try a very different approach: **Integer Linear Programming (ILP) approximation via Local Search**.
    # We define the "state" as the matrices U, V, W with entries in {-1, -0.5, 0, 0.5, 1}.
    # We start with the schoolbook solution (entries 0 or 1).
    # We perform moves that change one entry to another allowed value.
    # We accept a move if it reduces the error ||T_calc - T_target||^2.
    # If we reach error 0, we have a valid decomposition.
    # If we get stuck, we allow "bad" moves with probability exp(-delta_error / T) (Simulated Annealing).
    
    # This is robust and doesn't rely on continuous gradients.
    
    def run_simulated_annealing(R, max_iters=50000):
        # Initialize with Schoolbook-like structure but with R ranks.
        # If R < 12, we can't just copy schoolbook.
        # We need a random initialization that is "close" to valid.
        
        # Strategy: Start with R random rank-1 tensors that have high overlap with target.
        # Or, start with the first R schoolbook terms, and set the rest to random small values?
        # No, that leaves parts of the target uncovered.
        
        # Better Initialization: 
        # Assign each of the 12 target indices to one of the R ranks.
        # Some ranks will handle multiple indices.
        
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # Distribute the 12 target indices among R ranks
        # Each rank r gets a list of (a,b,c) it is responsible for.
        assignments = [[] for _ in range(R)]
        for idx, (a,b,c) in enumerate(target_indices):
            r = idx % R
            assignments[r].append((a,b,c))
            
        # For each rank r, we want U[:,r], V[:,r], W[:,r] to approximate the sum of the assigned tensors.
        # This is still hard.
        
        # Let's just use random initialization with values in {-1, 0, 1} and run SA.
        # The error landscape is discrete.
        
        rng = np.random.default_rng(42)
        
        def random_init():
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(nm, R))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(mp, R))
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(pn, R))
            return U, V, W
        
        U, V, W = random_init()
        current_error = compute_error(U, V, W, R)
        
        T_start = 1.0
        T_end = 0.001
        cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_error = current_error
        
        for i in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Pick a random entry to change
            mat_idx = rng.integers(0, 3)
            if mat_idx == 0:
                mat = U
                shape = U.shape
            elif mat_idx == 1:
                mat = V
                shape = V.shape
            else:
                mat = W
                shape = W.shape
                
            r, c = rng.integers(0, shape[0]), rng.integers(0, shape[1])
            old_val = mat[r, c]
            
            # Pick a new value from allowed set
            allowed = [-1, -0.5, 0, 0.5, 1]
            new_val = allowed[rng.integers(0, len(allowed))]
            
            if new_val == old_val:
                continue
                
            # Apply
            mat[r, c] = new_val
            new_error = compute_error(U, V, W, R)
            
            delta = new_error - current_error
            
            if delta < 0:
                # Accept
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error < 1e-8:
                        return best_U, best_V, best_W
            else:
                # Accept with probability
                if T_start > 0:
                    prob = np.exp(-delta / T_start)
                    if rng.random() < prob:
                        current_error = new_error
                    else:
                        # Revert
                        mat[r, c] = old_val
                
            T_start *= cooling_rate
            
        return best_U, best_V, best_W, best_error

    # Run SA for Rank 10 and 11
    for R in [10, 11]:
        if time.time() > deadline - 20.0:
            break
            
        # Run multiple restarts
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
                
            # Use different seeds for random init
            np.random.seed(42 + restart)
            U, V, W = run_simulated_annealing(R, max_iters=10000)
            
            if U is not None:
                err = compute_error(U, V, W, R)
                if err < 1e-8:
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If SA didn't find exact, return best found (might be invalid, so fallback to schoolbook)
    # Check if best_U is None
    if best_U is None:
        # Fallback to Schoolbook
        U_sb = np.zeros((nm, 12))
        V_sb = np.zeros((mp, 12))
        W_sb = np.zeros((pn, 12))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

