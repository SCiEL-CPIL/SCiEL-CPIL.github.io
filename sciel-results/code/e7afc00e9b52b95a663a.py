"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Precompute the trivial decomposition
    def make_trivial():
        U = [[0] * trivial_rank for _ in range(n * m)]
        V = [[0] * trivial_rank for _ in range(m * p)]
        W = [[0] * trivial_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_trivial()
    best_rank = trivial_rank
    
    # For a rank-r decomposition, we need:
    # U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
    # Such that for each (i,j,k): sum_t U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] = 1
    # And for all other combinations, the sum = 0
    
    # We'll search for decompositions with sparse structure
    # Strategy: try small ranks and see if we can find valid patterns
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But due to algebraic constraints, the actual minimum is likely higher
    
    # Let's try a structured search: assume each factor has a specific sparsity pattern
    # For rank r, each column t defines a "term" with:
    #   U support: subset of {0,...,n*m-1}
    #   V support: subset of {0,...,m*p-1}  
    #   W support: subset of {0,...,p*n-1}
    
    # A term contributes to (i,j,k) if:
    #   i*m+j in U_support AND j*p+k in V_support AND k*n+i in W_support
    # The coefficient is U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
    
    # For exact decomposition, we need the sum over all terms to equal 1 for valid (i,j,k)
    # and 0 for invalid ones.
    
    # Let's try a different approach: use the fact that for small matrices,
    # we can enumerate possible "patterns" for each rank term.
    
    # Key insight: if we can find a set of r "triplets" (u_pattern, v_pattern, w_pattern)
    # such that the linear system has a solution, we're done.
    
    # Let's try ranks from 28 upward and see if we can find a solution quickly.
    # For each rank, we'll try random sparse patterns and solve the linear system.
    
    def try_rank(r, max_attempts=100):
        """Try to find a rank-r decomposition with random sparse patterns."""
        if r < max(n*m, m*p, p*n):
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Generate random sparse patterns for each term
            # Each term has a support in U, V, W
            # We'll use a structured approach: each term covers a "box" in the tensor
            
            # For now, let's try a simple heuristic:
            # Each term t covers indices where:
            #   U has 1 at some (i,j), V has 1 at some (j',k), W has 1 at some (k',i')
            # The term contributes to (i,j,k) if i=i', j=j', k=k' (diagonal-like)
            # But we need off-diagonal contributions too.
            
            # Let's try: for each term, pick random subsets
            u_supports = []
            v_supports = []
            w_supports = []
            
            for t in range(r):
                # Random sparse pattern
                u_size = np.random.randint(1, min(n*m, 5) + 1)
                v_size = np.random.randint(1, min(m*p, 5) + 1)
                w_size = np.random.randint(1, min(p*n, 5) + 1)
                
                u_idx = np.random.choice(n*m, size=u_size, replace=False)
                v_idx = np.random.choice(m*p, size=v_size, replace=False)
                w_idx = np.random.choice(p*n, size=w_size, replace=False)
                
                u_supports.append(set(u_idx.tolist()))
                v_supports.append(set(v_idx.tolist()))
                w_supports.append(set(w_idx.tolist()))
            
            # Build the linear system
            # Variables: U[i][t], V[j][t], W[k][t] for all i,j,k,t
            # But this is nonlinear! The constraint is:
            #   sum_t U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] = delta(i,j,k)
            
            # This is a system of polynomial equations, not linear.
            # Let's linearize by fixing some variables.
            
            # Alternative: assume U and V have entries in {0,1} and solve for W,
            # or use a different approach.
            
            # Let's try: fix U to be 0/1 sparse, then the constraint becomes:
            #   sum_t U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] = delta
            # If U[i*m+j][t] is 0 or 1, and we fix V, then it's linear in W.
            
            # For simplicity, let's try a very structured approach:
            # Assume each term t corresponds to a unique (i,j,k) triple.
            # Then U[i*m+j][t] = 1 if this is the term for (i,j,k), else 0
            # Similarly for V and W.
            # This gives the trivial decomposition with rank = n*m*p.
            
            # To get lower rank, we need terms that cover multiple (i,j,k) triples.
            
            # Let's try: each term covers a "line" in one dimension.
            # E.g., term t covers all (i,j,k) where i = i_t (a fixed i).
            # Then U[i_t*m+j][t] can be nonzero for all j, V[j*p+k][t] for all j,k,
            # W[k*n+i_t][t] for all k.
            
            # This is getting complex. Let's just return the trivial solution for now
            # and add more sophisticated search in future iterations.
            
            pass
        
        return None
    
    # For now, return the trivial decomposition
    # Future work: implement the actual search
    return best_result
# EVOLVE-BLOCK-END
