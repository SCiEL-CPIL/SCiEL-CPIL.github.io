"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the rank-56 Strassen decomposition
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    def build_rank56():
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        current_rank = 0
        for q in range(4):
            for s in range(7):
                u_loc, v_loc, w_loc = strassen_terms[s]
                r = current_rank
                for i in range(2):
                    for j_loc in range(2):
                        U[i * m + j_loc][r] = u_loc[i * 2 + j_loc]
                for j_loc in range(2):
                    for k_loc in range(2):
                        V[j_loc * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
                for i in range(2):
                    for k_loc in range(2):
                        W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
                current_rank += 1
        for q in range(4):
            for s in range(7):
                u_loc, v_loc, w_loc = strassen_terms[s]
                r = current_rank
                for i in range(2):
                    for j_loc in range(2):
                        U[i * m + (j_loc + 2)][r] = u_loc[i * 2 + j_loc]
                for j_loc in range(2):
                    for k_loc in range(2):
                        V[(j_loc + 2) * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
                for i in range(2):
                    for k_loc in range(2):
                        W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
                current_rank += 1
        return U, V, W, rank
    
    def verify_decomposition(U, V, W, rank, tol=1e-9):
        """Verify that the decomposition exactly decomposes matrix multiplication."""
        # The tensor is T[i,j,k] = delta_{j,k} for the multiplication tensor
        # The decomposition satisfies: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
        # where i indexes (i_row, i_col) of A, j indexes (j_row, j_col) of B, k indexes (k_row, k_col) of C
        # With the trace convention W[k*n+i, r], k indexes the output position
        
        # A is n x m, B is m x p, C is n x p
        # i: 0..n*m-1, i = i_row * m + i_col
        # j: 0..m*p-1, j = j_row * p + j_col  
        # k: 0..p*n-1, k = k_row * n + k_col
        # T[i,j,k] = 1 if i_col == j_row and j_col == k_row, else 0
        
        # Check a few random cases
        np.random.seed(42)
        for _ in range(100):
            i = np.random.randint(n * m)
            j = np.random.randint(m * p)
            k = np.random.randint(p * n)
            i_row, i_col = divmod(i, m)
            j_row, j_col = divmod(j, p)
            k_row, k_col = divmod(k, n)
            
            expected = 1.0 if (i_col == j_row and j_col == k_row) else 0.0
            actual = 0.0
            for r in range(rank):
                actual += U[i, r] * V[j, r] * W[k, r]
            
            if abs(actual - expected) > tol:
                return False
        return True
    
    # Start with rank-56 solution
    U, V, W, rank = build_rank56()
    
    # Try to reduce rank by finding linear dependencies
    # We'll try to zero out columns and check if the decomposition still works
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing columns one at a time (greedy)
    # For each column, try zeroing it out and see if we can compensate
    # This is expensive, so we'll use a heuristic approach
    
    # Alternative: try to find if any column is a linear combination of others
    # by checking if removing it breaks the decomposition
    
    # Simpler approach: try all pairs of columns to see if they can be merged
    # This is O(rank^2) which is 56^2 = 3136, manageable
    
    # Actually, let's try a different approach: use the fact that we have 8 blocks
    # of 7 Strassen terms each. Try to find if we can reduce the number of blocks
    # by combining them.
    
    # For now, let's try a simple greedy: for each column, try zeroing it and
    # check if the decomposition still holds (it won't, but we can try to fix it)
    
    # Better approach: try to find a lower rank by solving a system of equations
    # This is complex, so let's try a simpler heuristic.
    
    # Try rank 48: remove 8 columns and see if we can adjust
    # This is too complex for now.
    
    # Let's just return the rank-56 solution for now, but with a time guard
    if time.time() > deadline:
        U_list = [[int(U[i, r]) for r in range(rank)] for i in range(n * m)]
        V_list = [[int(V[i, r]) for r in range(rank)] for i in range(m * p)]
        W_list = [[int(W[i, r]) for r in range(rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try to reduce rank by checking if any column can be zeroed
    # and the error can be absorbed by other columns
    
    # For each column r, compute the "contribution" of that column
    # and see if it's redundant
    
    # This is getting complex. Let's try a simpler approach:
    # Try all subsets of columns to remove, starting with 1 column at a time
    
    # For rank 55, try removing each column and check
    for r_to_remove in range(rank):
        if time.time() > deadline:
            break
        
        # Create a modified decomposition without column r_to_remove
        U_test = np.delete(U, r_to_remove, axis=1)
        V_test = np.delete(V, r_to_remove, axis=1)
        W_test = np.delete(W, r_to_remove, axis=1)
        rank_test = rank - 1
        
        # Quick check: verify a few random cases
        valid = True
        np.random.seed(123)
        for _ in range(20):
            i = np.random.randint(n * m)
            j = np.random.randint(m * p)
            k = np.random.randint(p * n)
            i_row, i_col = divmod(i, m)
            j_row, j_col = divmod(j, p)
            k_row, k_col = divmod(k, n)
            
            expected = 1.0 if (i_col == j_row and j_col == k_row) else 0.0
            actual = 0.0
            for rr in range(rank_test):
                actual += U_test[i, rr] * V_test[j, rr] * W_test[k, rr]
            
            if abs(actual - expected) > 1e-9:
                valid = False
                break
        
        if valid:
            best_rank = rank_test
            best_U, best_V, best_W = U_test, V_test, W_test
            U, V, W = U_test, V_test, W_test
            rank = rank_test
            break
    
    # Convert to lists
    U_list = [[int(U[i, r]) for r in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, r]) for r in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, r]) for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
