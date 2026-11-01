import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook Decomposition (Rank = n*m*p)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # 2. Combinatorial Search for Lower Rank
    # We search for rank R such that sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta(j1, j2)
    # We use integer arithmetic to avoid floating point errors.
    # We search for entries in {-1, 0, 1} or small integers.
    
    # The identity is:
    # For each i in [0, n), k in [0, p):
    #   The matrix M_{i,k} with entries M_{i,k}[j1, j2] = sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r]
    #   must equal the identity matrix I_m.
    
    # Note: W is indexed by (k, i) in the output factor W[k*n+i, r].
    # The term is U[i, j1, r] * V[j2, k, r] * W[k, i, r].
    
    # Let's define the check function using integers.
    # If we find integer solutions U_int, V_int, W_int such that the sum is 1 (or 4 if scaled), 
    # we can return them. The problem allows half-integers, so we can return U_int/2 etc if needed,
    # but integer solutions are valid half-integers (x.0).
    
    def check_integer_solution(U, V, W, rank):
        """
        Checks if U, V, W (integer arrays) satisfy the identity exactly.
        Identity: sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta(j1, j2)
        """
        for i in range(n):
            for k in range(p):
                # Target is I_m
                # Construct the matrix product
                # U_slice: (m, rank) -> U[i*m : (i+1)*m, :]
                # V_slice: (m, rank) -> V[:, k] is not right. V is (m*p, rank).
                # V row index is j2*p + k. So V_slice for fixed k is V[0:m*p:m*p? No.
                # V is indexed by [j*p + k].
                # So for fixed k, we need V[j*p + k, :] for j in 0..m-1.
                
                U_slice = U[i*m:(i+1)*m, :] # (m, rank)
                # V_slice: we need rows j*p+k for j=0..m-1
                V_slice = np.empty((m, rank), dtype=int)
                for j in range(m):
                    V_slice[j, :] = V[j*p + k, :]
                
                w_val = W[k*n + i, :] # (rank,)
                
                # Matrix product: sum_r U_slice[j1, r] * V_slice[j2, r] * w_val[r]
                # This is (U_slice * w_val) @ V_slice.T
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                
                if not np.array_equal(mat, np.eye(m, dtype=int)):
                    return False
        return True

    # Search Strategy:
    # Try ranks from schoolbook - 1 downwards.
    # For each rank, try random sparse integer vectors.
    # Since the space is huge, we use a heuristic:
    # 1. Initialize U, V, W with zeros.
    # 2. Randomly pick a column r.
    # 3. Assign random small integers (-1, 0, 1) to U[:, r], V[:, r], W[:, r].
    # 4. Check if the current partial sum (sum over filled columns) plus the new column
    #    can satisfy the identity. This is hard to check incrementally easily.
    
    # Alternative: Random restarts with full check.
    # Given the small size (n=2, m=5, p=6), rank 60 is schoolbook.
    # Known lower bounds for 2x5x6? 
    # Rank of 2x5x6 is at least max(nm, mp, pn) = max(10, 30, 12) = 30? No, that's for 2x2x2.
    # For 2x5x6, the border rank is likely close to schoolbook.
    # However, we can try to find *any* improvement.
    
    # A more effective search for small integers:
    # We can fix the structure of U and V to be somewhat structured (e.g., Kronecker products)
    # or just random sparse.
    
    # Let's try a targeted search for rank R in [rank_schoolbook - 10, rank_schoolbook - 1].
    # We will use random sparse vectors with entries in {-1, 0, 1}.
    
    max_search_time = 110.0
    ranks_to_try = range(rank_schoolbook - 1, max(29, rank_schoolbook - 15), -1)
    
    for rank in ranks_to_try:
        if time.time() - start_time > max_search_time:
            break
            
        # Try a few random restarts for this rank
        restarts = 500
        for _ in range(restarts):
            if time.time() - start_time > max_search_time:
                break
            
            # Initialize with zeros
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            
            # Fill with random sparse integers
            # Density around 0.1 to 0.2
            for r in range(rank):
                # Pick random non-zero entries
                # U
                if random.random() < 0.5:
                    idx = random.randint(0, n*m - 1)
                    U[idx, r] = random.choice([-1, 1])
                # V
                if random.random() < 0.5:
                    idx = random.randint(0, m*p - 1)
                    V[idx, r] = random.choice([-1, 1])
                # W
                if random.random() < 0.5:
                    idx = random.randint(0, p*n - 1)
                    W[idx, r] = random.choice([-1, 1])
            
            if check_integer_solution(U, V, W, rank):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                # We found a valid lower rank. We could stop, but let's see if we can go lower.
                # However, the probability of finding a valid random sparse solution is extremely low.
                # This search is likely to fail.
                pass

    # If no lower rank found via random sparse search, return schoolbook.
    # The schoolbook solution is guaranteed valid.
    
    # Convert to lists for return
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

# EVOLVE-BLOCK-END
