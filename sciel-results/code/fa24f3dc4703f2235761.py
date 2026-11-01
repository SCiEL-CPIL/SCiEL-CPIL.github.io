import numpy as np
import itertools
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 15
    # If we find a solution of rank 15, return it.
    # Otherwise, fallback to schoolbook rank 18.
    
    # Dimensions of the tensors
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T of shape (n, m, p) with all 1s
    # Flattened to 1D for easier manipulation
    T_flat = np.ones(n * m * p, dtype=int)
    
    # Generate candidate rank-1 tensors
    # We restrict vectors to have at most 3 non-zero entries, values in {-1, 0, 1}
    # This is a heuristic to keep the search space manageable.
    
    candidates = []
    
    # Helper to generate vectors with <= k non-zeros
    def gen_vectors(dim, max_nz):
        vecs = []
        # Iterate over all subsets of indices of size <= max_nz
        for size in range(0, max_nz + 1):
            for indices in itertools.combinations(range(dim), size):
                # Generate all sign combinations for these indices
                signs = itertools.product([-1, 1], repeat=size)
                for s in signs:
                    v = np.zeros(dim, dtype=int)
                    for idx, sign in zip(indices, s):
                        v[idx] = sign
                    vecs.append(tuple(v))
        return vecs

    # To avoid memory explosion, we might not generate all at once if too many.
    # dim_u=6, dim_v=9, dim_w=6.
    # max_nz=3.
    # Count u: sum( C(6,k) * 2^k for k in 0..3 ) = 1 + 12 + 60 + 160 = 233
    # Count v: sum( C(9,k) * 2^k for k in 0..3 ) = 1 + 18 + 144 + 672 = 835
    # Count w: 233
    # Total candidates: 233 * 835 * 233 ~ 45 million.
    # This is too many to store in a list of numpy arrays or tuples in Python efficiently within 120s.
    
    # Strategy: Iterate over u and v, and for each pair, see if we can complete the sum with w's?
    # No, the sum is over c.
    
    # Alternative: Use a smaller set of candidates.
    # Maybe restrict to max_nz=2?
    # Count u: 1 + 12 + 60 = 73
    # Count v: 1 + 18 + 144 = 163
    # Count w: 73
    # Total: 73 * 163 * 73 ~ 866,000.
    # This is manageable.
    
    # Let's try max_nz=2 first. If it fails, try max_nz=3 with more aggressive pruning or sampling.
    
    max_nz = 2
    vecs_u = gen_vectors(dim_u, max_nz)
    vecs_v = gen_vectors(dim_v, max_nz)
    vecs_w = gen_vectors(dim_w, max_nz)
    
    # Generate all candidate tensors
    # t = u (outer) v (outer) w
    # Flatten and store as tuple
    cand_list = []
    for u in vecs_u:
        for v in vecs_v:
            for w in vecs_w:
                # Compute outer product
                # u is (n*m,), v is (m*p,), w is (p*n,)
                # We need to map indices correctly.
                # u[i*m+j], v[j*p+k], w[k*n+i]
                # Tensor T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                
                t = np.zeros(n*m*p, dtype=int)
                # Iterate over non-zeros to be fast?
                # Or just use numpy broadcasting if we reshape.
                # Reshape u to (n, m), v to (m, p), w to (p, n)
                u_r = np.array(u).reshape(n, m)
                v_r = np.array(v).reshape(m, p)
                w_r = np.array(w).reshape(p, n)
                
                # T[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
                # This is not a simple outer product of 3 vectors.
                # It's a specific contraction.
                # We can compute it by iterating over i, j, k.
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            t[i*m*p + j*p + k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
                
                # Only keep non-zero tensors
                if np.any(t != 0):
                    cand_list.append(tuple(t))
                    
    # Remove duplicates
    cand_list = list(set(cand_list))
    
    # Sort candidates?
    # Maybe sort by number of non-zeros or sum of squares?
    
    # Backtracking search
    # Target: (1,)*18
    target = tuple([1] * (n*m*p))
    
    found_solution = None
    
    # Use a recursive search with pruning
    # To speed up, we can limit the depth or use a heuristic ordering.
    
    # Given the small number of candidates (~800k), we can try to find a subset of 15.
    # This is still a combinatorial explosion.
    # We will use a randomized approach to pick candidates for the search, or just a greedy approach.
    
    # Greedy Approach:
    # Start with empty sum.
    # At each step, pick the candidate that minimizes the distance to target.
    # If we get stuck, backtrack.
    
    # Given time constraints, I will implement a simple randomized local search on the coefficients of 15 fixed random supports.
    # This is more likely to succeed than exhaustive subset sum.
    
    # Let's switch to the randomized local search on 15 random supports.
    
    def random_search():
        # Pick 15 random supports (vectors)
        # Optimize their signs and values (from {-1, 0, 1})
        
        # Initialize 15 random vectors
        U = np.zeros((dim_u, 15), dtype=int)
        V = np.zeros((dim_v, 15), dtype=int)
        W = np.zeros((dim_w, 15), dtype=int)
        
        for c in range(15):
            # Random sparse vectors
            for idx in np.random.choice(dim_u, 3, replace=False):
                U[idx, c] = np.random.choice([-1, 1])
            for idx in np.random.choice(dim_v, 3, replace=False):
                V[idx, c] = np.random.choice([-1, 1])
            for idx in np.random.choice(dim_w, 3, replace=False):
                W[idx, c] = np.random.choice([-1, 1])
                
        # Compute sum tensor
        def compute_sum(U, V, W):
            S = np.zeros(n*m*p, dtype=int)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(15):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        S[i*m*p + j*p + k] = val
            return S
            
        S = compute_sum(U, V, W)
        error = np.sum((S - T_flat)**2)
        
        # Local search
        best_error = error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(10000):
            # Pick a random term c and a random index in one of the vectors
            c = np.random.randint(15)
            vec_choice = np.random.randint(3) # 0:U, 1:V, 2:W
            if vec_choice == 0:
                idx = np.random.randint(dim_u)
                old_val = U[idx, c]
                U[idx, c] = np.random.choice([-1, 0, 1])
                if U[idx, c] == old_val:
                    continue
                S_new = compute_sum(U, V, W)
                new_error = np.sum((S_new - T_flat)**2)
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error == 0:
                        return best_U, best_V, best_W
                else:
                    U[idx, c] = old_val
            elif vec_choice == 1:
                idx = np.random.randint(dim_v)
                old_val = V[idx, c]
                V[idx, c] = np.random.choice([-1, 0, 1])
                if V[idx, c] == old_val:
                    continue
                S_new = compute_sum(U, V, W)
                new_error = np.sum((S_new - T_flat)**2)
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error == 0:
                        return best_U, best_V, best_W
                else:
                    V[idx, c] = old_val
            else:
                idx = np.random.randint(dim_w)
                old_val = W[idx, c]
                W[idx, c] = np.random.choice([-1, 0, 1])
                if W[idx, c] == old_val:
                    continue
                S_new = compute_sum(U, V, W)
                new_error = np.sum((S_new - T_flat)**2)
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error == 0:
                        return best_U, best_V, best_W
                else:
                    W[idx, c] = old_val
                    
        return None
        
    start_time = time.time()
    solution = random_search()
    
    if solution:
        U, V, W = solution
        # Convert to lists of lists
        U_list = U.T.tolist()
        V_list = V.T.tolist()
        W_list = W.T.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
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

# EVOLVE-BLOCK-END
