import time
import numpy as np
import itertools
import random
from scipy.optimize import linprog

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 15
    
    t_start = time.time()
    deadline = 118.0 # Safety margin (limit is 120s)
    
    # 1. Precomputation
    # Generate all words as indices 0..2186
    # index = d1*q^(n-1) + ... + dn
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for val in t:
            idx = idx * q + val
        return idx
        
    N = q ** n
    all_words = [idx_to_tuple(i) for i in range(N)]
    
    # Precompute balls: balls[i] is a numpy array of indices within distance radius
    # Radius 1: center + 1-substitutions
    balls = []
    for i in range(N):
        w = list(all_words[i])
        ball = [i]
        for pos in range(n):
            old_val = w[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w[pos] = new_val
                    ball.append(tuple_to_idx(tuple(w)))
            w[pos] = old_val
        balls.append(np.array(ball, dtype=np.int32))
        
    # Convert to list of lists for easier manipulation if needed, but keep numpy for speed
    balls_np = np.array(balls, dtype=np.int32)
    
    # Precompute distance-2 neighbors for 2-packing
    # dist2_neighbors[i] is a list of indices at Hamming distance 2 from i
    dist2_neighbors = []
    for i in range(N):
        w = list(all_words[i])
        neighbors = []
        # Generate all pairs of positions
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                old1, old2 = w[pos1], w[pos2]
                # Change both positions to any value different from original
                # There are (q-1)^2 = 4 combinations for q=3
                for v1 in range(q):
                    if v1 == old1: continue
                    for v2 in range(q):
                        if v2 == old2: continue
                        w[pos1] = v1
                        w[pos2] = v2
                        neighbors.append(tuple_to_idx(tuple(w)))
                w[pos1] = old1
                w[pos2] = old2
        dist2_neighbors.append(neighbors)
        
    # 2. Construct a large 2-packing (Independent Set in dist-2 graph)
    # Greedy Randomized 2-Packing
    def find_2_packing(seed):
        random.seed(seed)
        available = [True] * N
        packing = []
        # Order vertices randomly or by degree? Random is fine for diversity.
        # We can also try to pick vertices that have many neighbors still available?
        # Simple greedy: pick random available, mark its neighbors unavailable.
        # To make it better, we can use a priority queue or just random selection with restarts.
        
        # A better greedy: Pick the vertex with the smallest degree in the remaining graph?
        # Or largest? Smallest degree (greedy coloring heuristic) often works well for MIS.
        # Let's try a simple randomized greedy:
        # 1. Create list of available vertices.
        # 2. While list not empty:
        #    Pick a vertex v from list (randomly or with some heuristic).
        #    Add v to packing.
        #    Remove v and all its neighbors from available.
        
        available_set = set(range(N))
        # To speed up, we can maintain a list and remove.
        # N=2187, so even O(N^2) is ~4.7M ops, which is fast.
        
        while available_set:
            # Heuristic: Pick vertex with min degree in induced subgraph?
            # Or just random?
            # Let's try random for now, maybe add a "min degree" variant if time permits.
            v = random.choice(list(available_set))
            packing.append(v)
            # Remove v and neighbors
            to_remove = [v] + dist2_neighbors[v]
            for u in to_remove:
                if u in available_set:
                    available_set.remove(u)
            
        return packing

    # 3. Greedy Completion to Covering Code
    def complete_to_cover(packing, seed=None):
        if seed is not None:
            random.seed(seed)
        # Initialize covered mask
        covered = np.zeros(N, dtype=bool)
        code = list(packing)
        for idx in code:
            covered[balls_np[idx]] = True
            
        # Precompute incidence matrix for greedy set cover
        # We can reuse the M from before or build it here.
        # For efficiency, let's build a sparse matrix M where M[i,j]=1 if j in balls[i]
        # This allows fast gain calculation: gain = M @ (1 - covered)
        
        # We'll use a simple loop for gain calculation if N is small, but vectorized is better.
        # Let's use the sparse matrix approach from the original code.
        
        import scipy.sparse as sp
        # Build M once outside? No, it's local.
        # Actually, we can just compute gains using the balls_np array directly if we vectorize.
        # gains[i] = np.sum(1 - covered[balls_np[i]])
        # This is a matrix-vector product if we have the incidence matrix.
        
        # Let's build the sparse matrix M here for the completion phase.
        row_indices = []
        col_indices = []
        data = []
        for i in range(N):
            for j in balls_np[i]:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)
        M = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
        
        while not covered.all():
            uncovered_mask = 1.0 - covered.astype(float)
            gains = M.dot(uncovered_mask)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            idx = candidates[random.randint(0, len(candidates)-1)]
            code.append(idx)
            covered[balls_np[idx]] = True
            
        return code

    best_code = None
    best_len = N + 1
    
    # Run several 2-packing initializations
    num_packing_trials = 20
    for i in range(num_packing_trials):
        if time.time() - t_start > deadline - 30: break
        packing = find_2_packing(seed=100 + i)
        code = complete_to_cover(packing, seed=200 + i)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            
    # Fallback to original greedy if packing didn't find anything good
    if best_code is None or best_len > 200:
        # Original greedy init
        def greedy_init(seed=None):
            if seed is not None:
                random.seed(seed)
            covered = np.zeros(N, dtype=bool)
            code = []
            import scipy.sparse as sp
            row_indices = []
            col_indices = []
            data = []
            for i in range(N):
                for j in balls_np[i]:
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(1)
            M = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
            
            while not covered.all():
                uncovered_mask = 1.0 - covered.astype(float)
                gains = M.dot(uncovered_mask)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                idx = candidates[random.randint(0, len(candidates)-1)]
                code.append(idx)
                covered[balls_np[idx]] = True
            return code
            
        for i in range(5):
            if time.time() - t_start > deadline - 10: break
            code = greedy_init(seed=42 + i)
            if len(code) < best_len:
                best_len = len(code)
                best_code = code[:]

    # --- Phase 3: Local Search (Add, Remove, Swap) ---
    # This is the most critical part for improvement.
    # Start with best_code found so far.
    
    if best_code is None:
        # Should not happen, but safe guard
        best_code = list(range(N)) 
        
    # Ensure best_code is a list
    best_code = list(best_code)
    
    # Precompute incidence structure for fast updates?
    # We maintain 'counts' array: counts[j] = number of codewords in best_code that cover word j.
    # A word is covered if counts[j] > 0.
    
    def get_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[balls_np[idx]] += 1
        return counts
    
    counts = get_counts(best_code)
    
    # Helper to check if a set of indices forms a valid cover
    def is_valid_cover(counts_arr):
        return np.all(counts_arr > 0)
        
    # Local Search Loop
    improved = True
    while improved and time.time() - t_start < deadline - 2:
        improved = False
        
        # 1. Remove Redundant Codewords
        # If removing idx keeps all words in its ball covered by others, remove it.
        # Condition: for all j in balls[idx], counts[j] > 1
        for idx in list(best_code):
            if time.time() - t_start > deadline - 2: break
            ball = balls_np[idx]
            if np.min(counts[ball]) > 1:
                # Remove
                counts[ball] -= 1
                best_code.remove(idx)
                improved = True
        
        # 2. Swap Moves
        # Try to replace one codeword u with another v such that:
        # - u is in code, v is not in code
        # - The new set is a valid cover
        
        if time.time() - t_start < deadline - 5:
            # Optimization: Only try swaps if we have time.
            num_swap_attempts = 1000
            for _ in range(num_swap_attempts):
                if time.time() - t_start > deadline - 2: break
                if len(best_code) == 0: break
                
                u = random.choice(best_code)
                v = random.randint(0, N - 1)
                if v in best_code: continue
                
                ball_u = balls_np[u]
                ball_v = balls_np[v]
                
                # Efficient check:
                # Words in ball_u \ ball_v will lose coverage from u.
                # If any of those had count 1, they become 0 -> Invalid.
                
                # We need to check min(counts[j]) for j in ball_u \ ball_v.
                # Creating masks is O(N). Let's try to optimize.
                # Since N=2187, O(N) is ~2k ops. 1000 swaps * 2k ops = 2M ops. Fast.
                
                mask_u = np.zeros(N, dtype=bool)
                mask_u[ball_u] = True
                mask_v = np.zeros(N, dtype=bool)
                mask_v[ball_v] = True
                
                losing_mask = mask_u & ~mask_v
                if np.any(losing_mask):
                    if np.min(counts[losing_mask]) <= 1:
                        continue # Invalid swap
                
                # Swap is valid
                counts[ball_u] -= 1
                counts[ball_v] += 1
                best_code.remove(u)
                best_code.append(v)
                improved = True
                
    # Final Cleanup: Remove any remaining redundancies
    counts = get_counts(best_code)
    changed = True
    while changed and time.time() - t_start < deadline:
        changed = False
        for idx in list(best_code):
            ball = balls_np[idx]
            if np.min(counts[ball]) > 1:
                counts[ball] -= 1
                best_code.remove(idx)
                changed = True

    # Convert indices back to list of lists of integers
    result = []
    for idx in best_code:
        w = []
        temp_idx = idx
        for _ in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
        
    return result
