import numpy as np
import random
import time

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # But we handle general q, n, radius if small enough.
    
    N = q ** n
    if N > 10000:
        # Fallback to simple greedy if too large for this specific NumPy approach
        # (Memory for balls matrix is N*N)
        return _fallback_greedy(q, n, radius)

    # Precompute all words as tuples? No, use indices 0..N-1.
    # We need to map index to word to generate balls.
    # Index to word: base-q representation.
    
    # 1. Generate balls matrix
    # balls[i, j] = 1 if dist(word_i, word_j) <= radius
    
    # Helper to get word from index
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute distances?
    # For radius=1, dist <= 1 means dist=0 or dist=1.
    # We can compute balls more efficiently.
    
    balls = np.zeros((N, N), dtype=np.uint8)
    
    # For each word i
    for i in range(N):
        w_i = idx_to_word(i)
        # Add self
        balls[i, i] = 1
        
        # Add neighbors at distance 1
        for pos in range(n):
            original_val = w_i[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                w_j = w_i.copy()
                w_j[pos] = new_val
                # Convert w_j to index
                j = 0
                multiplier = 1
                for k in range(n):
                    j += w_j[k] * multiplier
                    multiplier *= q
                balls[i, j] = 1
                
    # 2. Greedy Initialization using NumPy
    def greedy_init():
        uncovered = np.ones(N, dtype=np.uint8)
        in_code = np.zeros(N, dtype=bool)
        code_list = []
        
        while np.any(uncovered):
            # Count how many uncovered words each candidate covers
            # cov_counts[i] = sum(uncovered[j] * balls[i, j])
            cov_counts = balls @ uncovered
            # Pick the one with max coverage
            best_idx = np.argmax(cov_counts)
            
            # If best coverage is 0, we are stuck (should not happen if balls are correct)
            if cov_counts[best_idx] == 0:
                break
                
            # Add to code
            in_code[best_idx] = True
            code_list.append(best_idx)
            
            # Update uncovered
            uncovered &= ~balls[best_idx]
            
        return in_code, code_list

    # 3. Local Search with Swaps
    # State: in_code (bool array), coverage (int array)
    # coverage[i] = number of codewords covering word i
    
    def local_search(in_code, code_list, time_limit):
        coverage = balls @ in_code.astype(np.int32)
        current_size = len(code_list)
        best_size = current_size
        best_in_code = in_code.copy()
        
        start_time = time.time()
        
        # Tabu list to avoid cycling (simple: last K moves)
        # Or just random.
        
        # We will perform many steps.
        # Move: 
        # 1. Try to remove a random codeword. If valid, accept.
        # 2. If not, try a swap: remove random c_in, add random c_out. If valid, accept.
        
        # To make it efficient, we maintain a list of indices in code.
        # But in_code is the source of truth.
        
        while time.time() - start_time < time_limit:
            if not code_list:
                break
                
            move_type = random.random()
            
            if move_type < 0.3:
                # Try Remove
                idx_in = random.choice(code_list)
                
                # Check validity: coverage - balls[idx_in] > 0 for all
                # new_coverage = coverage - balls[idx_in]
                # if np.any(new_coverage == 0): invalid
                
                # Optimization: 
                # Only words covered by idx_in can become 0.
                # Those are indices j where balls[idx_in, j] == 1.
                # We only need to check coverage[j] == 1 for those j.
                
                # Get neighbors of idx_in
                neighbors = np.where(balls[idx_in] == 1)[0]
                
                # Check if any neighbor has coverage 1
                # If coverage[neighbor] == 1, removing idx_in makes it 0.
                # So if min(coverage[neighbors]) == 1, it's invalid?
                # Wait. If coverage[j] > 1, it's fine.
                # If coverage[j] == 1, it becomes 0.
                
                # So valid iff all neighbors have coverage > 1.
                # i.e., min(coverage[neighbors]) > 1
                
                if np.min(coverage[neighbors]) > 1:
                    # Valid removal
                    in_code[idx_in] = False
                    coverage -= balls[idx_in]
                    code_list.remove(idx_in)
                    current_size -= 1
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_in_code = in_code.copy()
                        
            else:
                # Try Swap
                # Pick c_in from code_list
                # Pick c_out from NOT in_code
                
                if not code_list:
                    continue
                    
                idx_in = random.choice(code_list)
                
                # Pick random idx_out not in code
                # We can pick random indices and check
                # Or maintain a list of non-code indices?
                # N is small (2187).
                
                # To speed up, pick random idx_out
                # Retry a few times if it's in code
                idx_out = random.randint(0, N - 1)
                while in_code[idx_out]:
                    idx_out = random.randint(0, N - 1)
                
                # Check validity of swap:
                # new_coverage = coverage - balls[idx_in] + balls[idx_out]
                # Valid if new_coverage > 0 everywhere.
                
                # Words that might become 0 are those in neighbors(idx_in) 
                # that are NOT covered by idx_out (or covered by less others).
                
                # Efficient check:
                # 1. Calculate new_coverage for neighbors of idx_in.
                # 2. Also, idx_out adds coverage to its neighbors.
                
                # Let's just calculate the full vector difference for the critical parts.
                # Critical words: neighbors of idx_in.
                # new_cov = coverage - balls[idx_in]
                # Then add balls[idx_out].
                
                # We only care if new_cov + balls[idx_out] > 0.
                
                # Let's compute delta = balls[idx_out] - balls[idx_in]
                # new_coverage = coverage + delta
                # We need np.all(new_coverage > 0)
                
                # Optimization:
                # Only words where delta < 0 can potentially become 0.
                # delta < 0 means balls[idx_in]=1 and balls[idx_out]=0.
                # These are neighbors of idx_in that are NOT neighbors of idx_out.
                
                # Let's find indices where balls[idx_in] == 1 and balls[idx_out] == 0.
                # mask = balls[idx_in] & ~balls[idx_out]
                # critical_indices = np.where(mask)[0]
                
                # For these indices, new_coverage = coverage - 1.
                # So we need coverage[critical_indices] > 1.
                
                # What about words where balls[idx_in]=0 and balls[idx_out]=1?
                # coverage increases, so no risk of 0.
                
                # What about words where both are 1 or both are 0?
                # No change.
                
                # So the condition is:
                # For all j in (neighbors_in \ neighbors_out), coverage[j] > 1.
                
                mask_risky = balls[idx_in] & ~balls[idx_out]
                if np.any(mask_risky):
                    risky_indices = np.where(mask_risky)[0]
                    if np.min(coverage[risky_indices]) > 1:
                        # Valid swap
                        in_code[idx_in] = False
                        in_code[idx_out] = True
                        coverage += balls[idx_out] - balls[idx_in]
                        code_list.remove(idx_in)
                        code_list.append(idx_out)
                        # Size unchanged
                else:
                    # No risky words, swap is always valid?
                    # Yes, if no word loses coverage, it stays >= 0.
                    # Since it was >= 1 before, it stays >= 1.
                    in_code[idx_in] = False
                    in_code[idx_out] = True
                    coverage += balls[idx_out] - balls[idx_in]
                    code_list.remove(idx_in)
                    code_list.append(idx_out)
                    
        return best_in_code

    # Main Loop
    import time
    total_start = time.time()
    total_limit = 110.0
    
    best_code_global = None
    best_size_global = float('inf')
    
    while time.time() - total_start < total_limit:
        # Greedy Init
        in_code, code_list = greedy_init()
        
        # Local Search
        # Allocate remaining time
        remaining = total_limit - (time.time() - total_start)
        # Leave 10s buffer
        ls_time = max(1.0, remaining - 10.0)
        
        best_in_code = local_search(in_code, code_list, ls_time)
        
        size = np.sum(best_in_code)
        if size < best_size_global:
            best_size_global = size
            best_code_global = best_in_code
            
    # Convert result to list of lists
    if best_code_global is None:
        # Fallback
        return _fallback_greedy(q, n, radius)
        
    result = []
    for i in range(N):
        if best_code_global[i]:
            w = idx_to_word(i)
            result.append(w)
            
    return result

def _fallback_greedy(q, n, radius):
    # Simple fallback if N is too large
    import itertools
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    while uncovered:
        best_w = None
        max_cov = 0
        for w in all_words:
            cov = 0
            for u in uncovered:
                # Check dist
                d = sum(1 for a, b in zip(w, u) if a != b)
                if d <= radius:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_w = w
        if best_w is None:
            break
        code.append(best_w)
        new_uncovered = set()
        for u in uncovered:
            d = sum(1 for a, b in zip(best_w, u) if a != b)
            if d > radius:
                new_uncovered.add(u)
        uncovered = new_uncovered
    return code
