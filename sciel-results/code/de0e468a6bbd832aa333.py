import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    N = q ** n  # 2187 for q=3, n=7
    
    # Generate all words as tuples
    words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        words.append(tuple(w))
    
    # Word to index mapping
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Compute neighbors (ball of radius 1) for each word
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        w_i = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbors[i] = neigh
    
    # Precompute neighbor arrays as numpy for fast operations
    # Build incidence matrix: rows = codewords, cols = words covered
    # Using sparse-like approach with numpy
    # For each codeword c, neighbors[c] gives the words it covers
    # We'll build a binary matrix where M[c, u] = 1 if u in neighbors[c]
    # But N=2187, so full matrix is 2187*2187 ~ 4.8M entries, manageable
    
    # Actually, let's use a different approach:
    # For greedy, we need to count how many uncovered words each candidate covers
    # We can maintain an array uncovered_flag of length N
    # And for each candidate c, count = sum(uncovered_flag[u] for u in neighbors[c])
    # With numpy, we can vectorize this if we precompute the neighbor indices
    
    # Build neighbor index matrix (sparse representation)
    # For each c, store the list of neighbor indices
    # We already have neighbors[c] as a list
    
    # Better: use numpy to create a 2D array where each row is padded to max degree
    max_degree = max(len(neigh) for neigh in neighbors)
    # Actually all have same degree: 1 + n*(q-1) = 1 + 7*2 = 15
    degree = 1 + n * (q - 1)  # 15 for q=3, n=7
    
    # Create neighbor matrix
    neighbor_matrix = np.zeros((N, degree), dtype=np.int32)
    for i in range(N):
        neighbor_matrix[i, :len(neighbors[i])] = neighbors[i]
    
    best_code = None
    best_size = float('inf')
    
    def greedy_cover(uncovered_set, code):
        """Greedy set cover. uncovered_set is a set of word indices not yet covered.
        Returns the code list (modified in place) and uncovered_set (modified in place)."""
        uncovered = uncovered_set
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find candidate that covers most uncovered words
            # Use numpy for speed: for each candidate c, count how many of its neighbors are in uncovered
            # Convert uncovered to a boolean array
            uncovered_arr = np.zeros(N, dtype=np.uint8)
            for u in uncovered:
                uncovered_arr[u] = 1
            
            # For each candidate c, coverage = sum(uncovered_arr[neighbor_matrix[c, :]])
            # This can be vectorized: 
            # coverage = uncovered_arr[neighbor_matrix].sum(axis=1)
            # But neighbor_matrix has shape (N, degree), so uncovered_arr[neighbor_matrix] gives (N, degree)
            # Then .sum(axis=1) gives (N,)
            
            coverage = uncovered_arr[neighbor_matrix].sum(axis=1)
            
            # Find max coverage
            max_cov = coverage.max()
            if max_cov == 0:
                break
            
            # Get all candidates with max coverage
            best_candidates = np.where(coverage == max_cov)[0]
            
            # Randomly pick one for diversity
            chosen = int(random.choice(best_candidates))
            code.append(chosen)
            
            # Update uncovered
            for u in neighbors[chosen]:
                if u in uncovered:
                    uncovered.discard(u)
        
        return code, uncovered
    
    def prune_redundant(code):
        """Remove redundant codewords from code. Returns pruned code."""
        if not code:
            return code
        
        # Build cover_count: for each word u, how many codewords cover it
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            for u in neighbors[c]:
                cover_count[u] += 1
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_code = []
            for c in code:
                # Check if c is redundant: all words in B(c) are covered by at least one other codeword
                # i.e., for all u in neighbors[c], cover_count[u] > 1
                redundant = True
                for u in neighbors[c]:
                    if cover_count[u] <= 1:
                        redundant = False
                        break
                if not redundant:
                    new_code.append(c)
                else:
                    # Remove c from cover_count
                    for u in neighbors[c]:
                        cover_count[u] -= 1
                    changed = True
            code = new_code
        
        return code
    
    def kick_and_repair(code, kick_size=3):
        """Perform kick-and-repair local search.
        Remove kick_size random codewords, then greedily re-add to cover gaps."""
        if len(code) <= kick_size:
            return code
        
        # Make a copy
        code = list(code)
        
        # Remove kick_size random codewords
        remove_indices = random.sample(range(len(code)), kick_size)
        remove_set = set(remove_indices)
        removed = [code[i] for i in remove_indices]
        remaining = [code[i] for i in range(len(code)) if i not in remove_set]
        
        # Find uncovered words
        covered = np.zeros(N, dtype=np.uint8)
        for c in remaining:
            for u in neighbors[c]:
                covered[u] = 1
        
        uncovered = set(np.where(covered == 0)[0].tolist())
        
        if not uncovered:
            return remaining
        
        # Greedily add codewords to cover uncovered
        for _ in range(len(removed) + 5):  # Add up to kick_size+5 codewords
            if not uncovered:
                break
            if time.time() > deadline:
                break
            
            uncovered_arr = np.zeros(N, dtype=np.uint8)
            for u in uncovered:
                uncovered_arr[u] = 1
            
            coverage = uncovered_arr[neighbor_matrix].sum(axis=1)
            max_cov = coverage.max()
            if max_cov == 0:
                break
            
            best_candidates = np.where(coverage == max_cov)[0]
            chosen = int(random.choice(best_candidates))
            remaining.append(chosen)
            
            for u in neighbors[chosen]:
                if u in uncovered:
                    uncovered.discard(u)
        
        # Prune the result
        pruned = prune_redundant(remaining)
        return pruned
    
    # Main loop: multiple restarts of greedy + prune + kick-and-repair
    num_restarts = 50
    for iteration in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomize the order of candidates to get different greedy solutions
        # We do this by adding random noise to the coverage counts
        
        uncovered = set(range(N))
        code = []
        
        # Run greedy cover with some randomization
        # Instead of pure greedy, we can use a randomized greedy
        # For each step, with some probability, pick from top-k candidates instead of just the best
        top_k = random.choice([1, 2, 3, 5])
        
        while uncovered:
            if time.time() > deadline:
                break
            
            uncovered_arr = np.zeros(N, dtype=np.uint8)
            for u in uncovered:
                uncovered_arr[u] = 1
            
            coverage = uncovered_arr[neighbor_matrix].sum(axis=1)
            max_cov = coverage.max()
            if max_cov == 0:
                break
            
            # Get candidates with coverage close to max
            threshold = max_cov - random.randint(0, min(2, max_cov - 1))
            candidates = np.where(coverage >= threshold)[0]
            
            chosen = int(random.choice(candidates))
            code.append(chosen)
            
            for u in neighbors[chosen]:
                if u in uncovered:
                    uncovered.discard(u)
        
        if time.time() > deadline:
            break
        
        # Prune
        code = prune_redundant(code)
        
        # Kick-and-repair
        if time.time() < deadline - 5:
            for _ in range(3):  # 3 kick-and-repair iterations
                if time.time() > deadline - 2:
                    break
                code = kick_and_repair(code, kick_size=random.randint(2, 5))
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            if best_size <= 186:
                break
    
    # Convert internal indices back to lists of integers
    if best_code is None:
        # Fallback: return a valid covering
        # Use a simple construction
        import itertools
        # Direct product of a (3,1)-covering code of length 3
        # Actually, let's just use all words as a fallback (valid but large)
        return [list(words[i]) for i in range(N)]
    
    return [list(words[c]) for c in best_code]
