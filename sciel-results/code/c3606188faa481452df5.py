import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    
    # 2. Precompute balls
    # Ball[i] contains indices of words within distance 'radius' of all_words[i]
    # For radius 1, we can generate neighbors explicitly.
    balls = []
    for i in range(N):
        w = all_words[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                # Find index of new_w
                # Since N is small (2187), a linear search or a dict is fine.
                # But we can be smarter.
                # Let's use a dictionary mapping tuple to index for fast lookup.
                pass
        # This loop is too slow for precomputation if done naively.
        # Better: compute distances or use the fact that radius is 1.
        
    # Let's use a dictionary for word->index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    balls = []
    for i in range(N):
        w = all_words[i]
        neighbors = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors.append(idx)
        balls.append(neighbors)
        
    # balls is a list of lists of indices.
    # Convert to np.array for faster ops? Or keep as lists?
    # We need to check coverage.
    
    # State:
    # counts: array of length N, counts[j] = number of codewords that cover word j
    # in_code: boolean array of length N, True if word j is a codeword
    
    # Initial state: All words are codewords.
    # This is a valid cover.
    in_code = np.ones(N, dtype=bool)
    # counts[j] is the number of codewords in balls[j]? No.
    # counts[j] = number of codewords c such that j in Ball(c).
    # This is the same as: number of c such that c in Ball(j) (symmetry).
    
    # Precompute Ball(j) as the list of codewords that cover j?
    # Actually, Ball(j) is the set of words within dist 1 of j.
    # So counts[j] = sum(1 for c in range(N) if in_code[c] and c in Ball(j))
    # Note: Ball(j) is the same set of neighbors as Ball(c) logic, just reversed.
    # Since distance is symmetric, if c is in Ball(j), j is in Ball(c).
    # So we can just use the precomputed `balls` array.
    # balls[i] contains indices of words within dist 1 of all_words[i].
    # So counts[j] = sum(in_code[c] for c in balls[j])
    
    # Initialize counts
    counts = np.zeros(N, dtype=np.int32)
    for j in range(N):
        counts[j] = np.sum(in_code[balls[j]])
        
    # Helper to update counts when adding/removing a codeword
    def add_codeword(c_idx):
        if in_code[c_idx]:
            return
        in_code[c_idx] = True
        for j in balls[c_idx]:
            counts[j] += 1
            
    def remove_codeword(c_idx):
        if not in_code[c_idx]:
            return
        in_code[c_idx] = False
        for j in balls[c_idx]:
            counts[j] -= 1
            
    def is_redundant(c_idx):
        # c_idx is redundant if for all j in balls[c_idx], counts[j] > 1
        # i.e., min(counts[balls[c_idx]]) > 1
        return np.min(counts[balls[c_idx]]) > 1
        
    # Phase 1: Greedy Deletion
    # Iterate through codewords and remove if redundant.
    # Repeat until no changes.
    # To speed up, we can just do passes.
    
    changed = True
    while changed:
        changed = False
        for c in range(N):
            if in_code[c] and is_redundant(c):
                remove_codeword(c)
                changed = True
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
            
    # Phase 2: Local Search (Swap and Delete)
    # We want to find swaps that allow further deletions.
    # A swap replaces c with w.
    # Valid if U_c (words covered only by c) are covered by w.
    # U_c = { j in balls[c] | counts[j] == 1 }
    
    # To make this efficient, we can randomly sample swaps or iterate.
    # Given the small N, iterating is possible but might be slow if done naively.
    # Let's try a few passes of "Best Swap" or "First Valid Swap".
    
    # We will perform a number of iterations.
    # In each iteration, we try to improve the code.
    # Improvement metric: Number of redundant codewords (potential deletions).
    
    max_iters = 5000
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        
        # 1. Try Deletions
        for c in range(N):
            if in_code[c] and is_redundant(c):
                remove_codeword(c)
                improved = True
                # Don't break, try to remove more if possible? 
                # Or break to restart search? 
                # Removing one might make another redundant.
                # Let's just remove and continue.
        
        # 2. Try Swaps
        # For each codeword c, find a word w not in code such that U_c subset Ball(w).
        # We prioritize swaps that increase the number of "critical" words covered by w?
        # Or just find any valid swap.
        
        # To avoid O(N^2) or worse, we can be smart.
        # But N=2187 is small.
        
        for c in range(N):
            if not in_code[c]:
                continue
                
            # Find U_c
            u_c = [j for j in balls[c] if counts[j] == 1]
            
            if not u_c:
                # Should have been deleted in step 1
                continue
                
            # We need w such that all j in u_c are in balls[w].
            # i.e., w is in intersection of balls[j] for j in u_c?
            # No, w must be such that j is in balls[w].
            # This is equivalent to w in balls[j] (symmetry).
            # So w must be in intersection of sets balls[j] for all j in u_c.
            
            # Since u_c is small (usually 1), this is fast.
            candidates = None
            for j in u_c:
                if candidates is None:
                    candidates = set(balls[j])
                else:
                    candidates.intersection_update(balls[j])
                    
            if not candidates:
                continue
                
            # Remove current codewords from candidates
            # We want w not in code.
            # Also w != c (though c is in code, so it's removed if we subtract code)
            
            valid_candidates = [w for w in candidates if not in_code[w]]
            
            if not valid_candidates:
                continue
                
            # Pick a candidate. 
            # Heuristic: Pick w that covers the most words with count 1 (excluding those in U_c? No, U_c must be covered).
            # Actually, we want to pick a w that makes the code "denser" or creates new redundancies.
            # A good proxy: Maximize sum(1 for k in balls[w] if counts[k] == 1)
            # This increases the number of words that *could* be covered by multiple codewords?
            # No, if counts[k] is 1, it's critical. We want to *increase* counts of critical words?
            # If we add w, counts[k] becomes 2. Then k is no longer critical.
            # This makes the code more robust, potentially allowing deletion of other codewords.
            
            best_w = -1
            best_score = -1
            for w in valid_candidates:
                # Score: number of critical words in balls[w]
                # We don't need to subtract U_c because they are already covered by c (count 1) and will be covered by w (count becomes 2).
                # The "gain" is making other critical words non-critical.
                score = np.sum(counts[balls[w]] == 1)
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            if best_w != -1:
                # Perform swap
                remove_codeword(c)
                add_codeword(best_w)
                improved = True
                # Break out of swap loop? Or continue?
                # Let's break to re-evaluate redundancy.
                break
                
        if not improved:
            # No improvement found in this pass. 
            # We could try randomized restarts or perturbations, but simple local search often works well for this size.
            # If stuck, we might need to accept the current size.
            # Given the time limit, we just stop if no improvement.
            break
            
    # Extract codewords
    result = [all_words[i].tolist() for i in range(N) if in_code[i]]
    
    return result
