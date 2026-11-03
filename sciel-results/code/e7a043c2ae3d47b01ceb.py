import random
import time
import numpy as np

def construct(q, n, radius):
    # For the specific task q=3, n=7, radius=1
    # General implementation for small q, n
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        all_words.append(tuple(w))
        
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage bitmasks
    # Each mask is an integer where bit j is 1 if word j is covered by word i
    coverage_masks = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                j = word_to_idx[tuple(nw)]
                mask |= (1 << j)
        coverage_masks[i] = mask
        
    # Helper to check if a set of indices covers everything
    # universe_mask has bits 0 to total_words-1 set
    universe_mask = (1 << total_words) - 1
    
    def is_covered(indices):
        cover_mask = 0
        for i in indices:
            cover_mask |= coverage_masks[i]
        return (cover_mask & universe_mask) == universe_mask
        
    def get_uncovered_mask(cover_mask):
        return (~cover_mask) & universe_mask

    best_solution = None
    best_size = total_words
    
    # We will run multiple iterations of the heuristic
    # Each iteration: Greedy Init -> Local Search
    
    # Estimate max iterations based on time
    # Greedy is O(N^2 * radius * q) roughly, Local search is variable.
    # For N=2187, one greedy pass is fast. Local search can be slower.
    
    num_iterations = 200 # We will break early if time runs out
    
    for iter_num in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Greedy Initialization
        # We maintain a bitmask of currently covered words
        covered_mask = 0
        selected_indices = []
        
        # To speed up greedy, we can pre-sort candidates or just scan.
        # Scanning 2187 items 200 times is fine.
        # Optimization: Instead of scanning all, we can pick the one that covers most UNCOVERED.
        
        # A simple optimization for greedy: 
        # At each step, find c that maximizes popcount(coverage_masks[c] & ~covered_mask)
        
        while covered_mask != universe_mask:
            best_c = -1
            best_gain = 0
            # Random tie-breaking to diversify
            tie_candidates = []
            
            # We can sample if N is large, but 2187 is small enough to scan.
            for c in range(total_words):
                # If c is already selected, skip? 
                # Actually, selecting same c again is useless, but greedy usually picks distinct ones.
                # We should check if c is in selected. 
                # Using a set for selected is O(1).
                # But we need to update covered_mask.
                
                gain_mask = coverage_masks[c] & ~covered_mask
                # Count bits set
                gain = gain_mask.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    tie_candidates = [c]
                elif gain == best_gain and gain > 0:
                    tie_candidates.append(c)
            
            if best_c == -1:
                # Should not happen if universe is not fully covered
                break
                
            chosen = random.choice(tie_candidates)
            selected_indices.append(chosen)
            covered_mask |= coverage_masks[chosen]
            
        # 2. Local Search: Reduction
        # Try to remove any codeword that is redundant
        improved = True
        while improved:
            improved = False
            # Shuffle to break symmetry? Or just iterate.
            # Iterating in reverse order or random order might help.
            for i in range(len(selected_indices) - 1, -1, -1):
                c = selected_indices[i]
                # Check if c is redundant
                # Remove c, check if still covered
                temp_mask = 0
                # Recalculating full mask is O(K). 
                # Better: precompute prefix/suffix or just use a set and recompute?
                # For K~186, recomputing is 186 ORs. Fast.
                
                # Optimization: 
                # If we remove c, is there any word w covered ONLY by c?
                # This is hard to check directly with just masks.
                # Simpler: just recompute the union of the rest.
                
                # To make this faster, we can maintain the union mask incrementally?
                # No, because removing one element requires recomputing if we don't have counts.
                
                # Let's just recompute. 186 * 2187 bits is small.
                new_covered = 0
                for idx in selected_indices:
                    if idx == c:
                        continue
                    new_covered |= coverage_masks[idx]
                
                if (new_covered & universe_mask) == universe_mask:
                    # Redundant
                    selected_indices.pop(i)
                    improved = True
                    break # Restart loop as list changed
        
        current_size = len(selected_indices)
        if current_size < best_size:
            best_size = current_size
            best_solution = list(selected_indices)
            
        # 3. Local Search: Swaps
        # Try to swap a codeword with another to potentially reduce size later
        # We accept a swap if the new set is valid.
        # We don't require immediate size reduction, just validity, hoping reduction works after.
        
        max_swap_attempts = 500
        attempts = 0
        while attempts < max_swap_attempts:
            attempts += 1
            if time.time() - start_time > time_limit:
                break
                
            if not selected_indices:
                break
                
            # Pick a random codeword to remove
            i = random.randint(0, len(selected_indices) - 1)
            c = selected_indices[i]
            
            # Pick a random new word
            w = random.randint(0, total_words - 1)
            if w in selected_indices:
                continue
                
            # Check validity of swap
            # New set: selected_indices - {c} + {w}
            
            # Optimization: 
            # Let S = selected_indices.
            # Let M = union(coverage[x] for x in S).
            # We remove c. Let M_rest = union(coverage[x] for x in S\{c}).
            # New Mask = M_rest | coverage[w].
            # We need New Mask == Universe.
            
            # To avoid full recompute, we can check if M_rest covers everything except what w covers?
            # Actually, just recompute is safer and fast enough for ~200 codewords.
            
            new_covered = 0
            for idx in selected_indices:
                if idx == c:
                    continue
                new_covered |= coverage_masks[idx]
            
            new_covered |= coverage_masks[w]
            
            if (new_covered & universe_mask) == universe_mask:
                # Swap is valid
                selected_indices[i] = w
                
                # Run reduction again immediately
                improved = True
                while improved:
                    improved = False
                    for j in range(len(selected_indices) - 1, -1, -1):
                        cc = selected_indices[j]
                        # Recompute mask without cc
                        temp_covered = 0
                        for idx in selected_indices:
                            if idx == cc:
                                continue
                            temp_covered |= coverage_masks[idx]
                        
                        if (temp_covered & universe_mask) == universe_mask:
                            selected_indices.pop(j)
                            improved = True
                            break
        
        current_size = len(selected_indices)
        if current_size < best_size:
            best_size = current_size
            best_solution = list(selected_indices)

    if best_solution is None:
        # Fallback: return all words (should not happen)
        return [list(all_words[i]) for i in range(total_words)]
        
    return [list(all_words[i]) for i in best_solution]
