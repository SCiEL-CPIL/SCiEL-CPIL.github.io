import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate all words in the space
    # We can represent a word as an integer from 0 to q**n - 1
    # But for distance calculation, list/tuple is easier.
    # For speed, we can use base-q representation.
    
    # 1. Initial Construction: Direct sum of tetracode and all words of length n-4
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Generate tails
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    # Construct initial codewords
    initial_codewords = []
    for h in tetracode:
        for tail in tails:
            initial_codewords.append(h + list(tail))
            
    # Precompute neighbors for each possible word? No, space is 729.
    # Let's map word -> index.
    # Index = sum(w[i] * q**(n-1-i))
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute all neighbors (distance <= radius) for each word in the space
    # Space size is q**n = 729
    space_size = q ** n
    neighbors = [[] for _ in range(space_size)]
    
    for i in range(space_size):
        w = idx_to_word(i)
        # Neighbor is w itself
        neighbors[i].append(i)
        # Neighbors are words differing in exactly 1 position
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val != original_val:
                    if radius >= 1:
                        w[pos] = new_val
                        neighbors[i].append(word_to_idx(w))
            w[pos] = original_val
            
    # Precompute neighbor sets as frozensets for faster intersection checks
    neighbor_sets = [frozenset(nb) for nb in neighbors]
    
    def greedy_delete(codewords, counts, neighbors, neighbor_sets, deadline):
        """Perform greedy deletion to reduce codeword count."""
        current = list(codewords)
        cnt = counts.copy()
        
        max_passes = 50
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            removed_any = False
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    break
                idx_c = word_to_idx(current[i])
                
                # Check if any neighbor has count == 1
                redundant = True
                for nb in neighbors[idx_c]:
                    if cnt[nb] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for nb in neighbors[idx_c]:
                        cnt[nb] -= 1
                    current.pop(i)
                    removed_any = True
                    
            if not removed_any:
                break
        
        return current, cnt

    def local_search_swap(codewords, counts, neighbors, neighbor_sets, deadline, max_swaps=5000):
        """Try to swap codewords to enable further deletions."""
        current = list(codewords)
        cnt = counts.copy()
        current_set = set(word_to_idx(c) for c in current)
        
        # Build a map from index to position in current list
        idx_to_pos = {word_to_idx(c): i for i, c in enumerate(current)}
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try each codeword as a candidate for removal
            for i in range(len(current)):
                if time.time() > deadline:
                    return current, cnt
                
                idx_c = word_to_idx(current[i])
                
                # Check if this codeword can be removed
                can_remove = True
                for nb in neighbors[idx_c]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    for nb in neighbors[idx_c]:
                        cnt[nb] -= 1
                    removed_idx = current.pop(i)
                    del idx_to_pos[word_to_idx(removed_idx)]
                    # Rebuild idx_to_pos after removal
                    idx_to_pos = {word_to_idx(c): j for j, c in enumerate(current)}
                    improved = True
                    break
            
            if improved:
                continue
            
            # No simple removal possible, try swaps
            # For each codeword, try replacing it with a random word
            # that covers its "essential" neighbors
            for i in range(len(current)):
                if time.time() > deadline:
                    return current, cnt
                
                idx_c = word_to_idx(current[i])
                
                # Find which neighbors are only covered by this codeword (count == 1)
                essential = [nb for nb in neighbors[idx_c] if cnt[nb] == 1]
                
                if not essential:
                    # This codeword can be removed directly
                    for nb in neighbors[idx_c]:
                        cnt[nb] -= 1
                    current.pop(i)
                    idx_to_pos = {word_to_idx(c): j for j, c in enumerate(current)}
                    improved = True
                    break
                
                # Try to find a replacement that covers all essential neighbors
                # and doesn't introduce new uncovered words
                # Sample some candidate replacements
                candidates_tried = 0
                max_candidates = 20
                
                for _ in range(max_candidates):
                    if time.time() > deadline:
                        return current, cnt
                    
                    # Random candidate word
                    cand_idx = random.randint(0, space_size - 1)
                    if cand_idx in idx_to_pos:
                        continue
                    
                    cand_nb = neighbors[cand_idx]
                    
                    # Check if candidate covers all essential neighbors
                    covers_essential = all(nb in cand_nb for nb in essential)
                    if not covers_essential:
                        continue
                    
                    # Check if removing current and adding candidate maintains coverage
                    # Words that lose coverage: those in neighbors[idx_c] but not in cand_nb with count==1
                    # Words that gain coverage: those in cand_nb but not in neighbors[idx_c]
                    
                    # Simpler: temporarily remove current, add candidate, check all counts >= 1
                    # Then undo if invalid
                    
                    # First, remove current's contribution
                    for nb in neighbors[idx_c]:
                        cnt[nb] -= 1
                    
                    # Add candidate's contribution
                    for nb in cand_nb:
                        cnt[nb] += 1
                    
                    # Check if any word has count 0
                    valid = True
                    for nb in neighbors[idx_c]:
                        if cnt[nb] == 0:
                            valid = False
                            break
                    
                    if valid:
                        # Accept the swap
                        current[i] = idx_to_word(cand_idx)
                        idx_to_pos = {word_to_idx(c): j for j, c in enumerate(current)}
                        improved = True
                        break
                    else:
                        # Undo
                        for nb in cand_nb:
                            cnt[nb] -= 1
                        for nb in neighbors[idx_c]:
                            cnt[nb] += 1
                        continue
                
                if improved:
                    break
        
        return current, cnt

    def full_optimize(codewords, deadline):
        """Run greedy deletion followed by local search."""
        # Compute initial counts
        counts = np.zeros(space_size, dtype=np.int32)
        for c in codewords:
            idx_c = word_to_idx(c)
            for nb in neighbors[idx_c]:
                counts[nb] += 1
        
        # Greedy deletion
        codewords, counts = greedy_delete(codewords, counts, neighbors, neighbor_sets, deadline)
        
        # Local search with swaps
        codewords, counts = local_search_swap(codewords, counts, neighbors, neighbor_sets, deadline)
        
        # Another round of greedy deletion after swaps
        codewords, counts = greedy_delete(codewords, counts, neighbors, neighbor_sets, deadline)
        
        return codewords, counts

    # Run with different random seeds to explore different local optima
    best_codewords = None
    best_counts = None
    best_size = float('inf')
    
    # Multiple restarts with different random seeds
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        seed = 42 + restart * 1000
        random.seed(seed)
        
        # Use the initial construction as starting point
        # Shuffle the order to get different deletion paths
        shuffled = list(initial_codewords)
        random.shuffle(shuffled)
        
        codewords, counts = full_optimize(shuffled, deadline)
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords
            best_counts = counts
    
    # If no restart completed, use the initial result
    if best_codewords is None:
        counts = np.zeros(space_size, dtype=np.int32)
        for c in initial_codewords:
            idx_c = word_to_idx(c)
            for nb in neighbors[idx_c]:
                counts[nb] += 1
        best_codewords, best_counts = greedy_delete(initial_codewords, counts, neighbors, neighbor_sets, deadline)
    
    return best_codewords
