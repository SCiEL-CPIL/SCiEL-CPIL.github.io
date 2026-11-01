"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized for q=3, n=6, radius=1 using incremental coverage tracking and local search.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _generic_greedy(q, n, radius)

    # Specific optimization for q=3, n=6, radius=1
    all_words = list(itertools.product(range(3), repeat=n))
    total_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (distance <= 1)
    # neighbors[i] is a list of indices
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(3):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neighbors[i].append(word_to_idx[tuple(new_w)])

    # Start with the tetracode-based construction (9 * 3^2 = 81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            initial_code_indices.append(word_to_idx[w])
    
    # Use a list for code to allow fast removal? Set is better for membership.
    # But we need to iterate.
    code_set = set(initial_code_indices)
    
    # coverage_count[i] = number of codewords covering word i
    coverage_count = [0] * total_words
    
    def add_code(c):
        for nb in neighbors[c]:
            coverage_count[nb] += 1
            
    def remove_code(c):
        for nb in neighbors[c]:
            coverage_count[nb] -= 1

    # Initialize coverage counts
    for c in code_set:
        add_code(c)

    # Check if initial code is valid (should be)
    # assert all(count > 0 for count in coverage_count)

    # Local Search Phase 1: Removal
    # Iterate until no more removals possible
    improved = True
    while improved:
        improved = False
        # Shuffle to avoid bias? Or just iterate.
        for c in list(code_set):
            # Check if removing c leaves any word with count 0
            # Only words in neighbors[c] are affected
            valid = True
            for nb in neighbors[c]:
                if coverage_count[nb] == 1:
                    valid = False
                    break
            
            if valid:
                remove_code(c)
                code_set.remove(c)
                improved = True
                break # Restart loop to check other words

    # Local Search Phase 2: Swapping
    # Try to swap a codeword with another to maintain coverage
    # We can also try to reduce the "slack" to enable further removals, but simple coverage check is enough.
    improved = True
    while improved:
        improved = False
        code_list = list(code_set)
        # Shuffle to break symmetry?
        random.shuffle(code_list)
        
        for c_remove in code_list:
            # Temporarily remove c_remove
            remove_code(c_remove)
            
            # Identify words that are now uncovered (count 0)
            # Only in neighbors[c_remove]
            uncovered_by_remove = [nb for nb in neighbors[c_remove] if coverage_count[nb] == 0]
            
            if not uncovered_by_remove:
                # c_remove was redundant? But we already did removal phase.
                # This might happen if the removal phase missed something due to order?
                # Or if we are in a state where removal isn't possible but swap is?
                # Actually, if no words become uncovered, we can just remove it.
                code_set.remove(c_remove)
                improved = True
                break
            
            # We need to find a c_add that covers all words in uncovered_by_remove
            # And c_add must not be in code_set
            # To speed up, we can filter candidates.
            # Any c_add must be within distance 1 of all words in uncovered_by_remove.
            # This is a strong constraint.
            
            found_swap = False
            # Iterate through all possible c_add
            # Optimization: Only consider c_add that are "close" to the uncovered set?
            # For now, iterate all 729.
            for c_add in range(total_words):
                if c_add in code_set:
                    continue
                
                # Check if c_add covers all words in uncovered_by_remove
                # i.e., all words in uncovered_by_remove must be in neighbors[c_add]
                # Since neighbors is a list, we can use a set for c_add's neighbors for faster lookup?
                # Precomputed neighbors are lists. Let's make a set version for lookup if needed.
                # But 729 is small.
                # Check: for all u in uncovered_by_remove, u in neighbors[c_add]
                # Since neighbors[c_add] is a list of ~19 items, and uncovered_by_remove is small,
                # we can just check.
                
                # Faster: Check if set(uncovered_by_remove).issubset(set(neighbors[c_add]))
                # Creating sets is expensive.
                # Alternative: 
                # The intersection of neighbors[c_remove] and neighbors[c_add] must contain uncovered_by_remove.
                
                valid_add = True
                # Optimization: If c_add is far from any u in uncovered_by_remove, it's invalid.
                # Hamming distance check?
                # We can precompute a bitmask or just check membership.
                # Let's use a temporary set for neighbors[c_add] if the list is long, but it's short.
                nb_add_set = set(neighbors[c_add])
                
                for u in uncovered_by_remove:
                    if u not in nb_add_set:
                        valid_add = False
                        break
                
                if valid_add:
                    # Valid swap!
                    # Add c_add
                    add_code(c_add)
                    code_set.add(c_add)
                    code_set.remove(c_remove) # c_remove was already removed from coverage, just remove from set
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                break
            
            # If no swap found, restore c_remove
            add_code(c_remove)

    return [list(all_words[c]) for c in code_set]

def _generic_greedy(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total_words = len(all_words)
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        # Generate all words within Hamming distance 'radius'
        # For small n, radius, we can iterate all words
        for j, v in enumerate(all_words):
            if i == j: continue
            dist = sum(1 for k in range(n) if w[k] != v[k])
            if dist <= radius:
                neighbors[i].append(j)

    uncovered = set(range(total_words))
    code_indices = []
    
    while uncovered:
        best_word = -1
        max_covered = -1
        for w_idx in range(total_words):
            if w_idx in code_indices: # O(N) check, slow for large N, but OK for fallback
                continue
            # Count how many uncovered words this word covers
            covered_count = 0
            for nb in neighbors[w_idx]:
                if nb in uncovered:
                    covered_count += 1
            
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w_idx
        
        if best_word == -1:
            break
            
        code_indices.append(best_word)
        # Remove covered words from uncovered
        for nb in neighbors[best_word]:
            uncovered.discard(nb)
            
    return [list(all_words[c]) for c in code_indices]

# EVOLVE-BLOCK-END
