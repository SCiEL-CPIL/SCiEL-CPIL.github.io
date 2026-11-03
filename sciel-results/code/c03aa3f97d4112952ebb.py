import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        # Map word tuple to index
        # We use indices 0..728
        
        # Precompute all words
        all_words = list(itertools.product(range(q), repeat=n))
        
        # Precompute coverage masks
        # coverage[i] is an integer bitmask where bit j is set if word j is covered by word i
        # Since total_words = 729, we need an integer with 729 bits. Python handles this natively.
        coverage = [0] * total_words
        
        # Optimization: Precompute distance-1 neighbors for each word to build masks faster
        # Or just compute diff directly. 729*729 = 531k ops, trivial.
        
        # Let's build masks
        for i in range(total_words):
            w_i = all_words[i]
            mask = 0
            # Cover self
            mask |= (1 << i)
            # Cover neighbors at distance 1
            for pos in range(n):
                for val in range(q):
                    if val == w_i[pos]:
                        continue
                    # Create new word
                    new_word = list(w_i)
                    new_word[pos] = val
                    # Find index of new_word
                    # To make this fast, we can precompute a dict from tuple to index
                    # But 729 is small.
                    # Let's precompute index_map
                    pass
        
        # Better: Precompute index map
        index_map = {w: i for i, w in enumerate(all_words)}
        
        for i in range(total_words):
            w_i = all_words[i]
            mask = 1 << i
            for pos in range(n):
                for val in range(q):
                    if val == w_i[pos]:
                        continue
                    new_word = list(w_i)
                    new_word[pos] = val
                    j = index_map[tuple(new_word)]
                    mask |= (1 << j)
            coverage[i] = mask
        
        # Full mask
        full_mask = (1 << total_words) - 1
        
        def greedy_set_cover(uncovered_mask, code_list):
            """
            Greedily add codewords to cover uncovered_mask.
            Returns updated code_list.
            """
            uncovered = uncovered_mask
            while uncovered:
                best_idx = -1
                best_count = -1
                # Iterate all words, find one covering most uncovered bits
                # Optimization: Only consider words that cover at least one uncovered bit?
                # Actually, we can just iterate all 729.
                for i in range(total_words):
                    # Count bits in (coverage[i] & uncovered)
                    # Python's int.bit_count() is fast
                    count = (coverage[i] & uncovered).bit_count()
                    if count > best_count:
                        best_count = count
                        best_idx = i
                        if count == total_words: # Perfect
                            break
                if best_idx == -1:
                    break
                code_list.append(best_idx)
                uncovered &= ~coverage[best_idx]
            return code_list

        def local_search(code_list):
            """
            Local search: Prune and Swap.
            """
            code_set = set(code_list)
            # Calculate coverage count for each word
            # counts[j] = number of codewords covering word j
            # We can do this by iterating code_set and OR-ing? No, we need counts for pruning.
            # To prune, we need to know if a word is covered by >1 codeword.
            # We can maintain a counter array.
            
            counts = [0] * total_words
            for c in code_set:
                # Add 1 to counts for all bits set in coverage[c]
                # This is slow if done naively.
                # Alternative: Just check coverage overlap.
                pass
            
            # A faster local search strategy:
            # 1. Pruning: Try to remove each codeword. If the union of others still covers everything, remove it.
            # 2. Swapping: Try to replace one codeword with another.
            
            improved = True
            while improved:
                improved = False
                current_list = list(code_set)
                
                # Pruning
                for i, c in enumerate(current_list):
                    # Check if c is redundant
                    # Union of coverage of all other codewords
                    union_others = 0
                    for c2 in code_set:
                        if c2 == c:
                            continue
                        union_others |= coverage[c2]
                    
                    if (union_others & full_mask) == full_mask:
                        # c is redundant
                        code_set.remove(c)
                        improved = True
                        break
                
                if improved:
                    continue
                
                # Swapping / Optimization
                # Try to remove a codeword and replace it with a better one?
                # Or just try to find a smaller set?
                # Standard ILS: Perturb then Repair.
                # Here, let's try simple swaps:
                # For each c in code_set, try to replace it with w not in code_set
                # such that the new set is valid and maybe "better" (hard to define metric).
                # Actually, for minimum size, if we can't prune, we are locally optimal for size.
                # To escape local optima, we need the outer ILS loop.
                
            return code_set

        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        time_limit = 115 # Leave 5s buffer
        
        # Initial Solution
        initial_code = []
        greedy_set_cover(full_mask, initial_code)
        best_code = local_search(initial_code)
        best_size = len(best_code)
        
        # Iterated Local Search
        rng = random.Random(42)
        
        while time.time() - start_time < time_limit:
            # Perturb
            perturbed = list(best_code)
            # Remove 1 to 3 random codewords
            k_remove = rng.randint(1, 3)
            if len(perturbed) > k_remove:
                to_remove = rng.sample(perturbed, k_remove)
                for r in to_remove:
                    perturbed.remove(r)
            
            # Calculate uncovered
            covered_mask = 0
            for c in perturbed:
                covered_mask |= coverage[c]
            uncovered = full_mask & ~covered_mask
            
            # Repair (Greedy)
            new_code = list(perturbed)
            greedy_set_cover(uncovered, new_code)
            
            # Local Search
            ls_code = local_search(new_code)
            
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = ls_code
                if best_size <= 73:
                    # We are at the best known bound. Keep searching for 72?
                    # The lower bound is 71.
                    # If we hit 73, we should keep trying to see if 72 exists.
                    pass
        
        return [list(all_words[i]) for i in best_code]

    else:
        # Fallback for other parameters
        all_words = list(itertools.product(range(q), repeat=n))
        total_words = len(all_words)
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        coverage = [set() for _ in range(total_words)]
        for i, c in enumerate(all_words):
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    coverage[i].add(j)
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1: break
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        return [list(all_words[i]) for i in code_indices]
