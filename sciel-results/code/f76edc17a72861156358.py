"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code for {0,1,2}^n with covering radius 1.
    Uses a constraint satisfaction approach with critical word heuristics.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n <= 4:
            if n == 1:
                return [[i] for i in range(q)]
            elif n == 2:
                return [list(w) for w in itertools.product(range(q), repeat=2)]
            elif n == 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return tetracode
            else:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Initialize with the 486-word direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # Precompute all words in the space and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = q ** n
    
    # Precompute neighbors (distance <= 1) for each word index
    # neighbors[i] is a list of indices j such that d(all_words[i], all_words[j]) <= 1
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # distance 0
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    new_w = tuple(new_w)
                    neighbors[i].append(word_to_idx[new_w])
    
    # Initial state
    codewords = set(initial_codewords)
    # coverage_count[j] is the number of codewords covering word j
    coverage_count = [0] * N
    
    # Initialize coverage counts
    for c in codewords:
        ci = word_to_idx[c]
        for nb in neighbors[ci]:
            coverage_count[nb] += 1
            
    def try_remove(c):
        """Try to remove codeword c. Returns True if successful."""
        ci = word_to_idx[c]
        # Check if all neighbors of c are covered by at least 2 codewords
        for nb in neighbors[ci]:
            if coverage_count[nb] <= 1:
                return False
        
        # Perform removal
        for nb in neighbors[ci]:
            coverage_count[nb] -= 1
        codewords.remove(c)
        return True

    def try_swap(c_old, c_new):
        """
        Try to replace c_old with c_new.
        Returns True if successful.
        """
        if c_new in codewords:
            return False
        if c_old not in codewords:
            return False
            
        ci_old = word_to_idx[c_old]
        ci_new = word_to_idx[c_new]
        
        # Check validity:
        # 1. All words covered by c_old must still be covered after removal.
        #    A word w is covered by c_old if w in neighbors[ci_old].
        #    After removing c_old, w must be covered by c_new (w in neighbors[ci_new])
        #    OR w must have coverage_count[w] >= 2 (so other codewords cover it).
        
        # Optimization: Only check words in neighbors[ci_old].
        # If w is in neighbors[ci_old]:
        #   If w is in neighbors[ci_new], it will be covered by c_new.
        #   Else, it must have coverage_count[w] >= 2.
        
        # We can precompute the set of neighbors for c_new for faster lookup?
        # N is 6561, neighbors size is 1 + 8*2 = 17.
        # Checking membership in a list of 17 is fast.
        
        set_new_neighbors = set(neighbors[ci_new])
        
        for w in neighbors[ci_old]:
            if w not in set_new_neighbors:
                if coverage_count[w] <= 1:
                    return False
        
        # If valid, perform swap
        # Remove c_old
        for w in neighbors[ci_old]:
            coverage_count[w] -= 1
        codewords.remove(c_old)
        
        # Add c_new
        for w in neighbors[ci_new]:
            coverage_count[w] += 1
        codewords.add(c_new)
        
        return True

    # Phase 1: Greedy Removal
    # Repeat until no more removals are possible
    while True:
        removed_any = False
        # Shuffle to avoid bias
        cw_list = list(codewords)
        random.shuffle(cw_list)
        for c in cw_list:
            if c in codewords and try_remove(c):
                removed_any = True
                # Restart scanning from beginning as coverage counts changed
                # Or just continue? Restarting is safer for finding new removals.
                # But it's expensive. Let's just continue in this pass, then loop again.
                pass
        if not removed_any:
            break

    # Phase 2: Critical Word Swaps
    # Identify critical words (coverage count 1)
    # For each critical word w, let c_old be the unique codeword covering it.
    # We need to find c_new such that c_new covers w, and the swap is valid.
    
    # To make this efficient, we can iterate through critical words.
    # However, finding c_old for a critical word requires knowing which codeword covers it.
    # We can maintain a list of codewords covering each word? That's expensive to update.
    # Alternative: For each codeword c, find the critical words it covers.
    # If c covers a critical word w, then c is the *only* codeword covering w.
    
    # Let's iterate through codewords and try to swap them.
    # A codeword c is a candidate for swap if it covers at least one critical word.
    
    max_swap_iterations = 200
    for _ in range(max_swap_iterations):
        progress = False
        
        # Identify critical words and the codeword that covers them
        # critical_word_to_codeword: dict {w_idx: c_tuple}
        critical_map = {}
        
        # We need to know which codeword covers a critical word.
        # Since it's unique, we can just check:
        # For each codeword c, for each w in neighbors[c]:
        #   if coverage_count[w] == 1:
        #      critical_map[w] = c
        
        # This is O(|C| * 17). |C| ~ 400. 400*17 = 6800 ops. Very fast.
        for c in codewords:
            ci = word_to_idx[c]
            for w in neighbors[ci]:
                if coverage_count[w] == 1:
                    critical_map[w] = c
        
        if not critical_map:
            break
            
        # For each codeword involved in critical words, try to find a replacement
        # We collect candidate codewords to swap
        candidates_to_swap = set(critical_map.values())
        
        # Shuffle candidates
        cand_list = list(candidates_to_swap)
        random.shuffle(cand_list)
        
        for c_old in cand_list:
            if c_old not in codewords:
                continue
                
            # Find critical words covered by c_old
            ci_old = word_to_idx[c_old]
            critical_words_for_c = [w for w in neighbors[ci_old] if coverage_count[w] == 1]
            
            if not critical_words_for_c:
                continue
            
            # We want to find a c_new such that:
            # 1. c_new covers ALL critical_words_for_c (ideally, or at least one? 
            #    Actually, if we swap c_old with c_new, c_old is removed.
            #    So c_new MUST cover all critical words that c_old covered, otherwise those words become uncovered.
            #    Wait, if w is critical and covered by c_old, and we remove c_old, w must be covered by c_new.
            #    So c_new must be within distance 1 of ALL critical words covered by c_old.
            
            # Let's check if there exists a c_new that covers all critical_words_for_c.
            # The intersection of the neighborhoods of these critical words.
            # If |critical_words_for_c| is large, the intersection might be small or empty.
            
            # Optimization: If there are multiple critical words, find their common neighbors.
            if len(critical_words_for_c) > 1:
                # Intersect neighbors
                common_neighbors = set(neighbors[critical_words_for_c[0]])
                for w in critical_words_for_c[1:]:
                    common_neighbors.intersection_update(neighbors[w])
                    if not common_neighbors:
                        break
            else:
                common_neighbors = set(neighbors[critical_words_for_c[0]])
                
            if not common_neighbors:
                continue
                
            # Filter common_neighbors to find valid c_new
            # c_new must not be in codewords
            # c_new must satisfy the swap validity condition
            
            # To speed up, we can just iterate through common_neighbors
            # and check validity.
            
            # Shuffle common neighbors
            cn_list = list(common_neighbors)
            random.shuffle(cn_list)
            
            for w_new_idx in cn_list:
                c_new = all_words[w_new_idx]
                if c_new in codewords:
                    continue
                
                if try_swap(c_old, c_new):
                    progress = True
                    break
            
            if progress:
                # After a swap, coverage counts changed.
                # We should restart the critical map calculation or just break out of this codeword loop
                # and let the outer loop restart.
                break
        
        if not progress:
            # No swaps made in this iteration
            # Try removals again
            removed_any = False
            cw_list = list(codewords)
            random.shuffle(cw_list)
            for c in cw_list:
                if c in codewords and try_remove(c):
                    removed_any = True
            if not removed_any:
                break

    # Final check
    if any(coverage_count[w] == 0 for w in range(N)):
        # Fallback
        return [list(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]

    return [list(c) for c in codewords]

# EVOLVE-BLOCK-END
