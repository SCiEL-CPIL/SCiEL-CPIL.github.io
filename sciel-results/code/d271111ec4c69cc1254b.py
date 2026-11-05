import time
import random
import sys

def construct(q, n, radius):
    # We assume q=3, n=6, radius=1 based on the task description, 
    # but the function signature is generic. 
    # The task says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # However, the specific problem is K_3(6,1). 
    # If q, n, radius are different, the seed logic (tetracode) might not apply or be optimal.
    # The prompt says "Rewrite the program... for the following task". The task is specifically the football pool problem.
    # I will assume the evaluator calls construct(3, 6, 1).
    # If it calls with other params, I should probably fall back to a generic greedy or the seed logic if it fits.
    # But the seed logic is specific to q=3, n>=4.
    # Given the context "Football pool problem", I will optimize for q=3, n=6, radius=1.
    # I will add a check: if not (q==3 and n==6 and radius==1), use the seed logic or a simple greedy.
    
    if not (q == 3 and n == 6 and radius == 1):
        # Fallback to seed logic if possible, or simple greedy
        # For now, just return the seed result for non-target cases to be safe, 
        # or a basic greedy.
        # The seed logic:
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # This produces 9 words of length 4.
            # We need length n.
            # The seed used direct sum with tails.
            # This is a valid code for any n>=4.
            return [h + list(tail) for h in tetracode for tail in __import__('itertools').product(range(3), repeat=n - 4)]
        else:
            # Generic greedy for other cases
            # Not optimized, just to avoid crash
            all_words = list(__import__('itertools').product(range(q), repeat=n))
            code = []
            # Simple greedy set cover
            covered = set()
            for w in all_words:
                if w not in covered:
                    code.append(list(w))
                    for i in range(n):
                        for v in range(q):
                            if v == w[i]: continue
                            new_w = list(w)
                            new_w[i] = v
                            covered.add(tuple(new_w))
                    covered.add(w)
            return code

    # Target case: q=3, n=6, radius=1
    start_time = time.time()
    deadline = 110.0 # 10s safety margin
    
    # Precomputation
    # Map word tuple to index
    # 3^6 = 729
    # Index = w[0]*3^5 + w[1]*3^4 + ... + w[5]
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)

    # Precompute neighbors for each index
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    neighbors = [[] for _ in range(q**n)]
    for i in range(q**n):
        w = idx_to_word(i)
        # Self
        neighbors[i].append(i)
        # Dist 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                neighbors[i].append(word_to_idx(tuple(new_w)))
        
    # Initial code: Seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    seed_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    code_indices = set()
    counts = [0] * (q**n)
    
    for w in seed_words:
        idx = word_to_idx(tuple(w))
        code_indices.add(idx)
        for nb in neighbors[idx]:
            counts[nb] += 1
            
    # Helper to check if a word can be removed
    # A word c can be removed if for all u in neighbors[c], counts[u] > 1
    # Because removing c decreases counts[u] by 1 for u in neighbors[c].
    # If any counts[u] becomes 0, it's uncovered.
    
    def can_remove(c_idx):
        for nb in neighbors[c_idx]:
            if counts[nb] <= 1:
                return False
        return True

    def get_removability_score(c_idx):
        # Min count in neighborhood
        return min(counts[nb] for nb in neighbors[c_idx])

    best_size = len(code_indices)
    best_code = list(code_indices)
    
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        removed_any = False
        # Iterate over a copy of the set
        for c in list(code_indices):
            if can_remove(c):
                # Remove c
                for nb in neighbors[c]:
                    counts[nb] -= 1
                code_indices.remove(c)
                removed_any = True
                # Restart greedy removal loop to check new possibilities
                break
        
        if removed_any:
            if len(code_indices) < best_size:
                best_size = len(code_indices)
                best_code = list(code_indices)
            continue # Go back to top of while loop
        
        # Phase 2: Local Search (Swaps)
        # Try to find a swap that increases the maximum removability score
        # Current max score
        current_max_score = max(get_removability_score(c) for c in code_indices)
        
        # If current_max_score >= 2, we would have removed in Phase 1.
        # So current_max_score is likely 1.
        
        # We want to find a swap (c_out, c_in) such that the new state has a higher max score.
        # Or at least, a state that allows removal.
        
        # To be efficient, we don't check all pairs. We check promising ones.
        # But 80*650 is small enough to check all.
        
        best_swap = None
        best_swap_score = current_max_score
        best_swap_improvement = 0 # How much it improves the min count of the "bottleneck"
        
        # To speed up, we can just look for a swap that makes at least one word removable (score >= 2)
        # Or simply maximizes the global min count? No, that's hard.
        
        # Let's try to find a swap that results in a state where SOME word has score >= 2.
        # If we find one, we apply it, and then Phase 1 will remove that word.
        
        found_good_swap = False
        # Iterate c_out
        for c_out in list(code_indices):
            # We only care if c_out is "blocking" some removal?
            # Actually, any swap changes the counts.
            
            # To save time, we can sample c_in if the set is large, but here it's small.
            for c_in in range(q**n):
                if c_in in code_indices:
                    continue
                
                # Check if swap is valid
                # New counts = counts - neighbors[c_out] + neighbors[c_in]
                # We need new_counts[nb] >= 1 for all nb.
                
                # Optimization: 
                # The critical words are those with count 1.
                # If we remove c_out, any neighbor of c_out with count 1 becomes 0 (invalid) 
                # UNLESS c_in also covers it.
                
                # Let's just compute the new min count for the whole space? 
                # That's 729 checks. 80*650*729 is too slow.
                
                # Better check:
                # A word u is uncovered in the new state if:
                # 1. u is not in neighbors[c_in] (c_in doesn't cover it)
                # 2. AND (u is not in neighbors[c_out] OR counts[u] == 1)
                #    Wait.
                # new_count[u] = counts[u] - (1 if u in N(c_out) else 0) + (1 if u in N(c_in) else 0)
                # We need new_count[u] >= 1.
                
                # If u is in N(c_in):
                #   new_count[u] = counts[u] - (1 if u in N(c_out) else 0) + 1
                #   If u in N(c_out): new = counts[u]. (Must be >= 1, which is true)
                #   If u not in N(c_out): new = counts[u] + 1. (Must be >= 1, true)
                # So if u in N(c_in), it's always safe?
                # Yes, because counts[u] >= 1 always.
                
                # If u NOT in N(c_in):
                #   new_count[u] = counts[u] - (1 if u in N(c_out) else 0)
                #   If u not in N(c_out): new = counts[u]. (Safe)
                #   If u in N(c_out): new = counts[u] - 1.
                #       We need counts[u] - 1 >= 1 => counts[u] >= 2.
                #       So if u in N(c_out) AND u NOT in N(c_in), we require counts[u] >= 2.
                
                # So the swap is valid iff:
                # For all u in N(c_out) \ N(c_in), counts[u] >= 2.
                
                # This is a very fast check!
                # N(c_out) has 1 + 6*2 = 13 elements.
                
                valid = True
                for u in neighbors[c_out]:
                    if u not in neighbors[c_in]: # Using set lookup? 
                        # neighbors are lists. 
                        # Precompute neighbor sets for fast lookup?
                        # Or just loop through neighbors[c_in] to build a set? 
                        # 13 elements. 
                        # Let's precompute neighbor sets.
                        pass
                
                # Let's precompute neighbor sets
                # Actually, I can just check:
                # if u in neighbors[c_in] is O(13).
                # So total O(13*13) = 169 per swap check.
                # 80 * 650 * 169 ~ 8.8 million ops.
                # This is feasible in Python if optimized? Maybe 1-2 seconds per full sweep.
                # We have 100s. So ~50-100 sweeps.
                
                # Let's implement the check properly.
                
                # To make "u in neighbors[c_in]" faster, we can precompute a boolean matrix or sets.
                # Let's use sets for neighbors.
                
                # I'll refactor the neighbor storage to sets for membership testing.
                
                break # Dummy break to end loop structure in thought process
            
        # Re-implementing the search loop with Sets for neighbors
        
    # Refactoring for performance:
    # 1. neighbors_sets: list of sets
    # 2. Swap check:
    #    for u in neighbors_sets[c_out]:
    #        if u not in neighbors_sets[c_in]:
    #            if counts[u] < 2:
    #                valid = False
    #                break
    
    # 3. If valid, calculate new "potential".
    #    We want to know if this swap enables a removal.
    #    A removal of c is possible if min(counts[N(c)]) >= 2.
    #    After swap, counts change.
    #    Instead of checking all c, we can just apply the swap if it's "good".
    #    What is "good"?
    #    Let's define a heuristic: 
    #    Heuristic = sum of counts^2? (Minimize variance)
    #    Or simply: Does it create a word with min count >= 2?
    
    # Let's just find ANY valid swap that increases the number of words with min count >= 2.
    # Or, simpler: Just find a valid swap that maximizes the min count of the "worst" word?
    
    # Let's try: 
    # 1. Find all valid swaps.
    # 2. For each, calculate the new max removability score.
    # 3. Pick the best one.
    
    # This might be slow if we calculate score for all.
    # Score calculation is O(|C| * 13).
    
    # Alternative: 
    # Just pick a random valid swap if we are stuck.
    # Or, pick the swap that covers the most "critical" words (count=1).
    
    # Let's go with:
    # Find valid swaps.
    # Among them, pick one that maximizes the increase in "slack".
    # Slack = sum(max(0, counts[u]-1) for u in all words)?
    
    # Let's keep it simple:
    # 1. Try to find a swap that makes at least one word removable (score >= 2).
    # 2. If found, apply it.
    # 3. If not found, apply a random valid swap to perturb.
    
    # To check if a swap makes a word removable:
    # We need to check if there exists c in C such that min(new_counts[N(c)]) >= 2.
    # This is expensive to check for all c.
    
    # Approximation:
    # Check if the swap increases the count of any word that is currently "critical" (count 1).
    # If we increase a critical word's count to 2, it might become removable.
    
    # Let's implement the full search.
    
    # Precompute neighbor sets
    neighbor_sets = [set(nei) for nei in neighbors]
    
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        removed_any = False
        for c in list(code_indices):
            if all(counts[nb] > 1 for nb in neighbors[c]):
                for nb in neighbors[c]:
                    counts[nb] -= 1
                code_indices.remove(c)
                removed_any = True
                break
        
        if removed_any:
            if len(code_indices) < best_size:
                best_size = len(code_indices)
                best_code = list(code_indices)
            continue
        
        # Phase 2: Local Search
        # Find best swap
        best_swap = None
        best_score = -1
        
        # To speed up, we can limit the number of c_in checks per c_out
        # Or just do all.
        
        for c_out in code_indices:
            # If c_out is not "critical" (i.e., it doesn't have any neighbor with count 1), 
            # removing it is already safe? No, we checked that in Phase 1.
            # So every c_out has at least one neighbor with count 1.
            
            for c_in in range(q**n):
                if c_in in code_indices:
                    continue
                
                # Check validity
                valid = True
                for u in neighbors[c_out]:
                    if u not in neighbor_sets[c_in]:
                        if counts[u] < 2:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Calculate a score for this swap
                # We want to maximize the potential for removal.
                # Let's estimate the new min count for the "bottleneck" words.
                # This is expensive. 
                # Let's just count how many words have count 1 that are "fixed" by this swap.
                # A word u with count 1 is "fixed" if it is in N(c_in) and (not in N(c_out) or counts[u] > 1 in old state? No).
                # If u in N(c_in) and u in N(c_out): new count = counts[u]. (Unchanged)
                # If u in N(c_in) and u not in N(c_out): new count = counts[u] + 1. (Fixed if counts[u]==1)
                # If u not in N(c_in) and u in N(c_out): new count = counts[u] - 1. (Worse)
                
                # So we want to maximize the number of words u such that:
                # u in N(c_in) AND u not in N(c_out) AND counts[u] == 1.
                
                fixed_criticals = 0
                for u in neighbors[c_in]:
                    if u not in neighbor_sets[c_out] and counts[u] == 1:
                        fixed_criticals += 1
                
                # Also, we don't want to break too many non-criticals?
                # Validity check ensures we don't break any count < 2.
                
                if fixed_criticals > best_score:
                    best_score = fixed_criticals
                    best_swap = (c_out, c_in)
                    
                    # Early exit if we found a "very good" swap?
                    # Max possible is 13.
                    if best_score >= 5:
                        break
                
            if best_score >= 5:
                break
                
        if best_swap:
            c_out, c_in = best_swap
            # Apply swap
            for nb in neighbors[c_out]:
                counts[nb] -= 1
            for nb in neighbors[c_in]:
                counts[nb] += 1
            code_indices.remove(c_out)
            code_indices.add(c_in)
            
            # Update best if size decreased? Size didn't decrease.
            # But we moved to a potentially better state.
            
        else:
            # No good swap found. Try random valid swap to escape local minimum.
            # Or restart with perturbation.
            # Let's try a few random valid swaps.
            swapped = False
            for _ in range(10):
                c_out = random.choice(list(code_indices))
                c_in = random.randint(0, q**n - 1)
                if c_in in code_indices:
                    continue
                
                valid = True
                for u in neighbors[c_out]:
                    if u not in neighbor_sets[c_in]:
                        if counts[u] < 2:
                            valid = False
                            break
                            
                if valid:
                    for nb in neighbors[c_out]:
                        counts[nb] -= 1
                    for nb in neighbors[c_in]:
                        counts[nb] += 1
                    code_indices.remove(c_out)
                    code_indices.add(c_in)
                    swapped = True
                    break
                    
            if not swapped:
                # Stuck completely? Restart with seed + random perturbation?
                # Or just break and return best.
                # Let's try to restart the whole process with a different random seed?
                # Or just break.
                pass

    # Convert best_code indices back to words
    return [list(idx_to_word(idx)) for idx in best_code]
