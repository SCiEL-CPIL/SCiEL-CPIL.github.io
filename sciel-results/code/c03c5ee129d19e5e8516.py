import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin before 60s limit (evaluator says 60s, previous failed at 90s kill? Task says 60s finish. Previous error said "exceeded 60s". So 50s is safe.)
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    candidates = []
    candidate_tuples = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
        candidate_tuples.append(comb)
        
    N = len(candidates)
    
    # Precompute pairwise intersection sizes (0, 1, 2) for faster comparison
    # Storing sizes as small ints might be faster than bitmasks for equality checks?
    # Actually, bitmask equality is one CPU instruction. Size calculation is popcount.
    # Let's stick to bitmasks but store them in a flat list or 2D array.
    pair_int = [[0] * N for _ in range(N)]
    for i in range(N):
        ci = candidates[i]
        for j in range(i + 1, N):
            val = ci & candidates[j]
            pair_int[i][j] = val
            pair_int[j][i] = val

    def can_add(family_list, k_idx):
        """
        Check if candidate k_idx can be added to family (list of indices)
        without forming a sunflower.
        """
        # Group family members by their intersection with k_idx
        # Since intersections are small (0, 1, 2 bits), we can use a fixed-size array or dict
        # Using a dictionary for generality, but keys are bitmasks
        int_map = {}
        for idx in family_list:
            val = pair_int[idx][k_idx]
            if val not in int_map:
                int_map[val] = []
            int_map[val].append(idx)
        
        # For each group, check if any pair has mutual intersection equal to the group key
        for val, indices in int_map.items():
            if len(indices) < 2:
                continue
            # Check pairs in this group
            # Optimization: if group is large, it's likely a conflict exists? No.
            for i in range(len(indices)):
                a = indices[i]
                for j in range(i + 1, len(indices)):
                    b = indices[j]
                    if pair_int[a][b] == val:
                        return False
        return True

    best_family_list = []
    best_size = 0
    all_indices = list(range(N))

    # Phase 1: Randomized Greedy
    # We run multiple greedy passes.
    while time.time() < deadline - 5.0:
        random.shuffle(all_indices)
        family_list = []
        family_set = set() # For fast membership check
        
        for idx in all_indices:
            if can_add(family_list, idx):
                family_list.append(idx)
                family_set.add(idx)
        
        if len(family_list) > best_size:
            best_size = len(family_list)
            best_family_list = family_list.copy()
        
        # Phase 2: Local Search (Fast Version)
        # Instead of refilling everything, we do small swaps.
        # Try to improve by removing one element and trying to insert random others.
        improvements = 0
        local_deadline = time.time() + 2.0 # Limit local search time per restart
        
        while time.time() < local_deadline and improvements < 10:
            if not family_list:
                break
            
            # Pick a victim
            victim_idx = random.choice(family_list)
            
            # Try to find a replacement
            found_replacement = False
            # Sample a few random candidates to try replacing the victim
            # If we replace, size stays same, but might allow future additions?
            # Actually, standard local search for independent set:
            # Remove v. Try to add u. If valid, keep u.
            # This is a "1-swap".
            
            # To make it faster, we don't check ALL candidates. We check a random subset.
            # If we find one that fits, we accept it (even if size is same, it changes the structure).
            # Then we try to add more.
            
            family_list.remove(victim_idx)
            family_set.remove(victim_idx)
            
            # Try to add some random candidates
            # We want to see if we can get a larger set.
            # So we try to add ANY candidate.
            # To be fast, we pick a random permutation of a subset of remaining candidates.
            
            # Let's just try to greedily add from a random subset of the remaining candidates
            # This is still O(N * |F|^2) if we do it naively.
            
            # Better: Just try to add the victim back if we can't find an improvement?
            # No, let's try a specific heuristic:
            # Remove victim.
            # Try to add 100 random candidates from the rest.
            # If any fit, add them.
            # If size increased, great.
            # If size same, we might have swapped victim for a "better" candidate.
            
            remaining_indices = [i for i in all_indices if i not in family_set]
            random.shuffle(remaining_indices)
            
            # Try first 50 candidates in this random order
            added_any = False
            for idx in remaining_indices[:50]:
                if can_add(family_list, idx):
                    family_list.append(idx)
                    family_set.add(idx)
                    added_any = True
                    # We could break here, or continue to add more?
                    # If we add one, we should probably check if more can be added.
                    # But to keep it fast, let's just add this one and restart the local search loop.
                    break 
            
            if added_any:
                # We added at least one. The family size is same or larger?
                # We removed 1, added 1. Size is same.
                # But the structure changed.
                # Now try to fill up again?
                # Let's do a quick greedy pass on the remaining candidates to see if we can grow.
                # This might be expensive.
                
                # Instead, let's just accept this state and loop again.
                # If we can't grow, we might be stuck.
                
                # To ensure progress, let's do a full greedy pass on the current family_list
                # to see if it can grow.
                # This is O(N * |F|^2).
                # If we do this every local step, it's slow.
                
                # Alternative: Just accept the swap. The "growth" comes from the main greedy loop
                # which we can re-run? No, that's too slow.
                
                # Let's try a different local move:
                # Remove 2, try to add 3.
                # This is complex.
                
                # Let's stick to the simple swap and rely on the main loop's randomization.
                # If we didn't add anything, restore victim.
                if not added_any:
                    family_list.append(victim_idx)
                    family_set.add(victim_idx)
                else:
                    # We swapped. Now try to add more?
                    # Let's try to add from the remaining candidates (excluding the ones we tried)
                    # Just try the next 50.
                    for idx in remaining_indices[50:100]:
                        if can_add(family_list, idx):
                            family_list.append(idx)
                            family_set.add(idx)
                            break # Added one more, size increased!
                    
                    if len(family_list) > best_size:
                        best_size = len(family_list)
                        best_family_list = family_list.copy()
                        improvements += 1
                    elif len(family_list) == best_size:
                        # Same size, but maybe better structure?
                        # Accept with probability 0.5
                        if random.random() < 0.5:
                            improvements += 1
                        else:
                            # Roll back? Hard to roll back easily.
                            # Just keep it.
                            pass
            else:
                # Didn't add anything, so we restored victim.
                # No improvement.
                pass

    # Final check
    if len(best_family_list) < best_size:
        # This shouldn't happen
        pass

    # Convert best_family_list to a list of tuples
    result = []
    for idx in best_family_list:
        result.append(candidate_tuples[idx])
        
    return result
