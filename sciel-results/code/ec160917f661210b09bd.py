import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 16
    
    # 1. Precompute all triples and their indices
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    idx_map = {t: i for i, t in enumerate(all_triples)}
    
    # 2. Precompute companion masks for each triple
    # companion_masks[i] is a set of bitmasks. Each mask represents 7 triples
    # that, together with triple i, form a K(2,2,2).
    # If (H_mask & mask) == mask, then adding i creates a forbidden configuration.
    
    # Optimization: We can precompute the list of 7 companion indices for each 
    # (triple, perm, x, y, z) configuration.
    
    companion_masks = [set() for _ in range(num_triples)]
    
    points = list(range(n))
    
    # Iterate over all triples
    for i, t in enumerate(all_triples):
        a, b, c = t
        others = [p for p in points if p not in (a, b, c)]
        
        # Try all 6 permutations of assigning a,b,c to the 3 pair positions
        for perm in itertools.permutations((a, b, c)):
            p1, p2, p3 = perm
            
            # Choose x, y, z from others
            # x is opposite to p1, y to p2, z to p3
            for x in others:
                for y in others:
                    if y == x: continue
                    for z in others:
                        if z == x or z == y: continue
                        
                        # The 8 transversals are:
                        # 1: (p1, p2, p3) -> This is t
                        # 2: (p1, p2, z)
                        # 3: (p1, y, p3)
                        # 4: (p1, y, z)
                        # 5: (x, p2, p3)
                        # 6: (x, p2, z)
                        # 7: (x, y, p3)
                        # 8: (x, y, z)
                        
                        # We need the indices of 2-8
                        try:
                            mask = 0
                            mask |= 1 << idx_map[tuple(sorted((p1, p2, z)))]
                            mask |= 1 << idx_map[tuple(sorted((p1, y, p3)))]
                            mask |= 1 << idx_map[tuple(sorted((p1, y, z)))]
                            mask |= 1 << idx_map[tuple(sorted((x, p2, p3)))]
                            mask |= 1 << idx_map[tuple(sorted((x, p2, z)))]
                            mask |= 1 << idx_map[tuple(sorted((x, y, p3)))]
                            mask |= 1 << idx_map[tuple(sorted((x, y, z)))]
                            companion_masks[i].add(mask)
                        except KeyError:
                            # Should not happen as all are valid triples
                            pass
        
        # Convert set to list for faster iteration in the check
        companion_masks[i] = list(companion_masks[i])

    def is_safe(t_idx, H_mask):
        """Check if adding triple t_idx to H_mask is safe."""
        masks = companion_masks[t_idx]
        for mask in masks:
            if (H_mask & mask) == mask:
                return False
        return True

    def greedy_construction(seed):
        rng = random.Random(seed)
        indices = list(range(num_triples))
        rng.shuffle(indices)
        
        H_mask = 0
        count = 0
        added_indices = []
        
        for idx in indices:
            if is_safe(idx, H_mask):
                H_mask |= (1 << idx)
                count += 1
                added_indices.append(idx)
        
        return added_indices, count

    def local_search(added_indices, seed):
        """
        Try to improve the solution by adding more triples.
        We keep the current set and try to add unadded triples in random order.
        """
        rng = random.Random(seed)
        current_mask = 0
        for idx in added_indices:
            current_mask |= (1 << idx)
        
        current_count = len(added_indices)
        current_set = set(added_indices)
        
        # List of unadded triples
        all_indices_set = set(range(num_triples))
        unadded = list(all_indices_set - current_set)
        
        # Try to add unadded triples in random order
        # Repeat this process a few times with different shuffles
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            rng.shuffle(unadded)
            progress = False
            for idx in unadded:
                if is_safe(idx, current_mask):
                    current_mask |= (1 << idx)
                    current_count += 1
                    current_set.add(idx)
                    unadded.remove(idx)
                    progress = True
            if not progress:
                break
        
        return list(current_set), current_count

    best_count = 0
    best_solution = []
    
    # Phase 1: Greedy Restart
    # Estimate time per greedy run:
    # ~560 checks * ~10k masks/check = 5.6M ops.
    # Python speed ~10M-50M simple ops/sec.
    # So ~0.1s per run.
    # We can do ~1000 runs in 100s.
    
    num_greedy = 2000 # Cap, will be limited by time
    
    for i in range(num_greedy):
        if time.time() > deadline - 10:
            break
        
        sol, count = greedy_construction(i)
        
        if count > best_count:
            best_count = count
            best_solution = sol
            
        # Occasionally do local search on the best solution found so far?
        # Or local search on this specific solution?
        # Local search is slower. Let's do it on the global best.
        
        if i % 10 == 0 and best_solution:
            # Use remaining time to try local search on best_solution
            # But we need to be careful not to block the main loop too long.
            # Let's just do greedy for now, it's very fast.
            pass

    # Phase 2: Local Search on the Best Solution
    if best_solution and time.time() < deadline - 5:
        # Try a few local search improvements on the best solution
        for i in range(100):
            if time.time() > deadline - 2:
                break
            sol, count = local_search(best_solution, 1000 + i)
            if count > best_count:
                best_count = count
                best_solution = sol

    # Convert indices back to triples
    return [list(all_triples[i]) for i in best_solution]

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found {len(res)} triples")
