"""
Model-and-optimize: Precomputed 3-AP/4-AP structures + Fast Simulated Annealing with bitmasks.
"""
import time
import numpy as np
import random

N = 256
# Deadline fraction of 90s. We use 85s to be safe.
TIME_LIMIT = 85.0

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Precompute 3-APs and 4-APs
    # Bit i corresponds to element i+1. So element x has bit (x-1).
    
    # For each x, list of bitmasks of pairs {y, z} such that {x, y, z} is a 3-AP.
    # If we add x to a set S, the number of new 3-APs is the count of pairs in three_ap_pairs[x]
    # that are fully contained in S.
    three_ap_pairs = [[] for _ in range(N + 1)]
    
    # For each x, list of bitmasks of triples {y, z, w} such that {x, y, z, w} is a 4-AP.
    # If we add x to a set S, it's invalid if any triple in four_ap_triples[x] is fully contained in S.
    four_ap_triples = [[] for _ in range(N + 1)]
    
    # We also need to be able to count 3-APs in a set quickly.
    # We can precompute all 3-AP masks.
    all_3ap_masks = []
    
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
                all_3ap_masks.append(mask)
                
                # x=a: partners b, c
                three_ap_pairs[a].append((1 << (b - 1)) | (1 << (c - 1)))
                # x=b: partners a, c
                three_ap_pairs[b].append((1 << (a - 1)) | (1 << (c - 1)))
                # x=c: partners a, b
                three_ap_pairs[c].append((1 << (a - 1)) | (1 << (b - 1)))
                
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                # x=a: partners b, c, e
                four_ap_triples[a].append((1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
                # x=b: partners a, c, e
                four_ap_triples[b].append((1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
                # x=c: partners a, b, e
                four_ap_triples[c].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1)))
                # x=e: partners a, b, c
                four_ap_triples[e].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)))

    def count_3aps(mask):
        """Count 3-APs in a given bitmask."""
        count = 0
        for m in all_3ap_masks:
            if m & mask == m:
                count += 1
        return count

    def is_valid(mask, x):
        """Check if x can be added to mask without creating a 4-AP."""
        # x is 1-indexed
        for triple_mask in four_ap_triples[x]:
            if triple_mask & mask == triple_mask:
                return False
        return True

    def gain_add(mask, x):
        """Calculate number of new 3-APs if x is added to mask."""
        count = 0
        for pair_mask in three_ap_pairs[x]:
            if pair_mask & mask == pair_mask:
                count += 1
        return count

    def lost_remove(mask, x):
        """Calculate number of 3-APs lost if x is removed from mask."""
        count = 0
        for pair_mask in three_ap_pairs[x]:
            if pair_mask & mask == pair_mask:
                count += 1
        return count

    def greedy_construct(seed):
        rng = random.Random(seed)
        mask = 0
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        # We can try to be a bit smarter: sort by potential gain? 
        # But gain depends on current mask. 
        # Simple random greedy is robust.
        
        for x in order:
            if is_valid(mask, x):
                # Optional: only add if it increases 3-AP count or if we are sparse
                # For now, just add if valid to maximize cardinality/density early
                mask |= (1 << (x - 1))
                
        return mask

    def simulated_annealing(initial_mask, seed, time_budget):
        rng = random.Random(seed)
        current_mask = initial_mask
        current_score = count_3aps(current_mask)
        
        # Precompute list of elements in mask and not in mask for faster iteration?
        # N=256 is small enough to just iterate 1..256 if we use bit checks.
        
        best_mask = current_mask
        best_score = current_score
        
        # Parameters
        T = 5.0  # Initial temperature
        T_min = 0.1
        alpha = 0.999  # Cooling rate
        
        # We estimate steps per second to set a limit, or just loop until time is up.
        # Let's loop until time budget is exhausted.
        
        while time.time() < start_time + time_budget:
            # Pick a move
            # Move types: Add, Remove, Swap
            # Swap is most likely to be useful.
            
            # Randomly select a type
            r = rng.random()
            
            if r < 0.5:
                # Swap: remove x, add y
                # Pick x from mask
                # To pick x from mask efficiently, we can maintain a list, 
                # but rebuilding is O(N). N=256 is small.
                in_mask = [i for i in range(1, N + 1) if (current_mask >> (i - 1)) & 1]
                if not in_mask:
                    continue
                x = rng.choice(in_mask)
                
                # Pick y not in mask
                # We can just try random y's until we find one not in mask
                y = rng.randint(1, N)
                attempts = 0
                while (current_mask >> (y - 1)) & 1 and attempts < 10:
                    y = rng.randint(1, N)
                    attempts += 1
                if (current_mask >> (y - 1)) & 1:
                    continue # Mask is too dense? Unlikely.
                
                # Check validity of new mask
                new_mask = (current_mask & ~(1 << (x - 1))) | (1 << (y - 1))
                
                # Check if y creates 4-AP in new_mask
                valid = True
                for triple_mask in four_ap_triples[y]:
                    if triple_mask & new_mask == triple_mask:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate score delta
                # Lost: 3-APs involving x in current_mask
                # Gained: 3-APs involving y in new_mask (excluding those involving x? No, x is removed)
                # Actually, gain_add(new_mask_without_y, y) is the number of 3-APs y forms with others in new_mask.
                # Since x is removed, the "others" are the same as in current_mask except x.
                # But gain_add counts pairs in the mask.
                
                lost = lost_remove(current_mask, x)
                # For gain, we need to check y against new_mask without y.
                mask_for_gain = new_mask & ~(1 << (y - 1))
                gained = gain_add(mask_for_gain, y)
                
                delta = gained - lost
                
                # Acceptance
                if delta > 0:
                    current_mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    # Boltzmann acceptance
                    if T > T_min:
                        prob = np.exp(delta / T)
                        if rng.random() < prob:
                            current_mask = new_mask
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_mask = current_mask
            
            elif r < 0.7:
                # Add: add y
                y = rng.randint(1, N)
                if (current_mask >> (y - 1)) & 1:
                    continue
                if not is_valid(current_mask, y):
                    continue
                
                new_mask = current_mask | (1 << (y - 1))
                delta = gain_add(current_mask, y)
                
                if delta > 0:
                    current_mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    if T > T_min:
                        prob = np.exp(delta / T)
                        if rng.random() < prob:
                            current_mask = new_mask
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_mask = current_mask
                                
            else:
                # Remove: remove x
                in_mask = [i for i in range(1, N + 1) if (current_mask >> (i - 1)) & 1]
                if not in_mask:
                    continue
                x = rng.choice(in_mask)
                
                new_mask = current_mask & ~(1 << (x - 1))
                delta = -lost_remove(current_mask, x)
                
                if delta > 0:
                    current_mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    if T > T_min:
                        prob = np.exp(delta / T)
                        if rng.random() < prob:
                            current_mask = new_mask
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_mask = current_mask
            
            # Cool
            T *= alpha
            
        return best_mask, best_score

    best_overall_mask = 0
    best_overall_score = -1
    
    # Phase 1: Generate diverse initial solutions
    num_initials = 10
    time_per_initial = (deadline - time.time()) / (num_initials + 1) # Reserve time for SA
    
    candidates = []
    for i in range(num_initials):
        if time.time() > deadline - 10: # Stop if running out of time
            break
        m = greedy_construct(i)
        s = count_3aps(m)
        candidates.append((m, s))
        if s > best_overall_score:
            best_overall_score = s
            best_overall_mask = m
            
    # Phase 2: Refine best candidates with SA
    # Sort candidates by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Use top 3 candidates for SA
    top_candidates = candidates[:3]
    
    time_elapsed = time.time() - start_time
    remaining_time = deadline - time_elapsed
    time_per_sa = remaining_time / len(top_candidates)
    
    for m, s in top_candidates:
        if time.time() > deadline - 5:
            break
        # Run SA
        bm, bs = simulated_annealing(m, 42, time_per_sa)
        if bs > best_overall_score:
            best_overall_score = bs
            best_overall_mask = bm
            
    # Convert mask to list
    result = [x + 1 for x in range(N) if (best_overall_mask >> x) & 1]
    
    if not result:
        return [1]
        
    return sorted(result)

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result size: {len(res)}")
    print(f"First few: {res[:10]}")
