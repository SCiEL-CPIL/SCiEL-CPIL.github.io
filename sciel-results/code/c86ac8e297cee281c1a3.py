import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)  # 560
    
    # Precompute pairs for each triple
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
    
    # Precompute masks for each triple
    triple_masks = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
        
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute popcount for all 2^16 masks
    # And list of subsets by popcount
    max_mask = 1 << 16
    popcount = [0] * max_mask
    subsets_by_popcount = [[] for _ in range(17)]
    
    for i in range(max_mask):
        # Fast popcount
        c = 0
        x = i
        while x:
            x &= x - 1
            c += 1
        popcount[i] = c
        subsets_by_popcount[c].append(i)
        
    # Precompute supersets for each triple mask?
    # For a triple mask m, the supersets are all s such that (s & m) == m.
    # We can generate them on the fly or precompute.
    # Precomputing list of supersets for each of 560 triples:
    # Each has 2^13 = 8192 supersets. 560 * 8192 ~ 4.5M ints. ~36MB. Acceptable.
    triple_supersets = []
    for m in triple_masks:
        supersets = []
        # Iterate all masks? No, iterate submasks of complement.
        # Complement mask
        comp = (max_mask - 1) ^ m
        # Iterate all submasks of comp
        sub = comp
        while True:
            s = m | sub
            supersets.append(s)
            if sub == 0:
                break
            sub = (sub - 1) & comp
        triple_supersets.append(supersets)

    def get_alpha(bad_count):
        """Find max k such that exists s with popcount(s)=k and bad_count[s]==0"""
        for k in range(16, -1, -1):
            for s in subsets_by_popcount[k]:
                if bad_count[s] == 0:
                    return k
        return 0

    # State variables
    used_pairs = set()
    current_triples = []
    bad_count = [0] * max_mask
    current_alpha = 16
    
    def add_triple(idx):
        nonlocal current_alpha
        t = all_triples[idx]
        # Check validity (pairs)
        for p in triple_pairs[idx]:
            if p in used_pairs:
                return False
        
        # Add to state
        current_triples.append(t)
        for p in triple_pairs[idx]:
            used_pairs.add(p)
        
        # Update bad_count
        for s in triple_supersets[idx]:
            bad_count[s] += 1
            
        # Update alpha
        # Alpha can only decrease or stay same
        # Check if current_alpha is still valid
        # If not, find new alpha
        # Optimization: Check from current_alpha down
        new_alpha = current_alpha
        # Check if there is an independent set of size current_alpha
        found = False
        for s in subsets_by_popcount[current_alpha]:
            if bad_count[s] == 0:
                found = True
                break
        
        if not found:
            # Search lower
            for k in range(current_alpha - 1, -1, -1):
                for s in subsets_by_popcount[k]:
                    if bad_count[s] == 0:
                        new_alpha = k
                        found = True
                        break
                if found:
                    break
        current_alpha = new_alpha
        return True

    def remove_triple(idx):
        nonlocal current_alpha
        t = all_triples[idx]
        # Remove from state
        current_triples.remove(t)
        for p in triple_pairs[idx]:
            used_pairs.discard(p)
            
        # Update bad_count
        for s in triple_supersets[idx]:
            bad_count[s] -= 1
            
        # Update alpha
        # Alpha can only increase or stay same
        new_alpha = current_alpha
        # Check if there is an independent set of size current_alpha + 1
        # ... up to 16
        for k in range(current_alpha + 1, 17):
            for s in subsets_by_popcount[k]:
                if bad_count[s] == 0:
                    new_alpha = k
                    break
            if new_alpha > current_alpha:
                break
        current_alpha = new_alpha
        return True

    # Simulated Annealing
    best_alpha = 16
    best_triples = []
    
    # Initial random greedy
    indices = list(range(n_triples))
    random.shuffle(indices)
    for idx in indices:
        if add_triple(idx):
            pass # added
            
    if current_alpha < best_alpha:
        best_alpha = current_alpha
        best_triples = list(current_triples)
        
    # SA Parameters
    T = 1.0
    T_min = 0.01
    alpha_decay = 0.999
    
    # Precompute valid additions for speed? No, dynamic.
    
    sa_steps = 0
    while time.time() < deadline:
        sa_steps += 1
        
        # Choose move
        # 50% Add, 30% Remove, 20% Swap
        move_type = random.random()
        
        if move_type < 0.5:
            # Add a random valid triple
            # Find candidates
            candidates = []
            # To speed up, we can just pick random indices and check
            # Or iterate all? 560 is small.
            for idx in range(n_triples):
                if all(p not in used_pairs for p in triple_pairs[idx]):
                    candidates.append(idx)
            
            if not candidates:
                continue
                
            idx = random.choice(candidates)
            # Store old state to revert if rejected
            old_alpha = current_alpha
            old_triples_len = len(current_triples)
            
            # Apply Add
            # We need to revert bad_count and used_pairs if rejected
            # Since add_triple modifies state, we need a way to revert.
            # Let's implement a "try_add" that returns success and allows revert?
            # Or just copy state? Copying bad_count (65536 ints) is fast (~0.5ms).
            # Copying used_pairs (set of ~100 pairs) is fast.
            
            # Let's just use the functions and revert manually if needed.
            # But add_triple doesn't return enough info to revert easily without duplicating logic.
            
            # Let's refactor:
            # 1. Check if valid
            # 2. Save old bad_count? No, too slow.
            # 3. Apply change, calculate delta, if reject, undo change.
            
            # Undo Add:
            # 1. Remove from current_triples
            # 2. Remove from used_pairs
            # 3. Decrement bad_count
            # 4. Restore alpha (needs recalculation or storing old)
            
            # Let's store old_alpha before change.
            
            # Perform Add
            t = all_triples[idx]
            current_triples.append(t)
            for p in triple_pairs[idx]:
                used_pairs.add(p)
            for s in triple_supersets[idx]:
                bad_count[s] += 1
                
            # Calc new alpha
            new_alpha = current_alpha
            found = False
            for s in subsets_by_popcount[current_alpha]:
                if bad_count[s] == 0:
                    found = True
                    break
            if not found:
                for k in range(current_alpha - 1, -1, -1):
                    for s in subsets_by_popcount[k]:
                        if bad_count[s] == 0:
                            new_alpha = k
                            found = True
                            break
                    if found:
                        break
                            
            delta = new_alpha - current_alpha
            current_alpha = new_alpha
            
            # Accept?
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                # Accepted
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
            else:
                # Rejected: Undo
                current_triples.pop()
                for p in triple_pairs[idx]:
                    used_pairs.discard(p)
                for s in triple_supersets[idx]:
                    bad_count[s] -= 1
                current_alpha = old_alpha

        elif move_type < 0.8:
            # Remove a random triple
            if not current_triples:
                continue
                
            rem_pos = random.randrange(len(current_triples))
            t_rem = current_triples[rem_pos]
            idx = triple_to_idx[t_rem]
            
            old_alpha = current_alpha
            
            # Perform Remove
            current_triples.pop(rem_pos)
            for p in triple_pairs[idx]:
                used_pairs.discard(p)
            for s in triple_supersets[idx]:
                bad_count[s] -= 1
                
            # Calc new alpha
            new_alpha = current_alpha
            for k in range(current_alpha + 1, 17):
                for s in subsets_by_popcount[k]:
                    if bad_count[s] == 0:
                        new_alpha = k
                        break
                if new_alpha > current_alpha:
                    break
                    
            delta = new_alpha - current_alpha
            current_alpha = new_alpha
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                # Accepted
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
            else:
                # Rejected: Undo (Add back)
                current_triples.insert(rem_pos, t_rem)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
                for s in triple_supersets[idx]:
                    bad_count[s] += 1
                current_alpha = old_alpha

        else:
            # Swap: Remove one, Add one
            if not current_triples:
                continue
                
            # Remove
            rem_pos = random.randrange(len(current_triples))
            t_rem = current_triples[rem_pos]
            idx_rem = triple_to_idx[t_rem]
            
            # Find candidate to add (valid after removal)
            # Temporarily remove
            current_triples.pop(rem_pos)
            for p in triple_pairs[idx_rem]:
                used_pairs.discard(p)
            for s in triple_supersets[idx_rem]:
                bad_count[s] -= 1
                
            # Find valid add
            candidates = []
            for idx_add in range(n_triples):
                if idx_add == idx_rem:
                    continue
                if all(p not in used_pairs for p in triple_pairs[idx_add]):
                    candidates.append(idx_add)
                    
            if not candidates:
                # Undo removal
                current_triples.insert(rem_pos, t_rem)
                for p in triple_pairs[idx_rem]:
                    used_pairs.add(p)
                for s in triple_supersets[idx_rem]:
                    bad_count[s] += 1
                continue
                
            idx_add = random.choice(candidates)
            t_add = all_triples[idx_add]
            
            old_alpha = current_alpha
            
            # Perform Add
            current_triples.append(t_add)
            for p in triple_pairs[idx_add]:
                used_pairs.add(p)
            for s in triple_supersets[idx_add]:
                bad_count[s] += 1
                
            # Calc new alpha
            new_alpha = current_alpha
            found = False
            for s in subsets_by_popcount[current_alpha]:
                if bad_count[s] == 0:
                    found = True
                    break
            if not found:
                for k in range(current_alpha - 1, -1, -1):
                    for s in subsets_by_popcount[k]:
                        if bad_count[s] == 0:
                            new_alpha = k
                            found = True
                            break
                    if found:
                        break
                        
            delta = new_alpha - current_alpha
            current_alpha = new_alpha
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                # Accepted
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
            else:
                # Rejected: Undo Add, Undo Remove
                # Undo Add
                current_triples.pop()
                for p in triple_pairs[idx_add]:
                    used_pairs.discard(p)
                for s in triple_supersets[idx_add]:
                    bad_count[s] -= 1
                    
                # Undo Remove
                current_triples.insert(rem_pos, t_rem)
                for p in triple_pairs[idx_rem]:
                    used_pairs.add(p)
                for s in triple_supersets[idx_rem]:
                    bad_count[s] += 1
                current_alpha = old_alpha

        # Cool down
        T *= alpha_decay
        if T < T_min:
            T = T_min
            
        # Restart if stuck?
        # If T is low and no improvement for a while, restart?
        # Keep it simple for now.

    return best_triples
