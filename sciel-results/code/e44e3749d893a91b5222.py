import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build Adjacency Bitmasks
    adj = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        adj[i] = mask
        
    full_mask = (1 << m) - 1
    
    # --- Helper Functions ---
    
    def greedy_min_degree(shuffle_seed):
        rng = random.Random(shuffle_seed)
        candidates = full_mask
        selected = 0
        
        for _ in range(m):
            if candidates == 0:
                break
            
            min_deg = m + 1
            min_nodes = []
            c = candidates
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)
            
            if not min_nodes:
                break
                
            v = rng.choice(min_nodes)
            selected |= (1 << v)
            candidates &= ~(adj[v] | (1 << v))
            
        return selected

    def local_search_improve(selected_mask):
        improved = True
        while improved:
            improved = False
            blocked = 0
            s = selected_mask
            while s:
                lsb = s & (-s)
                u = lsb.bit_length() - 1
                s ^= lsb
                blocked |= adj[u]
                
            candidates = full_mask & ~blocked & ~selected_mask
            
            if candidates != 0:
                lsb = candidates & (-candidates)
                v = lsb.bit_length() - 1
                selected_mask |= (1 << v)
                improved = True
        
        return selected_mask

    def swap_local_search(selected_mask, rng):
        best_mask = selected_mask
        s = selected_mask
        
        set_nodes = []
        temp_s = s
        while temp_s:
            lsb = temp_s & (-temp_s)
            u = lsb.bit_length() - 1
            set_nodes.append(u)
            temp_s ^= lsb
            
        if not set_nodes:
            return best_mask
            
        u = rng.choice(set_nodes)
        new_set = s & ~(1 << u)
        
        blocked = 0
        temp_s = new_set
        while temp_s:
            lsb = temp_s & (-temp_s)
            w = lsb.bit_length() - 1
            temp_s ^= lsb
            blocked |= adj[w]
            
        valid_v = full_mask & ~blocked & ~new_set & ~(1 << u)
        
        if valid_v == 0:
            return best_mask
            
        lsb = valid_v & (-valid_v)
        v = lsb.bit_length() - 1
        
        candidate_mask = new_set | (1 << v)
        candidate_mask = local_search_improve(candidate_mask)
        
        if candidate_mask.bit_count() > best_mask.bit_count():
            return candidate_mask
        return best_mask

    # --- Simulated Annealing ---
    
    def simulated_annealing(initial_mask, max_steps=200000):
        # Start from initial mask
        current_mask = initial_mask
        current_size = current_mask.bit_count()
        
        best_mask = current_mask
        best_size = current_size
        
        # Pre-compute list of nodes in set for faster removal? 
        # We'll just iterate bits.
        
        # Temperature schedule
        T = 2.0
        T_min = 0.01
        cooling = 0.99999 # Adjust to reach T_min in max_steps
        
        # We need a way to pick a random node from the set and a random node from the complement.
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Pick random node u in current set
            set_nodes = []
            c = current_mask
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                set_nodes.append(u)
                c ^= lsb
                
            if not set_nodes:
                break
                
            u = random.choice(set_nodes)
            
            # Remove u
            new_mask = current_mask & ~(1 << u)
            
            # Find valid v to add
            # blocked by (current_mask \ {u})
            blocked = 0
            s = new_mask
            while s:
                lsb = s & -s
                w = lsb.bit_length() - 1
                s ^= lsb
                blocked |= adj[w]
                
            valid_v = full_mask & ~blocked & ~new_mask
            # Note: u is not in new_mask, and u is not in valid_v?
            # u was removed. Is u compatible with new_mask? Yes, because it was in current_mask.
            # So u is in valid_v.
            
            if valid_v == 0:
                # Can't swap? Just re-add u?
                # Or skip?
                # If we can't find any v, we can't improve size.
                # Let's just continue to next step?
                # Or re-add u and try another u?
                # For SA, we need a move.
                # If no move, we are stuck?
                # Actually, if we remove u, we *always* have at least u available to add back.
                # So valid_v should never be 0.
                # Wait, is u compatible with new_mask?
                # Yes, because u was compatible with current_mask, and new_mask is a subset.
                # So u is always a valid candidate.
                # So valid_v != 0.
                
                # If it is 0, something is wrong.
                continue

            # Pick random v from valid_v
            # To pick random from bitmask efficiently:
            # We can count bits and pick a random index?
            # Or just pick the first one? That's biased.
            # Let's pick random index.
            # Since m is small, we can just build a list if needed, but that's slow.
            # Let's use a trick:
            # We want a random set bit in valid_v.
            # We can generate a random integer R in [0, 2^m) and see if bit is set?
            # No, that's inefficient if density is low.
            # Let's just pick the first one for now to save time, or random choice from a small list?
            # Actually, let's just pick the first one. It's deterministic but SA randomness comes from u choice.
            # Wait, if we always pick the first v, we might cycle.
            # Let's pick random v.
            # Method:
            # 1. Count bits in valid_v.
            # 2. Pick random k in [0, count).
            # 3. Find k-th set bit.
            # This is O(m).
            
            count = valid_v.bit_count()
            k = random.randint(0, count - 1)
            
            # Find k-th set bit
            temp_v = valid_v
            for i in range(k):
                lsb = temp_v & -temp_v
                temp_v ^= lsb
            lsb = temp_v & -temp_v
            v = lsb.bit_length() - 1
            
            # New state: new_mask | (1 << v)
            candidate_mask = new_mask | (1 << v)
            candidate_size = candidate_mask.bit_count()
            
            # Delta
            delta = candidate_size - current_size
            
            # Acceptance
            if delta >= 0 or (T > T_min and random.random() < np.exp(delta / T)):
                current_mask = candidate_mask
                current_size = candidate_size
                
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            
            # Cool down
            T *= cooling
            
        return best_mask

    # --- Main Logic ---
    
    best_size = 0
    best_mask = 0
    
    # Phase 1: Rapid Greedy to find a good starting point
    # This is fast and gives a decent initial set.
    for seed in range(100):
        if time.time() > deadline - 100:
            break
        mask = greedy_min_degree(seed)
        mask = local_search_improve(mask)
        size = mask.bit_count()
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Phase 2: Simulated Annealing
    # We will run SA from the best mask found, and also from other good masks if time permits.
    # But SA is the main driver now.
    
    # Let's run SA for the remaining time.
    # We can run multiple SA instances if time allows, but one long run is better.
    
    # To escape local optima better, we can restart SA with a perturbed best_mask if it gets stuck?
    # Or just run one long SA.
    
    # Let's run SA on best_mask.
    # We can also try to run SA on masks found in Phase 1 that are close to best.
    
    # For now, just run SA on best_mask.
    # If we have time, we can do more.
    
    # To make SA more effective, we should ensure we don't just cycle.
    # The random choice of u and v should handle that.
    
    # Let's run SA.
    final_mask = simulated_annealing(best_mask)
    
    if final_mask.bit_count() > best_size:
        best_size = final_mask.bit_count()
        best_mask = final_mask
        
    # If we have time left, do we do more?
    # The loop in simulated_annealing checks time.
    # So it will use up the time.
    
    # Final extraction
    if best_mask == 0:
        return np.array([0], dtype=np.int64)
        
    res = []
    msk = best_mask
    while msk:
        lsb = msk & (-msk)
        v = lsb.bit_length() - 1
        res.append(v)
        msk ^= lsb
        
    return np.array(res, dtype=np.int64)
