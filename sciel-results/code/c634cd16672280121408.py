import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = m
        
    # Precompute compatibility matrix
    # compat[i][j] is True if triple i and j share at most 1 point
    compat = [[False] * n_triples for _ in range(n_triples)]
    for i in range(n_triples):
        for j in range(i, n_triples):
            intersection = triple_masks[i] & triple_masks[j]
            # bit_count of intersection
            if bin(intersection).count('1') <= 1:
                compat[i][j] = True
                compat[j][i] = True
            # else False (default)
            
    def get_alpha(T_indices):
        # T_indices is a list of indices into all_triples
        if not T_indices:
            return 16
        T_set = set(T_indices)
        
        # Check sizes from 15 down to 1
        # We can prune: if we know alpha is at most K from previous steps, start there?
        # But for simplicity, just compute exactly.
        for k in range(16, 0, -1):
            # Iterate all masks with k bits
            found = False
            # Generate combinations of points
            # Using itertools.combinations is slow if done repeatedly in Python loop
            # But 16 is small.
            for combo in itertools.combinations(range(16), k):
                mask = 0
                for p in combo:
                    mask |= (1 << p)
                
                # Check if mask contains any triple from T
                contains_triple = False
                for t_idx in T_set:
                    if (triple_masks[t_idx] & mask) == triple_masks[t_idx]:
                        contains_triple = True
                        break
                if not contains_triple:
                    return k
        return 1 # Should not reach here for n=16 unless T covers all points? 
                 # Actually if T is empty, alpha=16.
                 # If T has triples, alpha < 16.
                 # If k=1, single point never contains a triple. So alpha >= 1.
    
    def generate_maximal_packing():
        available = list(range(n_triples))
        random.shuffle(available)
        T = []
        # We need to remove incompatible triples as we add
        # Instead of maintaining a list, we can just check against current T
        # But for speed, let's maintain a set of available indices
        # Actually, checking against T is O(|T|). |T| is small (< 40).
        # So just iterating through shuffled list and checking is fine.
        
        for idx in available:
            # Check if idx is compatible with all in T
            ok = True
            for t_idx in T:
                if not compat[idx][t_idx]:
                    ok = False
                    break
            if ok:
                T.append(idx)
        
        # This gives a maximal packing?
        # Not necessarily maximal in the sense that no more can be added.
        # It just adds if possible in the order of 'available'.
        # To make it maximal, we should loop until no change.
        # But the above loop goes through all triples once.
        # If a triple was skipped because it conflicted with an earlier one,
        # it might be compatible with a later set? No, T only grows.
        # So if it conflicted with T_now, it conflicts with T_final.
        # So this is a maximal packing (saturated).
        
        return T

    best_T = []
    best_alpha = 17
    
    # Initial packing
    T = generate_maximal_packing()
    current_alpha = get_alpha(T)
    best_alpha = current_alpha
    best_T = T[:]
    
    # Local Search
    while time.time() < deadline:
        # Try to add a triple
        # Find all triples compatible with T
        # This is O(n_triples * |T|)
        # We can cache this?
        # Let's just pick a random triple from all and check.
        
        # Strategy 1: Add
        added = False
        for _ in range(10): # Try 10 random adds
            idx = random.randint(0, n_triples - 1)
            ok = True
            for t_idx in best_T:
                if not compat[idx][t_idx]:
                    ok = False
                    break
            if ok and idx not in best_T:
                new_T = best_T + [idx]
                new_alpha = get_alpha(new_T)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_T = new_T
                    added = True
                    break
                # If new_alpha == best_alpha, we might still add it to enable future swaps?
                # But adding a triple cannot increase alpha.
                # So if alpha doesn't decrease, we can still add it to make the system denser?
                # Denser system might have lower alpha?
                # No, adding a triple restricts independent sets.
                # So alpha is non-increasing.
                # If alpha doesn't change, adding it doesn't hurt, but doesn't help immediately.
                # However, it might block future bad swaps?
                # Let's only add if it improves alpha, or if we are stuck?
                # Actually, if alpha doesn't change, we can add it and continue.
                # But we must ensure we don't loop forever adding non-improving triples.
                # Max number of triples is ~30. So we can add at most 30.
                if new_alpha == best_alpha:
                    best_T = new_T
                    # Don't break, try to add more?
                    # But we want to minimize alpha.
                    # Let's just add if it doesn't increase alpha (which it can't).
                    # So we can fill the packing.
                    pass
                else:
                    # new_alpha > best_alpha is impossible
                    pass
        
        # Strategy 2: Swap
        if len(best_T) > 0:
            # Pick a random triple to remove
            remove_idx = random.choice(best_T)
            # Pick a random triple to add
            add_idx = random.randint(0, n_triples - 1)
            if add_idx == remove_idx:
                continue
            # Check if add_idx is compatible with best_T \ {remove_idx}
            ok = True
            for t_idx in best_T:
                if t_idx == remove_idx:
                    continue
                if not compat[add_idx][t_idx]:
                    ok = False
                    break
            if ok:
                new_T = [t for t in best_T if t != remove_idx]
                new_T.append(add_idx)
                new_alpha = get_alpha(new_T)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_T = new_T

    # Convert indices back to triples
    result = [list(all_triples[i]) for i in best_T]
    return result
