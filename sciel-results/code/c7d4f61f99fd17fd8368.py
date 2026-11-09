import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    # Points 0..15
    n = 16
    points = list(range(n))
    
    # Helper to check if a triple is valid to add to a set of triples
    # A triple is valid if it shares at most 1 point with every other triple
    def is_valid(triple, triples_set):
        t_set = set(triple)
        for t in triples_set:
            if len(t_set & set(t)) > 1:
                return False
        return True

    # Helper to calculate independence number
    # Returns the size of the largest subset containing no triple
    def calc_alpha(triples):
        if not triples:
            return n
        
        # Convert to a set of frozensets for fast lookup
        tri_set = set(frozenset(t) for t in triples)
        
        # Check from largest possible size down
        # Max size is n
        # If a set of size k is independent, we return k
        # We can prune: if we find an independent set of size k, 
        # we don't need to check larger ones if we are going down? 
        # Actually, if size k is independent, size k+1 might not be.
        # We want the MAXIMUM size. So we check from n down to 1.
        # But checking n, n-1... is expensive if alpha is small.
        # Better: check from 1 up? No, we want the max.
        # However, if we know alpha is likely small (e.g. < 10), 
        # we can check sizes from 10 down to 1.
        # Or, we can use the property: if S is independent, all subsets are independent.
        # So if we find an independent set of size k, we know alpha >= k.
        # We want the largest k.
        
        # Optimization: 
        # If we have many triples, alpha is small.
        # Let's check sizes from n down to 1, but stop early if we find one?
        # No, if size 15 is independent, alpha is 15.
        # If size 15 is NOT independent, check 14.
        # This is efficient if alpha is close to n.
        # If alpha is small (e.g. 5), we check 16, 15, ..., 6 (fail), 5 (success).
        # Number of checks: sum_{k=6}^{16} C(16,k) which is huge.
        
        # Alternative: 
        # We can just iterate all subsets? 2^16 = 65536.
        # For each subset, check if it contains any triple.
        # This is O(2^n * |Triples| * 3).
        # 65536 * 40 * 3 ~ 8 million ops. Very fast in Python if optimized?
        # Maybe 1 second.
        # We can do this.
        
        max_alpha = 0
        # Iterate all subsets
        # Using bitmasks is faster
        for mask in range(1, 1 << n):
            # Check if this subset is independent
            # A subset is independent if it does NOT contain any triple
            # i.e., for all t in triples, t is NOT a subset of mask
            
            # Optimization: Precompute a list of triples as bitmasks
            # independent = True
            # for t_mask in tri_masks:
            #     if (mask & t_mask) == t_mask:
            #         independent = False
            #         break
            
            # If independent, update max_alpha
            # This is O(2^n * |T|).
            # 65536 * 40 = 2.6 million iterations.
            # This is acceptable.
            
            pass # Placeholder
            
        # Let's implement the bitmask check properly
        
        tri_masks = [0] * len(triples)
        for i, t in enumerate(triples):
            m = 0
            for p in t:
                m |= (1 << p)
            tri_masks[i] = m
        
        max_alpha = 0
        # We can prune: if popcount(mask) <= max_alpha, skip
        # But we want the max, so we check all.
        
        # Actually, we can just check from high sizes down and return the first one found?
        # No, because a large set might not be independent, but a smaller one is.
        # We want the LARGEST independent set.
        # So we check sizes k = n, n-1, ..., 1.
        # For a fixed k, check all C(n,k) subsets.
        # If any is independent, return k.
        
        for k in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(points, k):
                # Check if subset contains any triple
                subset_mask = 0
                for p in subset:
                    subset_mask |= (1 << p)
                
                independent = True
                for tm in tri_masks:
                    if (subset_mask & tm) == tm:
                        independent = False
                        break
                
                if independent:
                    found = True
                    break
            
            if found:
                return k
        
        return 0 # Should not happen, at least 1 point is independent

    # Initialize with a random greedy packing
    def random_packing():
        triples = []
        # Shuffle points
        pts = points[:]
        random.shuffle(pts)
        
        # Simple greedy: pick random triples, add if valid
        # To make it denser, we can try to add many
        # Number of possible triples is C(16,3) = 560
        all_triples = list(itertools.combinations(points, 3))
        random.shuffle(all_triples)
        
        for t in all_triples:
            if is_valid(t, triples):
                triples.append(t)
            if time.time() > deadline:
                break
        return triples

    # Local Search
    best_alpha = n
    best_triples = None
    
    while time.time() < deadline:
        # Start with a random packing
        current_triples = random_packing()
        
        # Optimize
        while time.time() < deadline:
            alpha = calc_alpha(current_triples)
            
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = current_triples[:]
                # If we hit a very low alpha, we can stop or continue
                if alpha <= 5: # Theoretical lower bound might be 5 or 6
                    # We can break early if we are confident
                    pass
            
            # Try to improve
            # Find largest independent set
            # We need to find the set, not just the size
            # Let's modify calc_alpha to return the set or re-implement finding it
            
            # For now, let's just do random swaps/adds
            # 1. Try to add a random valid triple
            if len(current_triples) < 560: # Max possible
                # Pick random triple
                t = random.choice(list(itertools.combinations(points, 3)))
                if is_valid(t, current_triples):
                    current_triples.append(t)
                    continue
            
            # 2. Try a swap
            # Remove a random triple
            idx = random.randrange(len(current_triples))
            removed = current_triples[idx]
            current_triples.pop(idx)
            
            # Try to add a different triple that was blocked by 'removed'
            # Or just add a random valid one
            # To be smarter, we can look for triples that share points with 'removed'
            # and were previously invalid.
            
            # Simple: try to re-add 'removed' or a random one
            # If we just re-add, no change.
            # Let's try to add a random valid triple to the modified set
            added = False
            for _ in range(10): # Try a few times
                t = random.choice(list(itertools.combinations(points, 3)))
                if is_valid(t, current_triples):
                    current_triples.append(t)
                    added = True
                    break
            
            if not added:
                # Re-add the removed one if nothing else works
                current_triples.append(removed)
        
        # Restart with different random seed if stuck?
        # The random_packing uses global random, so it varies.

    if best_triples is None:
        # Fallback to seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
