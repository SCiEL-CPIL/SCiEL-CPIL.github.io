"""Linear triple systems on 16 points with small independence number."""
import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    random.seed(42)
    
    N = 16
    ALL_TRIPLES = list(itertools.combinations(range(N), 3))
    
    # Precompute compatibility: compatible[i][j] = True if triples i and j share at most 1 point
    # We'll use bitmasks for speed
    triple_masks = []
    for t in ALL_TRIPLES:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # Precompute pair masks for each triple (to check linearity via pairs)
    # Two triples conflict if they share a pair (i.e., 2 points)
    # We can precompute for each triple the set of pairs it contains
    triple_pairs = []
    for t in ALL_TRIPLES:
        pairs = frozenset((min(t[0],t[1]), max(t[0],t[1])),)
        pairs = frozenset([
            frozenset((t[0], t[1])),
            frozenset((t[0], t[2])),
            frozenset((t[1], t[2]))
        ])
        triple_pairs.append(pairs)
    
    # Precompute conflict matrix: conflict[i][j] = True if triples i and j share a pair
    # This is equivalent to sharing 2 or 3 points
    num_triples = len(ALL_TRIPLES)
    conflict = [None] * num_triples
    for i in range(num_triples):
        conflict[i] = [False] * num_triples
        mi = triple_masks[i]
        for j in range(i + 1, num_triples):
            mj = triple_masks[j]
            shared = mi & mj
            # Count bits in shared
            if bin(shared).count('1') >= 2:
                conflict[i][j] = True
                conflict[j][i] = True
    
    def independence_number(triples):
        """Find the largest subset of {0..15} containing no triple."""
        if not triples:
            return N
        
        triple_masks_local = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks_local.append(mask)
        
        best = 0
        # Try all subsets from large to small
        # 2^16 = 65536 is feasible
        for mask in range((1 << N) - 1, -1, -1):
            popcount = bin(mask).count('1')
            if popcount <= best:
                continue
            has_triple = False
            for tm in triple_masks_local:
                if (mask & tm) == tm:
                    has_triple = True
                    break
            if not has_triple:
                best = popcount
        return best
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                shared = triple_masks[ALL_TRIPLES.index(triples[i])] & triple_masks[ALL_TRIPLES.index(triples[j])]
                # This is slow, use direct check
                if len(set(triples[i]) & set(triples[j])) > 1:
                    return False
        return True
    
    def greedy_maximal_independent_set(points, triples):
        """Greedy: find a large independent set (set of points with no triple)."""
        # This is actually the opposite - we want to find independent sets to target
        # For now, use the brute force
        return independence_number(triples)
    
    def random_greedy_packing(seed_offset=0):
        """Build a dense linear triple system via randomized greedy packing."""
        rng = random.Random(42 + seed_offset)
        used_pairs = set()
        triples = []
        
        # Order: prioritize triples that use "rare" points or spread out
        candidates = list(range(num_triples))
        rng.shuffle(candidates)
        
        # Multiple passes: first add triples that don't conflict
        for idx in candidates:
            t = ALL_TRIPLES[idx]
            pairs = [
                frozenset((t[0], t[1])),
                frozenset((t[0], t[2])),
                frozenset((t[1], t[2]))
            ]
            # Check if any pair is already used
            can_add = True
            for p in pairs:
                if p in used_pairs:
                    can_add = False
                    break
            if can_add:
                for p in pairs:
                    used_pairs.add(p)
                triples.append(t)
        
        return triples
    
    def local_search(triples, max_time=100):
        """Improve the system by swapping or replacing triples to reduce independence number."""
        best_triples = triples[:]
        best_score = independence_number(best_triples)
        
        current_triples = triples[:]
        current_score = best_score
        
        timeout = time.time() + max_time
        
        # Precompute current triple indices
        triple_to_idx = {t: i for i, t in enumerate(current_triples)}
        
        for iteration in range(50000):
            if time.time() > timeout:
                break
            
            # Random move: replace one triple with a random valid one
            idx = random.randint(0, len(current_triples) - 1)
            old_triple = current_triples[idx]
            
            # Generate a random replacement
            found = False
            for attempt in range(20):
                new_triple = tuple(sorted(random.sample(range(N), 3)))
                
                # Check if it can replace old_triple (i.e., doesn't conflict with other triples)
                valid = True
                for j, t in enumerate(current_triples):
                    if j == idx:
                        continue
                    if len(set(new_triple) & set(t)) > 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Try the replacement
                current_triples[idx] = new_triple
                new_score = independence_number(current_triples)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_triples = current_triples[:]
                    found = True
                    break
                else:
                    # Revert
                    current_triples[idx] = old_triple
                    break
            
            # Occasionally try adding a triple
            if len(current_triples) < 50 and random.random() < 0.3:
                new_triple = tuple(sorted(random.sample(range(N), 3)))
                valid = True
                for t in current_triples:
                    if len(set(new_triple) & set(t)) > 1:
                        valid = False
                        break
                if valid:
                    current_triples.append(new_triple)
                    new_score = independence_number(current_triples)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_triples = current_triples[:]
                    else:
                        current_triples.pop()
        
        return best_triples, best_score
    
    def targeted_local_search(triples, max_time=100):
        """Targeted search: find large independent sets and add triples to break them."""
        best_triples = triples[:]
        best_score = independence_number(best_triples)
        
        current_triples = triples[:]
        current_score = best_score
        
        timeout = time.time() + max_time
        
        for iteration in range(30000):
            if time.time() > timeout:
                break
            
            # Find a large independent set
            # Use a heuristic: try to find an independent set of size current_score
            # Then add a triple that hits it
            
            # Find largest independent set
            alpha = current_score
            if alpha < 4:
                break
            
            # Find an actual independent set of size alpha
            # Brute force for small alpha
            found_iset = None
            # Try random subsets of size alpha
            for attempt in range(100):
                subset = set(random.sample(range(N), alpha))
                is_independent = True
                for t in current_triples:
                    if set(t).issubset(subset):
                        is_independent = False
                        break
                if is_independent:
                    found_iset = subset
                    break
            
            if found_iset is None:
                # Try smaller
                for size in range(alpha - 1, 3, -1):
                    for attempt in range(50):
                        subset = set(random.sample(range(N), size))
                        is_independent = True
                        for t in current_triples:
                            if set(t).issubset(subset):
                                is_independent = False
                                break
                        if is_independent:
                            found_iset = subset
                            break
                    if found_iset:
                        break
            
            if found_iset is None:
                # No independent set found, maybe score is wrong
                current_score = independence_number(current_triples)
                if current_score < best_score:
                    best_score = current_score
                    best_triples = current_triples[:]
                continue
            
            # Try to add a triple that hits found_iset
            # A triple hits the independent set if it contains at least one point from it
            # But we want to add a triple that is a subset of found_iset to break it
            # Actually, to break an independent set S, we need to add a triple that is a subset of S
            
            # Find all triples that are subsets of found_iset
            iset_list = list(found_iset)
            candidates = []
            for a, b, c in itertools.combinations(iset_list, 3):
                candidates.append((a, b, c))
            
            random.shuffle(candidates)
            
            added = False
            for new_triple in candidates:
                # Check if it conflicts with existing triples
                valid = True
                for t in current_triples:
                    if len(set(new_triple) & set(t)) > 1:
                        valid = False
                        break
                if valid:
                    current_triples.append(new_triple)
                    new_score = independence_number(current_triples)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_triples = current_triples[:]
                    added = True
                    break
            
            if not added:
                # Try removing a random triple and adding a better one
                if len(current_triples) > 5:
                    idx = random.randint(0, len(current_triples) - 1)
                    removed = current_triples.pop(idx)
                    
                    # Try to add a triple from the independent set
                    for new_triple in candidates:
                        valid = True
                        for t in current_triples:
                            if len(set(new_triple) & set(t)) > 1:
                                valid = False
                                break
                        if valid:
                            current_triples.append(new_triple)
                            new_score = independence_number(current_triples)
                            if new_score < current_score:
                                current_score = new_score
                                if new_score < best_score:
                                    best_score = new_score
                                    best_triples = current_triples[:]
                            break
                    else:
                        # Couldn't add, put back
                        current_triples.append(removed)
        
        return best_triples, best_score
    
    # Phase 1: Generate initial systems with different seeds
    best_triples = None
    best_score = N + 1
    
    for seed_offset in range(10):
        if time.time() > deadline - 20:
            break
        initial = random_greedy_packing(seed_offset)
        if not initial:
            continue
        score = independence_number(initial)
        if score < best_score:
            best_score = score
            best_triples = initial[:]
    
    if best_triples is None:
        # Fallback
        best_triples = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
        best_score = independence_number(best_triples)
    
    # Phase 2: Local search
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        best_triples, best_score = local_search(best_triples, max_time=min(remaining_time, 60))
    
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        best_triples, best_score = targeted_local_search(best_triples, max_time=remaining_time)
    
    # Final check
    if not is_linear(best_triples):
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    return best_triples
