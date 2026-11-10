import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin
    
    n = 180
    all_elems = list(range(1, n+1))
    
    # Precompute forbidden pairs for each a
    # forbidden[a] is a set of frozensets {b, c} or a list of tuples (b, c)
    # Using a list of tuples for easy iteration
    forbidden = {}
    for a in range(1, n+1):
        pairs = set()
        a2 = a * a
        # x * y = a2
        # b = a + x, c = a + y
        # b, c in [1, 180], b != c, b != a, c != a
        for x in range(1-a, 180-a+1):
            if x == 0:
                continue
            if a2 % x != 0:
                continue
            y = a2 // x
            if y == 0 or x == y:
                continue
            b = a + x
            c = a + y
            if 1 <= b <= n and 1 <= c <= n and b != a and c != a and b != c:
                # Store sorted pair to deduplicate
                pairs.add((min(b,c), max(b,c)))
        forbidden[a] = list(pairs)
    
    def is_valid(S):
        S_set = set(S)
        for a in S:
            for b, c in forbidden[a]:
                if b in S_set and c in S_set:
                    return False
        return True

    # Strategy: Start with a large valid subset
    # We know {91, ..., 180} is valid.
    # Let's try to start with even larger threshold if possible, or just 90.
    # Actually, let's try to construct greedily from large to small.
    
    best_S = set()
    
    # Initialize with top half
    initial_S = set(range(91, n+1))
    
    # Try to add elements from 90 down to 1
    for a in range(90, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Check if a can be added to initial_S
        conflict_found = False
        conflict_pairs = []
        for b, c in forbidden[a]:
            if b in initial_S and c in initial_S:
                conflict_found = True
                conflict_pairs.append((b, c))
        
        if not conflict_found:
            initial_S.add(a)
        else:
            # Conflict exists. We can try to remove one element from each conflicting pair.
            # To maximize size, we might just skip 'a' if it causes many conflicts.
            # Or remove the 'least critical' element.
            # Let's try to remove one element from the first conflicting pair and see if it resolves all.
            # This is a simplification. A better approach:
            # If number of conflicts is small, try removing an element.
            
            # Heuristic: Remove the element that appears in the most conflicts?
            # Or just skip 'a' to keep the current valid set.
            # Given the structure, large numbers are very stable.
            # Small numbers are very unstable.
            # So skipping small numbers is often better.
            
            # However, we can try to remove one element from initial_S to accommodate a.
            # Let's try removing each candidate from the first conflict and see if valid.
            # This is expensive.
            
            # For now, just skip 'a' if it conflicts.
            # We can try to add it back later in local search.
            pass
            
    best_S = set(initial_S)
    best_size = len(best_S)
    
    # Local Search: Try to add back skipped elements or swap
    removed = set(all_elems) - best_S
    
    # Try to add removed elements one by one
    for elem in sorted(removed, key=lambda x: -x):
        if time.time() - start_time > DEADLINE:
            break
        test_S = best_S | {elem}
        if is_valid(test_S):
            best_S = test_S
            best_size += 1
            removed.remove(elem)
            
    # Hill Climbing: Swap
    for _ in range(500):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        # Try swapping an element in best_S with one in removed
        if len(best_S) > 0 and len(removed) > 0:
            # Sample a few candidates to keep it fast
            candidates_in = random.sample(list(best_S), min(10, len(best_S)))
            candidates_out = random.sample(list(removed), min(10, len(removed)))
            
            for rem in candidates_in:
                for add in candidates_out:
                    test_S = best_S - {rem} | {add}
                    if is_valid(test_S):
                        # Accept if size increases (it won't in swap) or with some probability
                        # Here size is same. We might want to accept if it allows future additions?
                        # For now, only accept if it leads to a better "state" (hard to define).
                        # Let's just try to find a valid swap that allows adding more later?
                        # Simpler: Random walk on valid states of same size to escape local minima.
                        if random.random() < 0.05: # Low probability acceptance for same size
                            best_S = test_S
                            removed = set(all_elems) - best_S
                            improved = True
                            break
                if improved:
                    break
        
        # Try to add more
        for elem in list(removed):
            test_S = best_S | {elem}
            if is_valid(test_S):
                best_S = test_S
                removed.remove(elem)
                improved = True
                break

    return list(best_S)
