def largest_sidon_subset(S):
    """Find largest subset of S with all pair sums (i<=j) unique."""
    S = sorted(S)
    n = len(S)
    from itertools import combinations
    
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            sums = set()
            valid = True
            for i in range(len(subset)):
                for j in range(i, len(subset)):
                    s = subset[i] + subset[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return k
    return 2

def construct():
    import random
    from itertools import combinations
    
    # Start with the densest possible set
    current_set = list(range(12))
    current_size = largest_sidon_subset(current_set)
    
    best_set = current_set[:]
    best_size = current_size
    
    # Local search to minimize the largest Sidon subset size
    # We try to break Sidon subsets of the current best size
    
    def is_sidon(subset):
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def find_sidon_subsets(S, k):
        S = sorted(S)
        results = []
        for subset in combinations(S, k):
            if is_sidon(subset):
                results.append(subset)
        return results

    # If best_size is > 4, we try to reduce it
    # We perform a limited number of iterations to stay within time limit
    max_iterations = 200
    for _ in range(max_iterations):
        if best_size <= 4:
            break
            
        # Find all Sidon subsets of size best_size
        # This can be expensive, so we limit or sample if too many
        subsets = find_sidon_subsets(best_set, best_size)
        
        if not subsets:
            # Should not happen if best_size is correct, but re-calc just in case
            # Actually, if we broke them, best_size should have been updated
            # But we only update after a move.
            # So if subsets is empty, it means our best_size is stale?
            # No, we didn't change the set.
            # So if subsets is empty, best_size was wrong?
            # Let's just recalculate and break if it dropped.
            new_size = largest_sidon_subset(best_set)
            if new_size < best_size:
                best_size = new_size
            continue

        # Pick a random subset to break
        sub = random.choice(subsets)
        
        # Pick a random element from the subset to move
        idx_in_sub = random.randint(0, len(sub)-1)
        val = sub[idx_in_sub]
        
        # Find its index in best_set
        try:
            idx_in_set = best_set.index(val)
        except ValueError:
            continue
            
        # Try to move it to a new value
        # We want to create a collision.
        # Simple strategy: move it to a value that is close to other elements
        # or just random.
        
        improved = False
        for _ in range(20):
            new_val = random.randint(0, 128)
            if new_val in best_set:
                continue
            
            # Save old state
            old_val = best_set[idx_in_set]
            
            # Apply move
            best_set[idx_in_set] = new_val
            best_set.sort()
            
            # Check new size
            # Optimization: Only check if it's smaller than best_size
            # We can check if any subset of size best_size is still Sidon
            # But largest_sidon_subset does that.
            
            new_size = largest_sidon_subset(best_set)
            
            if new_size < best_size:
                best_size = new_size
                improved = True
                if best_size <= 4:
                    break
                break
            else:
                # Revert
                best_set[idx_in_set] = old_val
                best_set.sort()
                
        if best_size <= 4:
            break

    return best_set
