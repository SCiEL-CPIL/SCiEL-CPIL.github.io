import time
import random
import itertools

def _count_comparable(family):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(family)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = family[i], family[j]
            if len(a) < len(b):
                if all(x in b for x in a):
                    count += 1
            elif len(b) < len(a):
                if all(x in a for x in b):
                    count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    n = 12
    m = 40
    
    # Precompute all subsets as frozensets and their masks
    all_subsets = []
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            all_subsets.append(frozenset(combo))
    
    all_subsets_set = set(all_subsets)
    
    # Initial family: build from chains
    # Create a base chain: empty set, then add elements one by one
    # Then create multiple such chains with different orderings
    
    # Strategy: Create 3-4 chains of length ~10-13
    # Chain 1: {}, {0}, {0,1}, {0,1,2}, ..., {0,1,...,11} (length 13)
    # Chain 2: {}, {1}, {1,2}, {1,2,3}, ... but we can't reuse {}
    
    # Better: use a greedy approach to build initial family
    # Start with empty and full set
    family = [frozenset(), frozenset(range(n))]
    remaining = all_subsets_set - set(family)
    
    # Greedily add sets that maximize comparable pairs with existing family
    while len(family) < m:
        best_set = None
        best_gain = -1
        for s in remaining:
            gain = 0
            for t in family:
                if len(s) < len(t) and all(x in t for x in s):
                    gain += 1
                elif len(t) < len(s) and all(x in s for x in t):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_set = s
        family.append(best_set)
        remaining.discard(best_set)
    
    # Convert to lists for scoring, but keep frozensets for fast comparison
    current_score = _count_comparable(family)
    
    # Local search: try swapping one set with another to improve
    random.seed(42)
    
    def try_improve(family, remaining, deadline):
        """Try to improve by swapping sets."""
        best_family = list(family)
        best_score = current_score
        improved = False
        
        # Try adding a new set and removing one that hurts
        for new_s in list(remaining):
            if time.time() > deadline:
                break
            # Find which existing set to remove
            for i in range(len(best_family)):
                old_s = best_family[i]
                # Calculate score change
                # Remove old_s, add new_s
                # Score change = (comparables of new_s with family minus old_s) - (comparables of old_s with family minus new_s)
                # But this is approximate since new_s and old_s might be comparable
                
                # Quick check: if new_s is comparable with old_s, it's more complex
                new_with_old = 0
                if len(new_s) < len(old_s) and all(x in old_s for x in new_s):
                    new_with_old = 1
                elif len(old_s) < len(new_s) and all(x in new_s for x in old_s):
                    new_with_old = 1
                
                # Count comparables of new_s with (family - {old_s})
                new_gain = 0
                for j in range(len(best_family)):
                    if j == i:
                        continue
                    t = best_family[j]
                    if len(new_s) < len(t) and all(x in t for x in new_s):
                        new_gain += 1
                    elif len(t) < len(new_s) and all(x in new_s for x in t):
                        new_gain += 1
                new_gain += new_with_old
                
                # Count comparables of old_s with (family - {new_s}) = (family - {old_s}) since new_s not in family yet
                old_loss = 0
                for j in range(len(best_family)):
                    if j == i:
                        continue
                    t = best_family[j]
                    if len(old_s) < len(t) and all(x in t for x in old_s):
                        old_loss += 1
                    elif len(t) < len(old_s) and all(x in old_s for x in t):
                        old_loss += 1
                old_loss += new_with_old  # old_s was comparable with new_s position? No, new_s not there yet
                
                # Actually, the change is:
                # Remove old_s: lose old_loss comparables (with other family members)
                # Add new_s: gain new_gain comparables (with other family members) + new_with_old if comparable
                # But wait, new_with_old counts if new_s and old_s are comparable, but we're removing old_s
                # So the net change is: new_gain - old_loss
                
                delta = new_gain - old_loss
                if delta > 0:
                    new_score = best_score + delta
                    if new_score > best_score:
                        best_score = new_score
                        # Apply the swap
                        best_family[i] = new_s
                        remaining.discard(new_s)
                        remaining.add(old_s)
                        improved = True
                        break
            if time.time() > deadline:
                break
            if improved:
                break
        
        return best_family, remaining, best_score, improved
    
    # Iterative local search
    max_iterations = 100
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        family, remaining, current_score, improved = try_improve(family, remaining, deadline)
        if not improved:
            # Try random restarts or different perturbations
            # Add some randomness: swap a random pair
            if random.random() < 0.3:
                # Perturb: swap two random sets
                i, j = random.sample(range(m), 2)
                family[i], family[j] = family[j], family[i]
            else:
                break
    
    # Final: convert to lists of ints
    result = [list(s) for s in family]
    
    # Verify types
    for i, s in enumerate(result):
        result[i] = [int(x) for x in s]
    
    return result
