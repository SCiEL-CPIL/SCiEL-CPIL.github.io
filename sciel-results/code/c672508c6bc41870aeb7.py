"""Program for maximizing 3-AP count in 4-AP-free subset of [1,256]."""
# EVOLVE-BLOCK-START
import time
import random

N = 256

def _precompute_aps():
    """Precompute all 3-APs and 4-APs as (a, d) pairs and as element tuples."""
    three_aps = []
    four_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            three_aps.append((a, b, c))
            if e <= N:
                four_aps.append((a, b, c, e))
    return three_aps, four_aps

def _precompute_conflict_bitmasks():
    """For each element, precompute bitmask of elements that form a 4-AP with it."""
    # conflict_mask[i] = bitmask of elements j such that i and j are in some 4-AP together
    # Actually, we need: for each pair (i, j), whether they can coexist without forming a 4-AP
    # Better: for each element x, store bitmask of all elements that form a 4-AP containing x
    # But we need: adding x to set S is valid iff no 4-AP is fully contained in S ∪ {x}
    # A 4-AP is fully contained if all 4 elements are in S ∪ {x}
    # Since we're adding one element at a time, we check: does S already contain 3 elements of some 4-AP that includes x?
    
    # Precompute for each element x: list of (other1, other2, other3) triples such that {x, o1, o2, o3} forms a 4-AP
    # Then adding x is valid iff for no such triple are all three others already in S
    
    # For speed, use bitmask approach:
    # For each element x, precompute a list of bitmasks, each representing a set of 3 elements
    # such that together with x they form a 4-AP.
    # Adding x is valid iff S & mask != mask for all such masks (i.e., not all 3 are in S)
    
    four_aps_by_element = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                ap = (a, b, c, e)
                for idx, elem in enumerate(ap):
                    others = [ap[j] for j in range(4) if j != idx]
                    mask = 0
                    for o in others:
                        mask |= (1 << (o - 1))
                    four_aps_by_element[elem].append(mask)
    
    return four_aps_by_element

def _count_3aps(mask, three_aps):
    """Count 3-APs fully contained in the set represented by mask."""
    count = 0
    for (a, b, c) in three_aps:
        if (mask & (1 << (a - 1))) and (mask & (1 << (b - 1))) and (mask & (1 << (c - 1))):
            count += 1
    return count

def _can_add(mask, x, four_aps_by_element):
    """Check if element x can be added to the set represented by mask."""
    bit = 1 << (x - 1)
    if mask & bit:
        return False
    for ap_mask in four_aps_by_element[x]:
        if (mask & ap_mask) == ap_mask:
            return False
    return True

def _greedy_initial(four_aps_by_element):
    """Greedy construction: add elements in order if they don't create a 4-AP."""
    mask = 0
    for x in range(1, N + 1):
        if _can_add(mask, x, four_aps_by_element):
            mask |= (1 << (x - 1))
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    three_aps, four_aps = _precompute_aps()
    four_aps_by_element = _precompute_conflict_bitmasks()
    
    # Greedy initial solution
    best_mask = _greedy_initial(four_aps_by_element)
    best_score = _count_3aps(best_mask, three_aps)
    
    # Simulated annealing
    current_mask = best_mask
    current_score = best_score
    
    # Precompute for each element: which 3-APs it participates in (for fast delta computation)
    # three_aps_by_element[x] = list of (other1, other2) such that {x, o1, o2} is a 3-AP
    three_aps_by_element = [[] for _ in range(N + 1)]
    for (a, b, c) in three_aps:
        three_aps_by_element[a].append((b, c))
        three_aps_by_element[b].append((a, c))
        three_aps_by_element[c].append((a, b))
    
    # Also: for 4-AP conflicts, when removing x, some elements become addable
    # For efficiency, track the set as a Python set for membership checks
    
    current_set = set()
    for x in range(1, N + 1):
        if current_mask & (1 << (x - 1)):
            current_set.add(x)
    
    best_set = set(current_set)
    
    # Temperature schedule
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Pick a random element to toggle
        x = rng.randint(1, N)
        bit = 1 << (x - 1)
        
        if current_mask & bit:
            # Try removing x
            new_mask = current_mask ^ bit
            # Compute score change: count 3-APs that contain x and have other two in set
            delta = 0
            for (o1, o2) in three_aps_by_element[x]:
                if (new_mask & (1 << (o1 - 1))) and (new_mask & (1 << (o2 - 1))):
                    delta -= 1
            new_score = current_score + delta
            
            # Accept if better or with probability exp(delta/T) for delta < 0
            if delta >= 0:
                accept = True
            else:
                accept = rng.random() < (2.718281828 ** (delta / max(T, 1e-10)))
            
            if accept:
                current_mask = new_mask
                current_score = new_score
                current_set.discard(x)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_set = set(current_set)
        else:
            # Try adding x
            if not _can_add(current_mask, x, four_aps_by_element):
                continue
            new_mask = current_mask | bit
            # Compute score change: count 3-APs that contain x and have other two in set
            delta = 0
            for (o1, o2) in three_aps_by_element[x]:
                if (current_mask & (1 << (o1 - 1))) and (current_mask & (1 << (o2 - 1))):
                    delta += 1
            new_score = current_score + delta
            
            if delta >= 0:
                accept = True
            else:
                accept = rng.random() < (2.718281828 ** (delta / max(T, 1e-10)))
            
            if accept:
                current_mask = new_mask
                current_score = new_score
                current_set.add(x)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_set = set(current_set)
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Final local search: try all single-element additions/removals
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            if best_mask & bit:
                # Try removing
                new_mask = best_mask ^ bit
                delta = 0
                for (o1, o2) in three_aps_by_element[x]:
                    if (new_mask & (1 << (o1 - 1))) and (new_mask & (1 << (o2 - 1))):
                        delta -= 1
                if delta > 0:  # removing would increase score? No, delta is negative when removing
                    # Actually delta here is the change in score from removing x
                    # If delta > 0, that means removing x increases score (unlikely)
                    pass
            else:
                # Try adding
                if _can_add(best_mask, x, four_aps_by_element):
                    delta = 0
                    for (o1, o2) in three_aps_by_element[x]:
                        if (best_mask & (1 << (o1 - 1))) and (best_mask & (1 << (o2 - 1))):
                            delta += 1
                    if delta > 0:
                        best_mask |= bit
                        best_score += delta
                        improved = True
    
    result = []
    for x in range(1, N + 1):
        if best_mask & (1 << (x - 1)):
            result.append(x)
    
    return sorted(result)

# EVOLVE-BLOCK-END
