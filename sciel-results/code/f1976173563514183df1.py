"""Rewritten program: greedy construction of 40 subsets of {0..11} maximizing comparable pairs."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
N = 12
M = 40
ALL_MASKS = list(range(1 << N))  # 4096 masks
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _comparable(a, b):
    """Return True if a is strict subset of b or b is strict subset of a."""
    # Using bitmasks for speed
    return (a & b) == a and a != b or (a & b) == b and a != b

def _count_comparable(family_masks):
    """Count unordered pairs where one is strict subset of the other."""
    n = len(family_masks)
    count = 0
    for i in range(n):
        mi = family_masks[i]
        for j in range(i + 1, n):
            mj = family_masks[j]
            if (mi & mj) == mi and mi != mj:
                count += 1
            elif (mi & mj) == mj and mi != mj:
                count += 1
    return count

def _greedy_construct(seed, deadline):
    """Greedy construction with random restart, returns (count, family)."""
    rng = random.Random(seed)
    
    # Precompute: for each mask, list of all masks that are comparable with it
    # Actually, let's be smarter. For greedy, at each step we pick the mask
    # that maximizes the number of new comparable pairs added.
    
    # Precompute comparability matrix is 4096x4096 which is ~16M entries - too big.
    # Instead, compute on the fly but efficiently.
    
    # For a given mask m, the number of masks in current family F that are comparable with m
    # = |{f in F : f ⊂ m or f ⊃ m}|
    
    # We can precompute for each mask m:
    #   subsets_count: how many of the 4096 masks are strict subsets of m
    #   supersets_count: how many are strict supersets of m
    # But we need counts within the *current family*, not all masks.
    
    # Alternative: maintain for each candidate mask, a running score.
    # When we add mask x to the family, for every other mask y not yet added:
    #   score[y] += 1 if x ⊂ y or x ⊃ y
    
    # This is O(4096) per addition, so O(40 * 4096) = ~164K operations per restart. Very fast.
    
    # Initialize scores to 0
    scores = [0] * (1 << N)
    in_family = [False] * (1 << N)
    family = []
    
    # First element: pick randomly or pick a "central" one. Let's pick randomly from all masks.
    # Actually, to maximize comparability, we want a chain-like structure.
    # Let's just start with a random mask.
    first = rng.choice(ALL_MASKS)
    family.append(first)
    in_family[first] = True
    
    # Update scores for all other masks
    for y in ALL_MASKS:
        if y == first:
            continue
        if (first & y) == first and first != y:
            scores[y] += 1
        elif (first & y) == y and first != y:
            scores[y] += 1
    
    for _ in range(M - 1):
        if time.time() > deadline:
            break
        
        # Find the mask with highest score that's not in family
        best_mask = -1
        best_score = -1
        # To add randomness, we can consider top-k candidates
        # For now, just pick the best. Add ties randomly.
        candidates = []
        for y in ALL_MASKS:
            if not in_family[y]:
                if scores[y] > best_score:
                    best_score = scores[y]
                    best_mask = y
                    candidates = [y]
                elif scores[y] == best_score:
                    candidates.append(y)
        
        if best_mask == -1:
            # Shouldn't happen since 4096 > 40
            break
        
        # Pick randomly among top candidates for diversity
        chosen = rng.choice(candidates)
        family.append(chosen)
        in_family[chosen] = True
        
        # Update scores for all other masks
        for y in ALL_MASKS:
            if in_family[y]:
                continue
            if (chosen & y) == chosen and chosen != y:
                scores[y] += 1
            elif (chosen & y) == y and chosen != y:
                scores[y] += 1
    
    if len(family) < M:
        # Fill remaining with random masks not in family
        remaining = [m for m in ALL_MASKS if not in_family[m]]
        rng.shuffle(remaining)
        family.extend(remaining[:M - len(family)])
        in_family = [False] * (1 << N)
        for m in family:
            in_family[m] = True
    
    count = _count_comparable(family)
    return count, family

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_count = 0
    best_family = None
    
    # Run multiple greedy restarts
    # Each restart is fast (~164K ops), so we can run many
    num_restarts = 500
    
    for i in range(num_restarts):
        if time.time() > deadline - 2.0:  # Leave 2s margin
            break
        
        count, family = _greedy_construct(i, deadline)
        
        if count > best_count:
            best_count = count
            best_family = family
    
    # If we didn't find anything (shouldn't happen), fallback to simple construction
    if best_family is None:
        # Use the 40 smallest sets by cardinality from {0..5}
        vertices = list(range(6))
        all_sets = []
        for r in range(7):
            for combo in itertools.combinations(vertices, r):
                all_sets.append(list(combo))
        all_sets.sort(key=len)
        best_family = all_sets[:M]
    
    # Convert to lists of ints
    result = []
    for mask in best_family:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
