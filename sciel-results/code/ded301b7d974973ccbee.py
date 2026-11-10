"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations, combinations_with_replacement

def _is_sidon(sets):
    """Check if all pair sums (including a+a) are unique within each set."""
    for s in sets:
        sums = [a + b for a, b in combinations_with_replacement(s, 2)]
        if len(sums) != len(set(sums)):
            return False
    return True

def _diffs_disjoint(sets):
    """Check if nonzero difference sets are disjoint."""
    d0 = set()
    d1 = set()
    for a, b in combinations(sets[0], 2):
        d0.add(abs(b - a))
    for a, b in combinations(sets[1], 2):
        d1.add(abs(b - a))
    return not (d0 & d1)

def _valid(sets):
    return _is_sidon(sets) and _diffs_disjoint(sets)

def _score(sets):
    return len(sets[0]) + len(sets[1])

def _build_sidon_with_diffs(allows, order, max_size=64):
    """
    Greedily build a Sidon set using only differences from 'allows'.
    order: list of candidate elements in a specific order.
    """
    s = []
    sums = set()
    
    for x in order:
        if len(s) >= max_size:
            break
        
        # Check difference constraint
        ok_diff = True
        for a in s:
            if abs(x - a) not in allows:
                ok_diff = False
                break
        
        if not ok_diff:
            continue
            
        # Check Sidon constraint
        ok_sidon = True
        for a in s:
            if x + a in sums:
                ok_sidon = False
                break
        
        if not ok_sidon:
            continue
        
        # Add x
        s.append(x)
        for a in s:
            sums.add(x + a)
            
    return s

def _construct_with_diff_pools(n=256, max_size=64, trials=100, seed=42):
    """
    Constructs two Sidon sets by partitioning the difference space.
    D(A) is restricted to Pool A, D(B) is restricted to Pool B.
    Pool A and Pool B are disjoint.
    """
    best_sets = [[], []]
    best_score = 0
    rng = random.Random(seed)
    
    all_diffs = list(range(1, n))
    elements = list(range(1, n + 1))
    
    # Pre-defined good partitions
    partitions = []
    
    # 1. Even/Odd
    p_even = set(d for d in all_diffs if d % 2 == 0)
    p_odd = set(d for d in all_diffs if d % 2 == 1)
    partitions.append((p_even, p_odd))
    partitions.append((p_odd, p_even))
    
    # 2. Mod 3
    p_m3_0 = set(d for d in all_diffs if d % 3 == 0)
    p_m3_1 = set(d for d in all_diffs if d % 3 == 1)
    p_m3_2 = set(d for d in all_diffs if d % 3 == 2)
    # Combine two for each side
    partitions.append((p_m3_0 | p_m3_1, p_m3_2))
    partitions.append((p_m3_2, p_m3_0 | p_m3_1))
    partitions.append((p_m3_0, p_m3_1 | p_m3_2))
    partitions.append((p_m3_1, p_m3_0 | p_m3_2))
    partitions.append((p_m3_1 | p_m3_2, p_m3_0))
    partitions.append((p_m3_2 | p_m3_0, p_m3_1))
    
    # 3. Random partitions
    for _ in range(trials):
        shuffled_diffs = list(all_diffs)
        rng.shuffle(shuffled_diffs)
        split = len(shuffled_diffs) // 2
        pa = set(shuffled_diffs[:split])
        pb = set(shuffled_diffs[split:])
        partitions.append((pa, pb))
        partitions.append((pb, pa)) # Symmetry

    for pa, pb in partitions:
        # Randomize element order
        order_a = list(elements)
        rng.shuffle(order_a)
        A = _build_sidon_with_diffs(pa, order_a, max_size)
        
        order_b = list(elements)
        rng.shuffle(order_b)
        B = _build_sidon_with_diffs(pb, order_b, max_size)
        
        if not _valid([A, B]):
            continue
            
        score = _score([A, B])
        if score > best_score:
            best_score = score
            best_sets = [A, B]
            
    return best_sets

def construct():
    # Use the new difference-pool construction
    sets = _construct_with_diff_pools(n=256, max_size=64, trials=100, seed=42)
    
    # Ensure non-empty
    if not sets[0]:
        sets[0] = [1]
    if not sets[1]:
        sets[1] = [2]
        
    # Final validation
    if not _valid(sets):
        # Fallback to a simple valid construction if something went wrong
        sets = [[1], [2]]
        
    # Cap sizes
    sets[0] = sets[0][:64]
    sets[1] = sets[1][:64]
    
    return [sets[0], sets[1]]
# EVOLVE-BLOCK-END
