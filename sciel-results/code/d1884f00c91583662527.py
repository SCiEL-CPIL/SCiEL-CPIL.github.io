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

def _greedy_construct(n=256, max_size=64, seed=42):
    """Greedy construction: add elements to maximize combined size."""
    rng = random.Random(seed)
    sets = [[], []]
    
    # Precompute all possible differences
    all_diffs = set(range(1, n))
    
    # Try multiple random orderings
    best_sets = [[], []]
    best_score = 0
    
    for trial in range(20):
        rng = random.Random(seed + trial)
        sets = [[], []]
        order = list(range(1, n + 1))
        rng.shuffle(order)
        
        used_diffs = [set(), set()]
        sums_sets = [set(), set()]
        
        for x in order:
            if len(sets[0]) >= max_size and len(sets[1]) >= max_size:
                break
            
            # Try adding to side 0
            if len(sets[0]) < max_size:
                # Check Sidon property for side 0
                ok_sidon = True
                for a in sets[0]:
                    if x + a in sums_sets[0]:
                        ok_sidon = False
                        break
                if ok_sidon:
                    # Check difference disjointness
                    new_diffs = set()
                    for a in sets[0]:
                        d = abs(x - a)
                        if d in used_diffs[1]:
                            ok_sidon = False
                            break
                        new_diffs.add(d)
                    if ok_sidon:
                        sets[0].append(x)
                        sums_sets[0].update(x + a for a in sets[0])
                        used_diffs[0].update(new_diffs)
                        continue
            
            # Try adding to side 1
            if len(sets[1]) < max_size:
                ok_sidon = True
                for a in sets[1]:
                    if x + a in sums_sets[1]:
                        ok_sidon = False
                        break
                if ok_sidon:
                    new_diffs = set()
                    for a in sets[1]:
                        d = abs(x - a)
                        if d in used_diffs[0]:
                            ok_sidon = False
                            break
                        new_diffs.add(d)
                    if ok_sidon:
                        sets[1].append(x)
                        sums_sets[1].update(x + a for a in sets[1])
                        used_diffs[1].update(new_diffs)
        
        score = _score(sets)
        if score > best_score:
            best_score = score
            best_sets = [list(sets[0]), list(sets[1])]
    
    return best_sets

def _local_search(sets, n=256, max_size=64, max_iters=5000, seed=42):
    """Local search to improve the solution."""
    rng = random.Random(seed)
    current = [list(sets[0]), list(sets[1])]
    current_score = _score(current)
    
    # Precompute helpers
    def can_add(side, x):
        s = current[side]
        # Check Sidon
        for a in s:
            s2 = s + [x]
            # Check if x+a collides
            pass
        # Proper check
        sums = set()
        for a, b in combinations_with_replacement(s, 2):
            sums.add(a + b)
        for a in s:
            if x + a in sums:
                return False
        # Check differences
        other_diffs = set()
        for a, b in combinations(current[1 - side], 2):
            other_diffs.add(abs(b - a))
        for a in s:
            d = abs(x - a)
            if d in other_diffs:
                return False
        return True
    
    def remove_from_side(side, x):
        current[side].remove(x)
    
    for _ in range(max_iters):
        improved = False
        
        # Try adding elements
        for x in range(1, n + 1):
            if x in current[0] or x in current[1]:
                continue
            for side in (0, 1):
                if len(current[side]) >= max_size:
                    continue
                if can_add(side, x):
                    current[side].append(x)
                    new_score = _score(current)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current[side].remove(x)
            if improved:
                break
        
        if improved:
            continue
        
        # Try removing and re-adding different elements
        for side in (0, 1):
            if not current[side]:
                continue
            for x in list(current[side]):
                current[side].remove(x)
                # Try to add something better
                best_add = None
                best_val = -1
                for y in range(1, n + 1):
                    if y in current[0] or y in current[1]:
                        continue
                    if can_add(side, y):
                        # Evaluate if this improves things
                        # For now, just try to fill
                        best_add = y
                        break
                if best_add is not None:
                    current[side].append(best_add)
                    if _score(current) > current_score:
                        current_score = _score(current)
                        improved = True
                        break
                else:
                    current[side].append(x)
            if improved:
                break
        
        if not improved:
            break
    
    return current

def construct():
    # Start with greedy construction
    sets = _greedy_construct(n=256, max_size=64, seed=42)
    
    # Ensure non-empty
    if not sets[0]:
        sets[0] = [1]
    if not sets[1]:
        sets[1] = [2]
    
    # Local search to improve
    sets = _local_search(sets, n=256, max_size=64, max_iters=2000, seed=123)
    
    # Final validation
    if not _valid(sets):
        # Fallback to a simple valid construction
        sets = [[1], [2]]
    
    # Cap sizes
    sets[0] = sets[0][:64]
    sets[1] = sets[1][:64]
    
    return [sets[0], sets[1]]
# EVOLVE-BLOCK-END
