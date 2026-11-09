"""Rewritten program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 50.0  # Safe margin for 90s wall-clock limit


def _pair_sums_ok(lst):
    """Check if all pair sums (including a+a) in lst are unique."""
    if len(lst) <= 1:
        return True
    sums = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i, len(lst)):
            s = a + lst[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _diffs(lst):
    """Return set of nonzero differences b-a for a,b in lst, a != b."""
    d = set()
    for i in range(len(lst)):
        for j in range(len(lst)):
            if i != j:
                d.add(lst[j] - lst[i])
    return d


def _can_add(lst, x, other_diffs):
    """Check if x can be added to lst while maintaining Sidon property and disjoint diffs."""
    # Check pair sums
    existing_sums = set()
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            existing_sums.add(lst[i] + lst[j])
    for a in lst:
        if a + x in existing_sums:
            return False
        existing_sums.add(a + x)
    # Also check x+x
    if x + x in existing_sums:
        return False
    
    # Check differences
    new_diffs = set()
    for a in lst:
        d = x - a
        new_diffs.add(d)
        d = a - x
        new_diffs.add(d)
    if new_diffs & other_diffs:
        return False
    # Check internal differences are valid (always true for distinct elements)
    return True


def _fast_greedy(order, deadline):
    """Fast greedy construction given an ordering of candidates."""
    sets = [[], []]
    diffs = [set(), set()]
    sums = [set(), set()]
    
    for x in order:
        if time.time() > deadline:
            break
        for side in (0, 1):
            if len(sets[side]) >= MAX_EACH:
                continue
            # Check pair sums
            ok = True
            for a in sets[side]:
                if a + x in sums[side]:
                    ok = False
                    break
            if ok:
                if x + x in sums[side]:
                    ok = False
            if not ok:
                continue
            # Check differences
            new_diffs = set()
            for a in sets[side]:
                new_diffs.add(x - a)
                new_diffs.add(a - x)
            if new_diffs & diffs[1 - side]:
                continue
            # Add x
            sets[side].append(x)
            diffs[side] |= new_diffs
            for a in sets[side]:
                sums[side].add(a + x)
            sums[side].add(x + x)
    
    return sets


def _polish(sets, order, deadline):
    """Local search: try to insert skipped elements."""
    sets = [list(s) for s in sets]
    diffs = [_diffs(sets[0]), _diffs(sets[1])]
    sums = [set(), set()]
    for side in (0, 1):
        for i in range(len(sets[side])):
            for j in range(i, len(sets[side])):
                sums[side].add(sets[side][i] + sets[side][j])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in order:
            if time.time() > deadline:
                break
            if x in sets[0] or x in sets[1]:
                continue
            for side in (0, 1):
                if len(sets[side]) >= MAX_EACH:
                    continue
                # Check pair sums
                ok = True
                for a in sets[side]:
                    if a + x in sums[side]:
                        ok = False
                        break
                if ok and x + x in sums[side]:
                    ok = False
                if not ok:
                    continue
                # Check differences
                new_diffs = set()
                for a in sets[side]:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                if new_diffs & diffs[1 - side]:
                    continue
                # Add x
                sets[side].append(x)
                diffs[side] |= new_diffs
                for a in sets[side]:
                    sums[side].add(a + x)
                sums[side].add(x + x)
                improved = True
                break
    
    return sets


def _swap_polish(sets, order, deadline):
    """Try swapping elements between sets to free up difference space."""
    sets = [list(s) for s in sets]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        diffs = [_diffs(sets[0]), _diffs(sets[1])]
        sums = [set(), set()]
        for side in (0, 1):
            for i in range(len(sets[side])):
                for j in range(i, len(sets[side])):
                    sums[side].add(sets[side][i] + sets[side][j])
        
        # Try removing an element from one set and adding a different one
        for side in (0, 1):
            if time.time() > deadline:
                break
            for idx, elem in enumerate(sets[side]):
                if time.time() > deadline:
                    break
                # Try removing this element
                test_sets = [list(s) for s in sets]
                test_sets[side] = [x for x in test_sets[side] if x != elem]
                test_diffs = [_diffs(test_sets[0]), _diffs(test_sets[1])]
                test_sums = [set(), set()]
                for s in (0, 1):
                    for i in range(len(test_sets[s])):
                        for j in range(i, len(test_sets[s])):
                            test_sums[s].add(test_sets[s][i] + test_sets[s][j])
                
                # Try adding a new element
                for x in order:
                    if time.time() > deadline:
                        break
                    if x in test_sets[0] or x in test_sets[1]:
                        continue
                    if len(test_sets[side]) >= MAX_EACH:
                        break
                    
                    # Check pair sums
                    ok = True
                    for a in test_sets[side]:
                        if a + x in test_sums[side]:
                            ok = False
                            break
                    if ok and x + x in test_sums[side]:
                        ok = False
                    if not ok:
                        continue
                    
                    # Check differences
                    new_diffs = set()
                    for a in test_sets[side]:
                        new_diffs.add(x - a)
                        new_diffs.add(a - x)
                    if new_diffs & test_diffs[1 - side]:
                        continue
                    
                    # Valid swap
                    sets = [list(s) for s in test_sets]
                    sets[side].append(x)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
    
    return sets


def _conflict_aware_order(sets, base_order, deadline):
    """Generate an ordering that prioritizes elements with fewer conflicts."""
    if time.time() > deadline:
        return base_order
    
    diffs = [_diffs(sets[0]), _diffs(sets[1])]
    all_diffs = diffs[0] | diffs[1]
    
    def conflict_score(x):
        """Score how many conflicts x would have."""
        score = 0
        for d in all_diffs:
            if x + d in range(1, N + 1):
                score += 1
            if x - d in range(1, N + 1):
                score += 1
        return score
    
    # Sort by conflict score (fewer conflicts first)
    scored = [(conflict_score(x), x) for x in base_order]
    scored.sort()
    return [x for _, x in scored]


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    best = [[], []]
    best_size = 0
    
    # Try multiple diverse orderings
    random.seed(42)
    base_order = list(range(1, N + 1))
    
    for trial in range(25):
        if time.time() > deadline:
            break
        
        # Create diverse ordering
        if trial == 0:
            order = base_order[:]
        elif trial == 1:
            order = base_order[::-1]
        elif trial % 2 == 0:
            # Low-first with noise
            order = base_order[:]
            random.shuffle(order)
            order.sort(key=lambda x: x + random.random() * 10)
        else:
            # Random shuffle
            order = base_order[:]
            random.shuffle(order)
        
        # Fast greedy
        sets = _fast_greedy(order, deadline)
        size = len(sets[0]) + len(sets[1])
        
        # Polish pass
        if time.time() < deadline:
            sets = _polish(sets, order, deadline)
            size = len(sets[0]) + len(sets[1])
        
        # Swap polish to escape local optima
        if time.time() < deadline:
            sets = _swap_polish(sets, order, deadline)
            size = len(sets[0]) + len(sets[1])
        
        # Conflict-aware reordering for next trial
        if time.time() < deadline and size > best_size:
            order = _conflict_aware_order(sets, base_order, deadline)
        
        if size > best_size:
            best_size = size
            best = [list(s) for s in sets]
    
    # Ensure nonempty
    if not best[0]:
        best[0] = [1]
    if not best[1]:
        best[1] = [2]
    
    return best
# EVOLVE-BLOCK-END
