"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 256
    max_size = 64
    
    def is_sidon(s):
        """Check if s is a Sidon set (all pair sums including a+a are unique)."""
        if not s:
            return True
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    def get_diffs(s):
        """Get nonzero difference set of s."""
        if not s:
            return set()
        diffs = set()
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    d = s[j] - s[i]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def try_add_greedy(sets, x, side):
        """Try to add x to sets[side] while maintaining Sidon property and disjoint diffs."""
        candidate = sets[side] + [x]
        if not is_sidon(candidate):
            return False
        diffs = get_diffs(candidate)
        other_diffs = get_diffs(sets[1 - side])
        if diffs & other_diffs:
            return False
        return True
    
    def random_order():
        """Generate a random permutation of [1, n]."""
        order = list(range(1, n + 1))
        random.shuffle(order)
        return order
    
    def greedy_construct(order, sets):
        """Greedy: for each element in order, try to add to set 0 then set 1."""
        for x in order:
            if len(sets[0]) >= max_size and len(sets[1]) >= max_size:
                break
            added = False
            for side in (0, 1):
                if len(sets[side]) >= max_size:
                    continue
                if try_add_greedy(sets, x, side):
                    sets[side].append(x)
                    added = True
                    break
            if not added:
                # Try the other order
                for side in (1, 0):
                    if len(sets[side]) >= max_size:
                        continue
                    if try_add_greedy(sets, x, side):
                        sets[side].append(x)
                        added = True
                        break
        return sets
    
    def local_search(sets):
        """Try to improve by swapping elements between sets or removing bad ones."""
        improved = True
        while improved:
            improved = False
            # Try to move an element from one set to the other
            for side in (0, 1):
                for i, x in enumerate(sets[side]):
                    if len(sets[1 - side]) >= max_size:
                        continue
                    # Remove x from sets[side]
                    sets[side].pop(i)
                    # Try to add x to sets[1-side]
                    if try_add_greedy(sets, x, 1 - side):
                        sets[1 - side].append(x)
                        improved = True
                    else:
                        # Put it back
                        sets[side].append(x)
            # Try to add any unused element
            used = set(sets[0]) | set(sets[1])
            for x in range(1, n + 1):
                if x in used:
                    continue
                for side in (0, 1):
                    if len(sets[side]) >= max_size:
                        continue
                    if try_add_greedy(sets, x, side):
                        sets[side].append(x)
                        used.add(x)
                        improved = True
                        break
        return sets
    
    # Main search loop
    best_sets = [[], []]
    best_score = 0
    
    # First pass: simple greedy with natural order
    sets = [[], []]
    for x in range(1, n + 1):
        if len(sets[0]) >= max_size and len(sets[1]) >= max_size:
            break
        for side in (0, 1):
            if len(sets[side]) >= max_size:
                continue
            if try_add_greedy(sets, x, side):
                sets[side].append(x)
                break
    best_sets = [list(sets[0]), list(sets[1])]
    best_score = len(sets[0]) + len(sets[1])
    
    # Multiple random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        if len(sets[0]) >= max_size and len(sets[1]) >= max_size:
            break
        
        sets = [[], []]
        order = random_order()
        greedy_construct(order, sets)
        local_search(sets)
        
        score = len(sets[0]) + len(sets[1])
        if score > best_score:
            best_score = score
            best_sets = [list(sets[0]), list(sets[1])]
    
    # Final local search on best
    if time.time() < deadline - 3:
        local_search(best_sets)
    
    # Ensure nonempty
    if not best_sets[0]:
        best_sets[0] = [1]
    if not best_sets[1]:
        best_sets[1] = [2]
    
    # Verify and fix if needed
    for side in (0, 1):
        best_sets[side] = sorted(best_sets[side])
    
    return best_sets
# EVOLVE-BLOCK-END
