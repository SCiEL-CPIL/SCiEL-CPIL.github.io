"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 52.0  # Safety margin before 90s limit
    best_sets = ([], [])
    best_size = 0

    def _can_add(S, sums, diffs, other_diffs, x):
        """Check if x can be added to set S."""
        # Sidon: all pair sums must be unique
        for a in S:
            s = a + x
            if s in sums:
                return False
        if (x + x) in sums:
            return False
        # Difference disjointness
        for a in S:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True

    def _add(S, sums, diffs, x):
        """Add x to set S, updating sums and diffs."""
        S.append(x)
        for a in S[:-1]:
            sums.add(a + x)
            diffs.add(x - a)
            diffs.add(a - x)
        sums.add(x + x)

    def _greedy_one_ordering(order, deadline, side_bias=None):
        """
        Single forward pass over the given ordering.
        side_bias: if 'first', always try set 0 first; if 'second', try set 1 first;
                   if None, try set 0 first (default).
        """
        sets = [[], []]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Determine order of sides to try
            if side_bias == 'second':
                sides = (1, 0)
            else:
                sides = (0, 1)
            
            for side in sides:
                S = sets[side]
                other_diffs = set_diffs[1 - side]
                
                if _can_add(S, set_sums[side], set_diffs[side], other_diffs, x):
                    _add(S, set_sums[side], set_diffs[side], x)
                    break
        
        return sets

    def _local_search(sets, deadline, max_iter=200):
        """
        Local search: try to improve by removing elements and re-adding,
        or by swapping elements between sets.
        """
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        for side in (0, 1):
            S = sets[side]
            for i, a in enumerate(S):
                for j, b in enumerate(S):
                    if i <= j:
                        set_sums[side].add(a + b)
                for b in S:
                    if a != b:
                        set_diffs[side].add(a - b)
                        set_diffs[side].add(b - a)
        
        current_size = len(sets[0]) + len(sets[1])
        all_candidates = list(range(1, 257))
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding a random unplaced element
            unplaced = [x for x in all_candidates if x not in sets[0] and x not in sets[1]]
            if unplaced:
                random.shuffle(unplaced)
                for x in unplaced[:20]:
                    for side in (0, 1):
                        S = sets[side]
                        other_diffs = set_diffs[1 - side]
                        if _can_add(S, set_sums[side], set_diffs[side], other_diffs, x):
                            _add(S, set_sums[side], set_diffs[side], x)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element from one set and re-adding it to the other
            if not improved and current_size > 4:
                for side in (0, 1):
                    if len(sets[side]) > 1:
                        idx = random.randint(0, len(sets[side]) - 1)
                        x = sets[side][idx]
                        other_side = 1 - side
                        
                        # Check if x can be added to the other set
                        S_other = sets[other_side]
                        if _can_add(S_other, set_sums[other_side], set_diffs[other_side], set_diffs[side], x):
                            # Remove x from current set
                            sets[side].pop(idx)
                            # Rebuild sums and diffs for current set
                            set_sums[side] = set()
                            set_diffs[side] = set()
                            S = sets[side]
                            for i, a in enumerate(S):
                                for j, b in enumerate(S):
                                    if i <= j:
                                        set_sums[side].add(a + b)
                                for b in S:
                                    if a != b:
                                        set_diffs[side].add(a - b)
                                        set_diffs[side].add(b - a)
                            # Add x to other set
                            _add(S_other, set_sums[other_side], set_diffs[other_side], x)
                            improved = True
                            break
            
            if not improved:
                break
        
        return sets

    base_order = list(range(1, 257))
    
    # First try natural (ascending) order - deterministic baseline
    sets = _greedy_one_ordering(base_order, deadline)
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try descending order
    sets = _greedy_one_ordering(base_order[::-1], deadline)
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try with side_bias='second'
    sets = _greedy_one_ordering(base_order, deadline, side_bias='second')
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = (list(sets[0]), list(sets[1]))
    
    # Try multiple random permutations
    random.seed(42)
    while time.time() < deadline - 5.0:
        random_order = base_order[:]
        random.shuffle(random_order)
        
        sets = _greedy_one_ordering(random_order, deadline)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = (list(sets[0]), list(sets[1]))
    
    # Apply local search to the best solution
    if best_size > 0 and time.time() < deadline - 3.0:
        polished = _local_search([list(best_sets[0]), list(best_sets[1])], deadline, max_iter=300)
        size = len(polished[0]) + len(polished[1])
        if size > best_size:
            best_size = size
            best_sets = (list(polished[0]), list(polished[1]))
    
    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets[0] = [1]
    elif not best_sets[1]:
        best_sets[1] = [1]
    
    return best_sets
# EVOLVE-BLOCK-END
