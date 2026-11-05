"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    base_order = list(range(1, 257))
    best_sets = ([1], [2])
    best_size = 2
    
    def _greedy_one_ordering(order, deadline):
        """
        Single forward pass over the given ordering of candidates.
        For each candidate x, try to add it to set 0 first, then set 1.
        Maintain incremental sums and diffs for fast checks.
        """
        sets = [[], []]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            for side in (0, 1):
                S = sets[side]
                other_diffs = set_diffs[1 - side]
                
                # Check Sidon condition: all new pair sums a+x (a in S) and x+x
                # must not collide with existing sums in set_sums[side]
                new_sums_valid = True
                for a in S:
                    s = a + x
                    if s in set_sums[side]:
                        new_sums_valid = False
                        break
                if new_sums_valid:
                    if (x + x) in set_sums[side]:
                        new_sums_valid = False
                
                if not new_sums_valid:
                    continue
                
                # Check difference disjointness: new diffs must not intersect other set's diffs
                conflict = False
                for a in S:
                    d1 = x - a
                    d2 = a - x
                    if d1 in other_diffs or d2 in other_diffs:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Accept x into sets[side]
                sets[side].append(x)
                for a in S:
                    set_sums[side].add(a + x)
                    set_diffs[side].add(x - a)
                    set_diffs[side].add(a - x)
                set_sums[side].add(x + x)
                break  # Only place in one set per candidate
        
        return sets
    
    def _local_search(sets, deadline):
        """
        Try to improve by removing elements that block many additions,
        then re-adding elements in a smart order.
        """
        sets = [list(sets[0]), list(sets[1])]
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for side in (0, 1):
                for idx in range(len(sets[side])):
                    # Try removing element at idx
                    removed = sets[side].pop(idx)
                    # Rebuild state
                    sets_after_removal = [list(sets[0]), list(sets[1])]
                    # Try to add back different elements
                    order = [x for x in base_order if x not in sets_after_removal[0] and x not in sets_after_removal[1]]
                    random.shuffle(order)
                    new_sets = _greedy_one_ordering(order, deadline)
                    new_size = len(new_sets[0]) + len(new_sets[1])
                    current_size = len(sets[0]) + len(sets[1])
                    if new_size > current_size:
                        sets = [list(new_sets[0]), list(new_sets[1])]
                        improved = True
                        break
                    else:
                        sets[side].insert(idx, removed)
        return sets
    
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
    
    # If we have time, try local search on the best solution found
    if time.time() < deadline - 10.0:
        improved_sets = _local_search(best_sets, deadline)
        size = len(improved_sets[0]) + len(improved_sets[1])
        if size > best_size:
            best_size = size
            best_sets = (list(improved_sets[0]), list(improved_sets[1]))
    
    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets = ([1]) + list(best_sets[1])
        best_sets = ([1], best_sets[1])
    elif not best_sets[1]:
        best_sets = list(best_sets[0]) + ([1],)
        best_sets = (best_sets[0], [1])
    
    return best_sets
# EVOLVE-BLOCK-END
