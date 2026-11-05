"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 55.0  # Safety margin before 90s limit
    best_sets = ([], [])
    best_size = 0
    base_order = list(range(1, 257))

    def _can_add(x, S, set_sums, set_diffs, other_diffs):
        """Check if x can be added to set S with given sums/diffs."""
        # Check Sidon condition: new pair sums a+x and x+x must be unique
        for a in S:
            if (a + x) in set_sums:
                return False
        if (x + x) in set_sums:
            return False
        
        # Check difference disjointness: new diffs must not intersect other set's diffs
        for a in S:
            if (x - a) in other_diffs or (a - x) in other_diffs:
                return False
        return True

    def _add_element(x, side, sets, set_sums, set_diffs):
        """Add element x to sets[side] and update sums/diffs."""
        S = sets[side]
        for a in S:
            set_sums[side].add(a + x)
            set_diffs[side].add(x - a)
            set_diffs[side].add(a - x)
        set_sums[side].add(x + x)
        sets[side].append(x)

    def _greedy_one_ordering(order, deadline, strategy="first_fit"):
        """
        Run greedy construction with a given ordering of candidates.
        strategy: "first_fit" (try side 0 then 1), "best_fit" (try side with fewer conflicts first),
                  "random" (randomly choose side)
        """
        sets = [[], []]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Determine order of sides to try
            if strategy == "first_fit":
                sides = [0, 1]
            elif strategy == "best_fit":
                # Prefer the side with fewer potential conflicts
                conflicts = []
                for side in (0, 1):
                    S = sets[side]
                    other_diffs = set_diffs[1 - side]
                    conf = 0
                    for a in S:
                        if (a + x) in set_sums[side] or (x - a) in other_diffs or (a - x) in other_diffs:
                            conf += 1
                    conflicts.append(conf)
                sides = [0, 1] if conflicts[0] <= conflicts[1] else [1, 0]
            else:  # random
                sides = [0, 1]
                random.shuffle(sides)
            
            for side in sides:
                S = sets[side]
                other_diffs = set_diffs[1 - side]
                
                if not _can_add(x, S, set_sums[side], set_diffs[side], other_diffs):
                    continue
                
                _add_element(x, side, sets, set_sums, set_diffs)
                break  # Only place in one set per candidate
        
        return sets

    def _local_search(sets, deadline, max_iter=500):
        """
        Local search: try to add more elements or swap elements to improve size.
        Returns improved sets.
        """
        sets = [list(sets[0]), list(sets[1])]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        # Rebuild sums and diffs
        for side in (0, 1):
            S = sets[side]
            for i in range(len(S)):
                for j in range(i, len(S)):
                    set_sums[side].add(S[i] + S[j])
            for i in range(len(S)):
                for j in range(len(S)):
                    if i != j:
                        set_diffs[side].add(S[j] - S[i])
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try to add any unused element to either set
            used = set(sets[0]) | set(sets[1])
            for x in base_order:
                if x in used:
                    continue
                
                for side in (0, 1):
                    other_diffs = set_diffs[1 - side]
                    if _can_add(x, sets[side], set_sums[side], set_diffs[side], other_diffs):
                        _add_element(x, side, sets, set_sums, set_diffs)
                        improved = True
                        break
                if improved:
                    break
            
            # Try swaps: remove one element and add another
            if not improved:
                for side in (0, 1):
                    for i, elem in enumerate(sets[side]):
                        # Remove elem
                        S = sets[side]
                        # Rebuild sums and diffs for this set
                        new_sums = set()
                        new_diffs = set()
                        remaining = S[:i] + S[i+1:]
                        for a in remaining:
                            for b in remaining:
                                if a <= b:
                                    new_sums.add(a + b)
                            new_diffs.add(a - b)
                            new_diffs.add(b - a)
                        new_diffs.discard(0)
                        
                        # Try adding each unused element
                        used = set(sets[0]) | set(sets[1])
                        used.discard(elem)
                        for x in base_order:
                            if x in used or x == elem:
                                continue
                            other_diffs = set_diffs[1 - side]
                            if _can_add(x, remaining, new_sums, new_diffs, other_diffs):
                                # Accept swap
                                sets[side] = remaining + [x]
                                set_sums[side] = new_sums
                                set_diffs[side] = new_diffs
                                for a in remaining:
                                    set_sums[side].add(a + x)
                                    set_diffs[side].add(x - a)
                                    set_diffs[side].add(a - x)
                                set_sums[side].add(x + x)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return sets

    # Phase 1: Try deterministic orderings
    strategies = ["first_fit", "best_fit"]
    for strategy in strategies:
        sets = _greedy_one_ordering(base_order, deadline, strategy)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = (list(sets[0]), list(sets[1]))
        
        # Try descending order
        sets = _greedy_one_ordering(base_order[::-1], deadline, strategy)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = (list(sets[0]), list(sets[1]))
    
    # Phase 2: Randomized restarts with local search
    random.seed(42)
    while time.time() < deadline - 5.0:
        random_order = base_order[:]
        random.shuffle(random_order)
        
        # Try different strategies
        for strategy in ["first_fit", "best_fit", "random"]:
            if time.time() > deadline - 3.0:
                break
            
            sets = _greedy_one_ordering(random_order, deadline, strategy)
            
            # Apply local search to improve
            if time.time() < deadline - 2.0:
                sets = _local_search(sets, deadline, max_iter=200)
            
            size = len(sets[0]) + len(sets[1])
            if size > best_size:
                best_size = size
                best_sets = (list(sets[0]), list(sets[1]))
    
    # Phase 3: Final local search on best result
    if time.time() < deadline - 1.0 and best_size > 0:
        improved = _local_search(best_sets, deadline, max_iter=500)
        size = len(improved[0]) + len(improved[1])
        if size > best_size:
            best_size = size
            best_sets = (list(improved[0]), list(improved[1]))
    
    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets[0] = [1]
    elif not best_sets[1]:
        best_sets[1] = [1]
    
    return list(best_sets)
# EVOLVE-BLOCK-END
