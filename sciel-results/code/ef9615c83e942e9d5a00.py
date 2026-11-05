import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    n = 256
    max_per_set = 64
    
    # Precompute all pairs for difference checks
    # We'll use a greedy approach with backtracking-lite or randomized restarts
    
    def get_diffs(S):
        """Get set of nonzero differences for set S."""
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[i] - S[j])
                diffs.add(S[j] - S[i])
        return diffs
    
    def get_sums(S):
        """Get set of pair sums (including a+a) for set S."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums
    
    def is_valid_addition(x, S, S_sums, S_diffs, other_diffs):
        """Check if x can be added to S maintaining Sidon property and disjoint diffs."""
        # Check Sidon: all pair sums including x+x must be unique
        new_sums = set(S_sums)
        for a in S:
            s = x + a
            if s in new_sums:
                return False
            new_sums.add(s)
        # Check x+x
        if 2 * x in new_sums:
            return False
        new_sums.add(2 * x)
        
        # Check differences don't conflict with other set
        for a in S:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
        
        return True
    
    def greedy_construct(order, max_time=None):
        """Greedy construction given a specific element order."""
        sets = [[], []]
        sets_sums = [set(), set()]
        sets_diffs = [set(), set()]
        
        for x in order:
            if max_time is not None and time.time() > max_time:
                break
            best_side = -1
            for side in (0, 1):
                if is_valid_addition(x, sets[side], sets_sums[side], sets_diffs[side], sets_diffs[1 - side]):
                    best_side = side
                    break
            if best_side >= 0:
                x_side = best_side
                # Update sums
                for a in sets[x_side]:
                    sets_sums[x_side].add(x + a)
                sets_sums[x_side].add(2 * x)
                # Update diffs
                for a in sets[x_side]:
                    sets_diffs[x_side].add(x - a)
                    sets_diffs[x_side].add(a - x)
                sets[x_side].append(x)
        
        return sets
    
    def validate(sets):
        """Validate that both sets are Sidon and their nonzero differences are disjoint."""
        if not sets[0] or not sets[1]:
            return False
        if len(sets[0]) > max_per_set or len(sets[1]) > max_per_set:
            return False
        
        # Check Sidon property for each set
        for S in sets:
            sums = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    s = S[i] + S[j]
                    if s in sums:
                        return False
                    sums.add(s)
        
        # Check disjoint nonzero differences
        diffs0 = get_diffs(sets[0])
        diffs1 = get_diffs(sets[1])
        if diffs0 & diffs1:
            return False
        
        return True
    
    # Strategy: Try multiple random orderings and keep the best valid result
    best_sets = [[], []]
    best_size = 0
    
    # First, try a deterministic greedy with natural order
    order = list(range(1, n + 1))
    sets = greedy_construct(order, deadline - 5)
    if validate(sets):
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = [list(sets[0]), list(sets[1])]
    
    # Try reverse order
    order = list(range(n, 0, -1))
    sets = greedy_construct(order, deadline - 5)
    if validate(sets):
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = [list(sets[0]), list(sets[1])]
    
    # Try several random orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 10:
        order = list(range(1, n + 1))
        random.shuffle(order)
        sets = greedy_construct(order, deadline - 10)
        if validate(sets):
            size = len(sets[0]) + len(sets[1])
            if size > best_size:
                best_size = size
                best_sets = [list(sets[0]), list(sets[1])]
        attempts += 1
        if attempts % 50 == 0 and time.time() > deadline - 15:
            break
    
    # If we didn't find a good solution, use a simple fallback
    if not best_sets[0] or not best_sets[1]:
        # Simple construction: use odd numbers for set A, even for set B (but check Sidon)
        # Actually, let's just do a basic greedy
        sets = greedy_construct(list(range(1, n + 1)))
        if validate(sets):
            best_sets = [list(sets[0]), list(sets[1])]
        else:
            # Fallback: minimal valid sets
            best_sets = [[1], [2]]
    
    return best_sets
