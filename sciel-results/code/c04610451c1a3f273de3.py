import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Candidate pool
    candidates = list(range(1, 257))
    
    best_set = []
    
    # Try multiple random restarts or different orderings
    # Strategy 1: Greedy from large to small
    # Strategy 2: Randomized Greedy
    
    def is_valid_with_x(S_set, x):
        """
        Check if adding x to set S (represented by subset sums S_set) results in a valid set.
        S_set contains all non-empty subset sums of the current set of elements.
        """
        # New sums are S_set U {s + x for s in S_set} U {x}
        # Note: if x is already in S_set? No, x is a new element, not a sum necessarily.
        # But x is a subset sum of {x}.
        
        # Generate new sums
        new_sums = set()
        for s in S_set:
            new_sums.add(s + x)
        new_sums.add(x)
        
        # We need to check divisibility between:
        # 1. New sums and old sums (S_set)
        # 2. New sums and new sums
        
        # Optimization: 
        # For each y in new_sums:
        #   Check if any z in S_set divides y or y divides z.
        #   Check if any y' in new_sums (y' != y) divides y or y divides y'.
        
        # This is O(|S| * |New| + |New|^2)
        # |S| can be up to 2^k. For k=10, 1000. |New| ~ 1000. 10^6 ops.
        # For k=15, 30000. 9*10^8 ops. Too slow.
        
        # We need a faster check or smaller k.
        # Let's assume we can only go up to k=12-15 with this method if optimized in C, but in Python it's slow.
        
        # Let's try a different approach: 
        # Instead of checking all pairs, we can check if y is a multiple of any z in S_set.
        # This is O(|S_set|) per y.
        # The pairwise check within new_sums is the killer.
        
        # Can we avoid the pairwise check within new_sums?
        # If S_set is valid, does that imply something about S_set + x?
        # Not directly.
        
        # Let's just implement the naive check but with early termination and hope that 
        # for valid sets, the conflicts are found early or the sets are small enough.
        
        # Actually, we can sort S_set.
        # For a new sum y:
        # - If y < min(S_set), it can only divide elements in S_set. Check z % y == 0.
        # - If y > max(S_set), it can only be divided by elements in S_set. Check y % z == 0.
        # - If min(S_set) <= y <= max(S_set), check both.
        
        # For new sums among themselves:
        # We can collect all new sums, sort them, and check adjacent/multiples?
        # No, must check all pairs.
        
        # Let's try to limit the search depth.
        # If the check is too slow, we reduce the number of candidates or restarts.
        
        pass # Placeholder

    # Let's implement a simpler, faster heuristic check first.
    # Maybe we can use a bitset or numpy?
    # No, standard library is fine.
    
    # Revised Plan:
    # 1. Greedy construction.
    # 2. At each step, try candidates in random order (or sorted by value).
    # 3. Use a fast validity check.
    
    def check_validity(S_set, new_sums):
        """
        S_set: set of existing sums.
        new_sums: set of new sums to add.
        Returns True if S_set U new_sums is valid.
        """
        # Check new_sums against S_set
        for y in new_sums:
            # y divides z in S_set?
            for z in S_set:
                if z > y and z % y == 0:
                    return False
                # z divides y?
                if z < y and y % z == 0:
                    return False
                # if z == y, it's a duplicate sum, which is invalid (two subsets same sum)
                if z == y:
                    return False
            
            # Check new_sums against new_sums
            # This is O(N^2). 
            # We can do this more efficiently by sorting new_sums and checking multiples.
            pass
        
        # Check within new_sums
        # Sort new_sums
        ns_list = sorted(new_sums)
        n = len(ns_list)
        for i in range(n):
            y = ns_list[i]
            # Check if y divides any later element
            # Multiples of y: 2y, 3y, ...
            # We can just iterate j > i
            for j in range(i+1, n):
                if ns_list[j] % y == 0:
                    return False
        return True

    # This check is still O(N^2) for the new sums part.
    # But N is the number of new sums, which is ~2^k.
    # For k=10, N=1000, N^2=10^6. OK.
    # For k=12, N=4000, N^2=1.6*10^7. OK.
    # For k=14, N=16000, N^2=2.5*10^8. Slow in Python.
    
    # So we are limited to k ~ 12-13 with this direct check.
    # To go higher, we need a smarter check or a different algorithm.
    
    # Let's try to optimize the "new_sums against S_set" part.
    # If we store S_set in a list, we can iterate.
    
    # Let's implement the greedy with this check.
    
    def try_greedy(order_func, time_limit):
        current_elements = []
        current_sums = set()
        
        candidates = list(range(1, 257))
        candidates.sort(key=order_func)
        
        for x in candidates:
            if time.time() > time_limit:
                break
            
            # Generate new sums
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + x)
            new_sums.add(x)
            
            # Check validity
            if check_validity(current_sums, new_sums):
                current_elements.append(x)
                current_sums.update(new_sums)
        
        return current_elements

    # Try different orderings
    # 1. Descending (large numbers first)
    best = []
    
    # Run a few times with random shuffles to escape local optima
    for i in range(50):
        if time.time() > deadline:
            break
        random.shuffle(candidates)
        # Maybe sort by some heuristic?
        # Let's just use random order
        res = try_greedy(lambda x: random.random(), deadline)
        if len(res) > len(best):
            best = res
    
    return best
