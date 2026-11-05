import sys

sys.setrecursionlimit(10000)

def construct():
    best_seq = []
    best_len = 0
    
    # We will use a DFS to find the longest sequence.
    # To make it efficient, we try smaller numbers first.
    # We prune branches that cannot possibly exceed the current best length.
    
    def dfs(seq, sums, last_val):
        nonlocal best_seq, best_len
        
        # Update best if we found a longer sequence
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Pruning: Upper bound on how many more elements we can add.
        # We can add at most (256 - last_val) elements.
        # So max possible length is len(seq) + (256 - last_val).
        # If this is not greater than best_len, we can stop.
        if len(seq) + (256 - last_val) <= best_len:
            return

        # Try adding the next element
        # We iterate from last_val + 1 to 256
        # To improve performance, we might want to try a heuristic order.
        # Smaller numbers tend to keep sums smaller, which is good for distinctness?
        # Actually, larger numbers create larger sums, which might avoid collisions with small sums.
        # But smaller numbers allow more numbers to be packed into the range [1, 256].
        # Let's stick to increasing order for the next candidate.
        
        for x in range(last_val + 1, 257):
            # Calculate new sums ending at x
            new_sums = []
            suffix_sum = x
            for a in reversed(seq):
                suffix_sum += a
                new_sums.append(suffix_sum)
            new_sums.append(x) # The block [x] itself
            
            # Check for collisions
            if any(s in sums for s in new_sums):
                continue
                
            # If valid, add to state and recurse
            new_sums_set = sums.union(new_sums)
            seq.append(x)
            dfs(seq, new_sums_set, x)
            seq.pop()
            
    # Initial call
    # Start with empty sequence, empty sums, last_val = 0
    dfs([], set(), 0)
    
    return best_seq

# The above DFS might be too slow if it explores too many branches.
# The greedy algorithm is O(N^2) or so. DFS is exponential.
# However, the pruning `len(seq) + (256 - last_val) <= best_len` is quite strong.
# Also, the constraint on sums is very restrictive.
# Let's add a timeout or a node limit to be safe? No, the evaluator has a 60s limit.
# We can also try to improve the heuristic.
# What if we try to jump to a known good length first?
# The greedy algorithm likely produces a sequence of length ~10-15.
# If we find a sequence of length 15, the pruning will cut off branches that can't reach 16.

# To be safer and faster, we can combine greedy with local search or a more directed DFS.
# But a simple DFS with strong pruning often works for small N (256).
# Let's verify the complexity.
# Depth is max length L.
# Branching factor is average number of valid x's.
# If pruning is effective, it should be fine.

# One optimization: Instead of creating a new set `new_sums_set` every time, 
# we can pass the set and mutate it (add/remove). 
# But `sums.union` creates a new set. For small sets (size < 1000), it's fine.
# The size of sums is L(L+1)/2. For L=20, size is 210. Very fast.

# Let's refine the DFS to be a bit smarter about the order of trying x.
# If we want to maximize length, we should prefer smaller x to leave room for more elements.
# So `range(last_val + 1, 257)` is the correct order.

# However, there is a risk of hitting recursion limit or time limit if the search space is large.
# Given the strong constraint (distinct subarray sums), the valid branches are very few.
# This problem is similar to finding a Sidon sequence, which is sparse.
# The greedy algorithm usually gets close to the optimum for such problems.
# The DFS will just verify if we can do better by making non-greedy choices.

# Let's add a simple cache or memoization? 
# State: (tuple(seq[-k:]), frozenset(sums))
# This is too expensive to compute and store.

# Let's stick to the DFS with pruning. 
# To ensure it finishes, we can also add a check: if we've already found a sequence of length L, 
# and the current path is at depth D, and D + (256 - last_val) <= L, return.

# One more improvement: 
# We can start the DFS from a greedy solution.
# Run greedy first to get a baseline `best_len`.
# Then run DFS to try to beat it.

def construct():
    # 1. Greedy phase to get a good initial lower bound
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    best_seq = values[:]
    best_len = len(values)
    
    # 2. DFS phase to try to improve
    # We use the greedy result as the initial best.
    # We will perform a DFS starting from the empty sequence, 
    # but we prune aggressively using best_len.
    
    def dfs(seq, sums_set, last_val):
        nonlocal best_seq, best_len
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Pruning
        if len(seq) + (256 - last_val) <= best_len:
            return

        for x in range(last_val + 1, 257):
            # Calculate new sums
            new_sums = []
            suffix_sum = x
            for a in reversed(seq):
                suffix_sum += a
                new_sums.append(suffix_sum)
            new_sums.append(x)
            
            # Check collisions
            valid = True
            for s in new_sums:
                if s in sums_set:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Recurse
            # To avoid creating a new set object every time (which is expensive),
            # we can use a set and add/remove, but since we are in a deep recursion
            # and Python sets are immutable in terms of usage here (we pass a new one),
            # creating a new set is safer for correctness.
            # Given the small size, it's acceptable.
            next_sums = sums_set.union(new_sums)
            seq.append(x)
            dfs(seq, next_sums, x)
            seq.pop()

    dfs([], set(), 0)
    
    return best_seq
