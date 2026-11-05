"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from functools import lru_cache

def construct():
    """
    Returns a strictly increasing sequence of integers in [1,256] such that
    sums of all nonempty consecutive blocks are distinct. Maximizes sequence length.
    
    Approach: Greedy with backtracking via DFS. We use prefix sums to efficiently
    check for collisions. The state is the set of existing block sums.
    We try to build the longest possible sequence.
    
    Key insight: The greedy approach (pick smallest valid number) is often optimal or near-optimal
    for this type of problem, but backtracking can find longer sequences if greedy gets stuck
    early. However, full backtracking is expensive. We use a hybrid:
    1. Try greedy first.
    2. If greedy fails to extend, backtrack to the last few steps and try larger numbers.
    
    Actually, let's use a more robust DFS with memoization on the "suffix sums" state
    to prune branches. But the state space is huge.
    
    Alternative: The greedy algorithm provided in the seed is quite strong. Let's analyze why.
    At each step, we pick the smallest x such that no new consecutive sum collides.
    This is a standard greedy for "distinct subset sums" variants. For consecutive blocks,
    it's easier.
    
    Let's try to improve by using a smarter search that looks ahead.
    We can use a depth-first search that prioritizes smaller numbers but allows
    backtracking if a dead end is reached.
    
    To make this fast enough (60s), we need good pruning.
    Pruning idea: If the current set of sums is S, and we have prefix sums P,
    adding a new number x creates new sums: x, x+P[-1], x+P[-2], ...
    If any of these is in S, we skip x.
    
    Let's implement a DFS that tries to maximize length.
    We'll keep the best solution found.
    
    State: (current_sequence, set_of_sums)
    Since set_of_sums is large, we can't memoize easily.
    However, the structure of the problem (strictly increasing, bounded by 256)
    means the search space is limited.
    
    Let's try a recursive backtracking search that tries numbers in increasing order.
    We stop when we can't add any more numbers.
    We keep the longest sequence found.
    
    Optimization: 
    - Use a bitset or boolean array for sums (max sum = 256*256/2 ~ 32768, actually max sum is 1+...+256 = 32896).
    - A set of ints is fine in Python if we are careful.
    
    Let's refine the greedy approach first. The seed program does:
    for x in 1..256:
        if x is valid, add it.
    
    This is O(N^2) where N is the length of the sequence.
    Can we do better?
    
    Let's try a DFS that branches on the next number.
    We want to maximize the length.
    
    def dfs(start_val, current_seq, sums_set):
        best_len = len(current_seq)
        for x in range(start_val, 257):
            # Check if x is valid
            new_sums = set()
            # Calculate new consecutive sums ending at x
            # They are: x, x + last, x + last + second_last, ...
            # We can compute these incrementally.
            
            # If valid, recurse
            new_sums_set = sums_set | new_sums
            # But we need to pass the new sums to the next level.
            # To avoid copying large sets, we can use a global set and undo (backtrack).
            
            # Let's use a global set and modify it.
            
    This might be too slow if the branching factor is high.
    However, the constraint is tight: only ~30k possible sums.
    The sequence length is likely around 10-20.
    
    Let's estimate the max length.
    Number of consecutive blocks of length k in a sequence of length n is n-k+1.
    Total blocks = n(n+1)/2.
    Max sum = 256*256 = 65536 (actually 32896).
    So n(n+1)/2 <= 32896 => n^2 ~ 65000 => n ~ 255.
    But the values are distinct and in [1,256].
    The sums must be distinct.
    
    The greedy algorithm in the seed produces a specific length. Let's run it mentally or assume it's good.
    The prompt says the seed scores 1.0. We need to beat it.
    
    Let's try to implement the DFS with backtracking.
    
    Global variables for speed:
    - sums_present: set of int
    - current_seq: list of int
    
    def try_add(x):
        # Check if x can be added
        # New sums are: x, x + seq[-1], x + seq[-1] + seq[-2], ...
        # We need to check if any of these are in sums_present.
        # Also, we need to check if any of these new sums collide with each other?
        # No, because the sequence is strictly increasing, the suffix sums are strictly increasing.
        # So x, x+last, x+last+second_last are all distinct.
        # We only need to check against existing sums.
        
        s = 0
        for a in reversed(current_seq + [x]):
            s += a
            if s in sums_present:
                return False
        return True
    
    Wait, the new sums are the suffix sums of the new sequence.
    If we append x to current_seq, the new consecutive blocks ending at x are:
    [x], [x, seq[-1]], [x, seq[-1], seq[-2]], ...
    Their sums are:
    x
    x + seq[-1]
    x + seq[-1] + seq[-2]
    ...
    x + sum(seq)
    
    We need to ensure none of these are in sums_present.
    Note: sums_present contains all sums of all consecutive blocks of current_seq.
    
    So:
    def is_valid(x):
        s = x
        if s in sums_present: return False
        for a in reversed(current_seq):
            s += a
            if s in sums_present: return False
        return True
    
    Then in DFS:
    for x in range(start, 257):
        if is_valid(x):
            # Add x
            current_seq.append(x)
            # Update sums_present
            new_sums = []
            s = x
            new_sums.append(s)
            for a in reversed(current_seq[:-1]): # current_seq[:-1] is the old seq
                s += a
                new_sums.append(s)
            sums_present.update(new_sums)
            
            # Recurse
            dfs(x+1)
            
            # Undo
            for ns in new_sums:
                sums_present.remove(ns)
            current_seq.pop()
    
    We need to keep track of the best solution found.
    
    This DFS might explore many branches.
    To prune, we can use the heuristic that picking smaller x is better (allows more room for future numbers).
    So we try x in increasing order.
    
    To speed up, we can add a simple bound:
    If len(current_seq) + (256 - start + 1) <= best_len, return.
    
    Let's implement this.
    
    One optimization: The set of sums can be large. Using a set of ints is okay.
    
    Let's also add a timeout check or just hope it's fast enough.
    The max depth is ~15-20. Branching factor is limited by validity.
    
    Let's try this approach.
    
    """
    global best_seq
    best_seq = []
    sums_present = set()
    current_seq = []
    
    # Precompute max possible length?
    # n(n+1)/2 <= 32896 => n <= 255. But values are 1..256.
    
    def is_valid(x):
        s = x
        if s in sums_present:
            return False
        for a in reversed(current_seq):
            s += a
            if s in sums_present:
                return False
        return True
    
    def dfs(start):
        global best_seq
        # Pruning: if even taking all remaining numbers doesn't beat best, return
        if len(current_seq) + (256 - start + 1) <= len(best_seq):
            return
        
        if len(current_seq) > len(best_seq):
            best_seq = current_seq[:]
        
        for x in range(start, 257):
            if is_valid(x):
                # Add x
                current_seq.append(x)
                
                # Calculate new sums to add to sums_present
                new_sums = []
                s = x
                new_sums.append(s)
                # We need to add sums of blocks ending at x
                # These are x, x+last, x+last+second_last...
                # We already checked them in is_valid, so they are not in sums_present.
                for a in reversed(current_seq[:-1]):
                    s += a
                    new_sums.append(s)
                
                sums_present.update(new_sums)
                
                dfs(x + 1)
                
                # Undo
                for ns in new_sums:
                    sums_present.remove(ns)
                current_seq.pop()
    
    dfs(1)
    return best_seq

# EVOLVE-BLOCK-END
