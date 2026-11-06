import time
import random

def construct():
    # 1. Generate initial solution using the seed's greedy strategy
    selected = set()
    reachable = {0}
    
    # Greedy descending
    for x in range(100, 0, -1):
        new_reachable = set()
        # Check if adding x creates 200
        # s + x == 200 => s == 200 - x
        if 200 - x in reachable:
            continue
        # If not, add x
        selected.add(x)
        # Update reachable
        for s in reachable:
            ns = s + x
            if ns <= 200:
                new_reachable.add(ns)
        reachable.update(new_reachable)
        
    # 2. Local Search / Hill Climbing
    # We want to maximize len(selected)
    # Moves: 
    # a) Add an element not in selected
    # b) Remove an element from selected
    # c) Swap: Remove one, add one
    
    best_set = set(selected)
    best_reachable = set(reachable)
    best_size = len(selected)
    
    # Precompute potential additions
    # We iterate a few times to allow chains of improvements
    
    def get_reachable(s_set):
        r = {0}
        for x in s_set:
            new_r = {s + x for s in r if s + x <= 200}
            if 200 in new_r:
                return None # Invalid
            r.update(new_r)
        return r

    # Optimization: Update reachable incrementally is complex for removals.
    # Given N=100, recalculating reachable for a set of size ~50 is fast.
    # 2^50 is huge, but the set of sums <= 200 is at most 201 elements.
    # So reachable set size is <= 201.
    # Updating reachable: O(|S| * |Reachable|) ~ 50 * 200 = 10,000 ops.
    # Very fast.
    
    def is_valid(s_set):
        r = {0}
        for x in s_set:
            new_r = {s + x for s in r if s + x <= 200}
            if 200 in new_r:
                return False
            r.update(new_r)
        return True

    current_set = set(selected)
    
    # Randomize start slightly? No, stick to seed + improvement.
    
    improved = True
    while improved:
        improved = False
        # Try adding elements
        for x in range(1, 101):
            if x in current_set:
                continue
            new_set = current_set | {x}
            if is_valid(new_set):
                current_set = new_set
                improved = True
                break # Restart loop with new set
        
        # Try swapping: remove y, add x
        # This is O(N^2) * validity_check. 100*100*10000 = 10^8. Might be slow in Python.
        # Optimize: Only try swaps if we can't add directly.
        if not improved:
             # Try single removals? No, size decreases.
             # Try swaps.
             for y in list(current_set):
                 for x in range(1, 101):
                     if x in current_set or x == y:
                         continue
                     new_set = (current_set - {y}) | {x}
                     if is_valid(new_set):
                         # Check if this swap allows future additions?
                         # For now, just accept if valid.
                         # But swap keeps size same. We want to increase size.
                         # So only swap if it enables an addition?
                         # This requires lookahead.
                         # Simpler: Just do swaps to escape local minima, then try additions again.
                         current_set = new_set
                         improved = True
                         break
                 if improved:
                     break
                     
    return list(current_set)

