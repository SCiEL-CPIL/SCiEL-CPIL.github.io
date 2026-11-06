import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_set = []
    best_size = 0
    
    # Initial greedy pass to get a baseline
    current_set, current_counts = [], {}
    for x in range(1, 1001):
        can_add = True
        for y in current_set:
            s = x + y
            if current_counts.get(s, 0) >= 2:
                can_add = False
                break
        if can_add:
            current_set.append(x)
            for y in current_set:
                s = x + y
                current_counts[s] = current_counts.get(s, 0) + 1
            # Note: the loop above updates counts for all pairs (x, y) where y in current_set.
            # Wait, current_set already contains x at this point?
            # No, I appended x before the loop.
            # Let's re-verify the update logic.
            # If I append x first, then `for y in current_set` includes x.
            # So it adds x+x=2x, and x+y for all y<x.
            # And for y>x, it adds y+x.
            # This correctly adds 1 to the count of each sum x+y for all y in the new set.
            # Wait, the check was done BEFORE adding x.
            # So current_counts reflects the state BEFORE x is added.
            # The check `current_counts.get(s, 0) >= 2` ensures that the new count will be <= 2.
            # Then we update counts.
            
    best_set = current_set
    best_size = len(current_set)
    
    # Randomized restarts
    while time.time() < deadline:
        random.shuffle_order = list(range(1, 1001))
        random.shuffle(random.shuffle_order)
        
        current_set, current_counts = [], {}
        for x in random.shuffle_order:
            can_add = True
            for y in current_set:
                s = x + y
                if current_counts.get(s, 0) >= 2:
                    can_add = False
                    break
            if can_add:
                current_set.append(x)
                for y in current_set:
                    s = x + y
                    current_counts[s] = current_counts.get(s, 0) + 1
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    return best_set
