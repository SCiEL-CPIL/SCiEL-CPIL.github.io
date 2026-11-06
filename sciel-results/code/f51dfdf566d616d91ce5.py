import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    best_set = []
    best_size = 0
    
    # Initial run with descending order (seed behavior)
    def greedy_greedy(order):
        reachable = [False] * 201
        reachable[0] = True
        selected = []
        for x in order:
            # Check if adding x creates sum 200
            # new_reachable[s] = reachable[s] or reachable[s-x]
            # We only care if 200 becomes reachable
            # 200 is reachable if reachable[200] (already true? no, we stop if it becomes true)
            # or reachable[200-x] is true
            
            if reachable[200 - x]:
                # Adding x would make 200 reachable
                # So we skip x
                continue
            
            # Safe to add x
            selected.append(x)
            # Update reachable
            # We need to update reachable for future checks
            # We can do this efficiently
            # new_reachable = reachable.copy()
            # for s in range(200 - x, -1, -1):
            #     if reachable[s]:
            #         new_reachable[s+x] = True
            # reachable = new_reachable
            
            # Optimization: Only update if we are going to use it.
            # Since we skip x if it's bad, we only update for good x.
            # We can use a list of booleans.
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
        
        return selected

    # Try the seed order first
    order = list(range(100, 0, -1))
    current_set = greedy_greedy(order)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set

    # Randomized restarts
    while time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        current_set = greedy_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # Optional: Try some heuristic orders
        # e.g., random swap of a few elements in the descending order

    return best_set
