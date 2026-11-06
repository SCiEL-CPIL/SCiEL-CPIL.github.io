import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    best_set = []
    best_size = 0
    
    def greedy_greedy(order):
        reachable = [False] * 201
        reachable[0] = True
        selected = []
        for x in order:
            if reachable[200 - x]:
                continue
            selected.append(x)
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
        return selected

    # 1. Descending
    order_desc = list(range(100, 0, -1))
    current_set = greedy_greedy(order_desc)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set

    # 2. Ascending
    order_asc = list(range(1, 101))
    current_set = greedy_greedy(order_asc)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set

    # 3. Randomized Descending (Perturbed Descending)
    while time.time() < deadline:
        # Start with descending
        order = list(range(100, 0, -1))
        # Perform a few random swaps
        for _ in range(10): # 10 swaps
            i = random.randint(0, 99)
            j = random.randint(0, 99)
            order[i], order[j] = order[j], order[i]
        
        current_set = greedy_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

        # Also try pure random shuffle occasionally
        if random.random() < 0.1:
            order = list(range(1, 101))
            random.shuffle(order)
            current_set = greedy_greedy(order)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set

    return best_set
