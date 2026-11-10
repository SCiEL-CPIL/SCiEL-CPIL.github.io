import time
import random

def construct():
    start_time = time.time()
    N = 512
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Map element to list of AP indices
    elem_to_aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(aps):
        for x in ap:
            elem_to_aps[x].append(i)
            
    # Precompute degree of each element
    degree = [len(elem_to_aps[i]) for i in range(N + 1)]

    # State
    in_set = [False] * (N + 1)
    ap_count = [0] * len(aps)
    
    def add(x):
        in_set[x] = True
        for idx in elem_to_aps[x]:
            ap_count[idx] += 1

    def remove(x):
        in_set[x] = False
        for idx in elem_to_aps[x]:
            ap_count[idx] -= 1

    def is_safe(x):
        for idx in elem_to_aps[x]:
            if ap_count[idx] == 3:
                return False
        return True

    def get_size():
        return sum(in_set)

    def run_fast_local_search(initial_order=None, max_iters=300):
        for i in range(1, N + 1):
            in_set[i] = False
        for i in range(len(aps)):
            ap_count[i] = 0
            
        if initial_order is None:
            candidates = list(range(1, N + 1))
        else:
            candidates = initial_order
            
        # Greedy Init
        for x in candidates:
            if is_safe(x):
                add(x)
                
        current_size = get_size()
        
        # Local Search
        for _ in range(max_iters):
            if time.time() - start_time > 85:
                break
                
            # Try to add any safe element
            added = False
            # Sort by degree? Or just iterate.
            # Iterating 1..N is fine.
            for x in range(1, N + 1):
                if not in_set[x] and is_safe(x):
                    add(x)
                    current_size += 1
                    added = True
                    break
            
            if added:
                continue
                
            # Stuck: Remove an element
            if current_size == 0:
                break
                
            # Find element in S with highest degree
            # To speed up, we can just pick the one with max degree from S
            # Since N is small, finding max is O(N)
            max_deg = -1
            best_y = -1
            for y in range(1, N + 1):
                if in_set[y] and degree[y] > max_deg:
                    max_deg = degree[y]
                    best_y = y
                    
            if best_y != -1:
                remove(best_y)
                current_size -= 1
            else:
                break
                
        return current_size, [i for i in range(1, N + 1) if in_set[i]]

    best_size = 0
    best_set = []
    
    # Run 1: Deterministic
    size, s = run_fast_local_search()
    if size > best_size:
        best_size = size
        best_set = s
        
    # Run multiple random restarts
    while time.time() - start_time < 80:
        order = list(range(1, N + 1))
        random.shuffle(order)
        size, s = run_fast_local_search(order)
        if size > best_size:
            best_size = size
            best_set = s

    return best_set

# EVOLVE-BLOCK-END
