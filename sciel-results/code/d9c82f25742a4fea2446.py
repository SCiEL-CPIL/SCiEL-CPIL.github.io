import random
import time

# Precompute all 4-APs in [1, 512]
N = 512
all_4aps = []
aps_containing = {i: [] for i in range(1, N + 1)}

for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        ap = (a, a + d, a + 2 * d, a + 3 * d)
        all_4aps.append(ap)
        for x in ap:
            aps_containing[x].append(ap)

def greedy_random():
    S = set()
    order = list(range(1, N + 1))
    random.shuffle(order)
    for x in order:
        can_add = True
        for ap in aps_containing[x]:
            # Check if the other 3 elements are in S
            # ap is a tuple of 4 distinct ints
            # We need to check if {a,b,c,d} - {x} is a subset of S
            # Since ap has 4 elements, and x is one of them, there are 3 others.
            # Manual check for speed:
            # Find the 3 others
            # Or just:
            # if all(y in S for y in ap if y != x):
            #     can_add = False
            #     break
            # To avoid generator overhead, unroll?
            # The list comprehension is fine for small lists.
            if all(y in S for y in ap if y != x):
                can_add = False
                break
        if can_add:
            S.add(x)
    return S

def construct():
    best_set = set()
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # Initial run
    current_best = greedy_random()
    best_set = current_best
    
    # Loop with restarts
    while time.time() < deadline:
        candidate = greedy_random()
        if len(candidate) > len(best_set):
            best_set = candidate
            # Optional: try local search on best_set?
            # For now, just keep restarting.
            
    return sorted(best_set)
