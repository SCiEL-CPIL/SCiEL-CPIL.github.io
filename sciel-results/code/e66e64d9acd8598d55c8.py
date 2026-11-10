import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin
    
    # Try to find a solution with small count (4, 5, 6) using DFS
    # This is faster than full DP if a small solution exists
    # If not found, fallback to DP
    
    def dfs(start, remaining_sum, count, current_set):
        # remaining_sum is fraction 1/a + ... = 1
        # We want to find 'count' more terms
        # Actually, let's fix the total count K and search
        pass

    # Since implementing a robust DFS for Egyptian fractions is complex and error-prone,
    # and the DP is already working, I will rely on the DP.
    # However, I can add a specific check for known small solutions if I had them.
    # Since I don't, I will return the DP result.
    
    # To ensure we don't timeout, the DP is fast.
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
        if time.time() > deadline:
             break
             
    if target in reached:
        return reached[target]
    raise RuntimeError("no baseline representation")
