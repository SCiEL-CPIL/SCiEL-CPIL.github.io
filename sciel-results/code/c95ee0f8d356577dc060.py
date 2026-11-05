import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Generate Hex(3)
    R = 3
    points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if max(abs(a), abs(b), abs(a + b)) <= R:
                points.append((a, b))
    
    if len(points) != 37:
        return [[a, b] for a in range(16) for b in range(2)]

    # Precompute adjacency or just check distance on the fly
    # Distance squared = da^2 + da*db + db^2
    # Unit distance if dist_sq == 1

    def count_edges(subset):
        count = 0
        n = len(subset)
        for i in range(n):
            a1, b1 = subset[i]
            for j in range(i + 1, n):
                a2, b2 = subset[j]
                da = a1 - a2
                db = b1 - b2
                if da*da + da*db + db*db == 1:
                    count += 1
        return count

    best_edges = -1
    best_subset = None
    
    # Iterate over all combinations of 5 points to remove
    # There are 37 choose 5 = 435,897 combinations
    for remove_indices in itertools.combinations(range(37), 5):
        if time.time() > deadline:
            break
        
        remove_set = set(remove_indices)
        subset = []
        for i in range(37):
            if i not in remove_set:
                subset.append(points[i])
        
        edges = count_edges(subset)
        if edges > best_edges:
            best_edges = edges
            best_subset = [list(p) for p in subset]
            
            # If we found 76, we can stop early as it's the theoretical max
            if best_edges == 76:
                break

    if best_subset is None:
        # Fallback
        return [[a, b] for a in range(16) for b in range(2)]
        
    return best_subset
