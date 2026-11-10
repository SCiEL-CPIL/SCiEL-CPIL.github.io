import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute modular inverses
    inv = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # Precompute all subsets and their residues, grouped by size
    # subsets_by_size[s] = list of (residue, subset)
    subsets_by_size = [list() for _ in range(11)]
    
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            res = sum(inv[x] for x in combo) % 101
            subsets_by_size[s].append((res, list(combo)))
    
    # For each size, build a set of achievable residues and one example per residue
    residues_by_size = [set() for _ in range(11)]
    example_by_size = [dict() for _ in range(11)]
    for s in range(1, 11):
        for res, subset in subsets_by_size[s]:
            residues_by_size[s].add(res)
            if res not in example_by_size[s]:
                example_by_size[s][res] = subset
    
    # Find minimum k such that union of residues for sizes 1..k covers all 101
    min_k = None
    for k in range(1, 11):
        covered = set()
        for s in range(1, k + 1):
            covered.update(residues_by_size[s])
        if len(covered) == 101:
            min_k = k
            break
    
    if min_k is None:
        min_k = 10
    
    # Now we need to find an assignment that minimizes the maximum size used
    # We want to assign each residue r a subset of size <= min_k
    # To minimize the max size, we try to use smaller sizes preferentially
    
    # Strategy: For each residue, find the smallest size s such that r is achievable
    # But this greedy approach might not be optimal - we need to ensure all residues get assigned
    
    # Better approach: Try to find the minimum possible max size
    # For a given max size M, check if all 101 residues can be assigned subsets of size <= M
    
    # Since min_k is the smallest k where coverage is complete, we know M >= min_k
    # Let's try to find the actual minimum max size by checking if we can do better
    
    # Actually, the problem is: we want to minimize the maximum row length.
    # The current approach just picks the smallest size for each residue independently.
    # But this might not be optimal - we need to consider conflicts.
    
    # However, since denominators can recur across rows, there are no conflicts!
    # Each row is independent - we just need to find, for each residue, a subset of size <= k
    # that achieves that residue.
    
    # So the optimal strategy is: for each residue, use the smallest possible subset size.
    # This automatically minimizes the maximum row length.
    
    result = []
    max_len_used = 0
    for r in range(101):
        found = None
        for s in range(1, min_k + 1):
            if r in example_by_size[s]:
                found = example_by_size[s][r]
                break
        if found is None:
            # Should not happen if min_k is correct
            found = [1]
        result.append(found)
        max_len_used = max(max_len_used, len(found))
    
    return result
