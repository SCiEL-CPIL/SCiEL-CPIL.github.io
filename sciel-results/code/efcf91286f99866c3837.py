"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def _count_pairs(sets):
    """Count unordered pairs where one is a strict subset of the other."""
    m = len(sets)
    fs = [frozenset(s) for s in sets]
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            if fs[i] < fs[j] or fs[j] < fs[i]:
                count += 1
    return count

def _delta_count(i, new_set, sets, fs):
    """Compute the change in pair count if we replace sets[i] with new_set."""
    m = len(sets)
    new_fs = frozenset(new_set)
    delta = 0
    for j in range(m):
        if j == i:
            continue
        old = 1 if (fs[i] < fs[j] or fs[j] < fs[i]) else 0
        new = 1 if (new_fs < fs[j] or fs[j] < new_fs) else 0
        delta += new - old
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    n = 12
    m = 40
    
    # Precompute all subsets as frozensets, indexed by mask
    all_subsets = []
    for mask in range(1 << n):
        s = frozenset(b for b in range(n) if (mask >> b) & 1)
        all_subsets.append(s)
    
    # Greedy chain-based initialization:
    # Start with a long chain, then greedily add sets that maximize comparability
    # Chain: {}, {0}, {0,1}, {0,1,2}, ..., {0,...,11} (13 sets)
    # Then greedily add remaining sets
    
    current = []
    current_fs = []
    used = set()
    
    # Add the chain
    for k in range(n + 1):
        s = frozenset(range(k))
        current.append(list(s))
        current_fs.append(s)
        used.add(s)
    
    # Greedily add sets that maximize total comparability
    while len(current) < m:
        if time.time() > deadline:
            break
        best_delta = -1
        best_mask = -1
        for mask in range(1 << n):
            if mask in used:
                continue
            s = all_subsets[mask]
            # Compute how many existing sets are comparable with s
            delta = 0
            for j in range(len(current)):
                if s < current_fs[j] or current_fs[j] < s:
                    delta += 1
            if delta > best_delta:
                best_delta = delta
                best_mask = mask
        if best_mask == -1:
            break
        s = all_subsets[best_mask]
        current.append(list(s))
        current_fs.append(s)
        used.add(s)
    
    if len(current) < m:
        # Fill with any remaining subsets
        for mask in range(1 << n):
            if len(current) >= m:
                break
            if mask not in used:
                s = all_subsets[mask]
                current.append(list(s))
                current_fs.append(s)
                used.add(s)
    
    # Local search: try swapping each element with any other subset
    best_score = _count_pairs(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(m):
            if time.time() > deadline:
                break
            best_delta = 0
            best_mask = -1
            for mask in range(1 << n):
                if mask in used and mask != current_fs[i]:
                    continue
                s = all_subsets[mask]
                if s == current_fs[i]:
                    continue
                delta = _delta_count(i, s, current, current_fs)
                if delta > best_delta:
                    best_delta = delta
                    best_mask = mask
            if best_mask >= 0 and best_delta > 0:
                old_mask = None
                for mm in range(1 << n):
                    if all_subsets[mm] == current_fs[i]:
                        old_mask = mm
                        break
                used.discard(old_mask)
                current[i] = list(all_subsets[best_mask])
                current_fs[i] = all_subsets[best_mask]
                used.add(best_mask)
                best_score += best_delta
                improved = True
    
    # Second pass: try 2-swap moves (swap two elements)
    if time.time() < deadline:
        for _ in range(5):  # limited iterations
            if time.time() > deadline:
                break
            found = False
            for i in range(m):
                if time.time() > deadline:
                    break
                for j in range(i + 1, m):
                    if time.time() > deadline:
                        break
                    # Try swapping i and j with other sets
                    best_delta = 0
                    best_mi = -1
                    best_mj = -1
                    # This is expensive, so only do it if we have time
                    if time.time() > deadline:
                        break
                    # Skip 2-swap for now as it's too expensive
                    break
                break
            if found:
                continue
            break
    
    # Ensure types are correct
    result = [list(s) for s in current]
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = int(result[i][j])
    
    return result

# EVOLVE-BLOCK-END
