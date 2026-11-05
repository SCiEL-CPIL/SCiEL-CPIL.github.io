import sys
import time
import numpy as np
from itertools import permutations

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def _standard_construct(n):
    """Classic recursive construction: 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _overlap_len(a, b):
    """Longest suffix of a that is a prefix of b."""
    max_ov = min(len(a), len(b))
    for ov in range(max_ov, 0, -1):
        if a[-ov:] == b[:ov]:
            return ov
    return 0

def _construct_from_order(order, n):
    """Given an ordered list of (n-1)-permutations, build the superpermutation."""
    if not order:
        return list(range(1, n + 1))
    
    # First block: order[0] + [n] + order[0]
    word = list(order[0]) + [n] + list(order[0])
    
    for i in range(1, len(order)):
        prev = order[i - 1]
        curr = order[i]
        block = list(curr) + [n] + list(curr)
        # Overlap: longest suffix of prev that matches prefix of curr
        ov = 0
        max_ov = min(len(prev), len(curr))
        for o in range(max_ov, 0, -1):
            if prev[-o:] == curr[:o]:
                ov = o
                break
        word += block[ov:]
    
    return word

def _verify_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive windows."""
    if len(word) < n:
        return False
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        found.add(window)
    return len(found) == _factorial(n)

def _compute_overlap_matrix(perms):
    """Compute overlap matrix for a list of permutations."""
    n_perms = len(perms)
    # Convert to numpy arrays for faster comparison
    perm_arrays = np.array(perms, dtype=np.uint8)
    
    overlap_matrix = np.zeros((n_perms, n_perms), dtype=np.int8)
    
    for i in range(n_perms):
        for j in range(n_perms):
            if i != j:
                # Compute longest suffix of perms[i] matching prefix of perms[j]
                a = perm_arrays[i]
                b = perm_arrays[j]
                max_ov = min(len(a), len(b))
                ov = 0
                for o in range(max_ov, 0, -1):
                    if np.array_equal(a[-o:], b[:o]):
                        ov = o
                        break
                overlap_matrix[i][j] = ov
    
    return overlap_matrix

def _path_score(path, overlap_matrix):
    """Compute total overlap for a path."""
    total = 0
    for i in range(len(path) - 1):
        total += overlap_matrix[path[i]][path[i+1]]
    return total

def _greedy_tsp(overlap_matrix, start_idx=0):
    """Greedy TSP: always pick unvisited neighbor with max overlap."""
    n_perms = overlap_matrix.shape[0]
    visited = np.zeros(n_perms, dtype=bool)
    path = [start_idx]
    visited[start_idx] = True
    current = start_idx
    
    for _ in range(n_perms - 1):
        best_next = -1
        best_ov = -1
        # Find unvisited node with max overlap from current
        unvisited = np.where(~visited)[0]
        if len(unvisited) == 0:
            break
        overlaps = overlap_matrix[current][unvisited]
        best_local_idx = np.argmax(overlaps)
        best_next = unvisited[best_local_idx]
        best_ov = overlaps[best_local_idx]
        
        visited[best_next] = True
        path.append(best_next)
        current = best_next
    
    return path

def _two_opt_improve(path, overlap_matrix, max_iterations=50, time_limit=None, start_time=None):
    """2-opt local search to maximize total overlap."""
    n_perms = len(path)
    path = path[:]
    best_score = _path_score(path, overlap_matrix)
    
    for iteration in range(max_iterations):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        for i in range(n_perms - 1):
            for j in range(i + 1, n_perms):
                # Try reversing segment [i+1..j]
                new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                new_score = _path_score(new_path, overlap_matrix)
                if new_score > best_score:
                    best_score = new_score
                    path = new_path
                    improved = True
                    if time_limit is not None and start_time is not None:
                        if time.time() - start_time > time_limit:
                            return path
        if not improved:
            break
    
    return path

def _or_opt_improve(path, overlap_matrix, max_iterations=30, time_limit=None, start_time=None):
    """OR-opt: move a segment of 1-3 nodes to a different position."""
    n_perms = len(path)
    path = path[:]
    best_score = _path_score(path, overlap_matrix)
    
    for iteration in range(max_iterations):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        # Try moving segments of length 1, 2, 3
        for seg_len in [1, 2, 3]:
            for i in range(n_perms - seg_len + 1):
                segment = path[i:i+seg_len]
                # Try inserting at each position
                for j in range(n_perms - seg_len + 1):
                    if j == i:
                        continue
                    if j > i:
                        j -= seg_len
                    if j == i:
                        continue
                    
                    # Build new path
                    remaining = path[:i] + path[i+seg_len:]
                    new_path = remaining[:j] + segment + remaining[j:]
                    
                    new_score = _path_score(new_path, overlap_matrix)
                    if new_score > best_score:
                        best_score = new_score
                        path = new_path
                        improved = True
                        if time_limit is not None and start_time is not None:
                            if time.time() - start_time > time_limit:
                                return path
        if not improved:
            break
    
    return path

def _construct_n7_optimized():
    """Optimized construction for n=7 using TSP on permutation ordering."""
    start_time = time.time()
    time_limit = 120.0
    
    # Permutations of 1..6
    perms_6 = list(permutations(range(1, 7)))
    n_perms = len(perms_6)
    
    # Precompute overlap matrix
    overlap_matrix = _compute_overlap_matrix(perms_6)
    
    # Start with the order from the standard construction
    word_6 = _standard_construct(6)
    order_6 = []
    seen = set()
    for i in range(len(word_6) - 6 + 1):
        window = tuple(word_6[i:i+6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            order_6.append(window)
    
    # Convert to indices
    perm_to_idx = {p: i for i, p in enumerate(perms_6)}
    initial_path = [perm_to_idx[p] for p in order_6]
    
    # Run greedy TSP as starting point
    greedy_path = _greedy_tsp(overlap_matrix)
    
    # Compare scores
    initial_score = _path_score(initial_path, overlap_matrix)
    greedy_score = _path_score(greedy_path, overlap_matrix)
    
    # Use the better of initial and greedy as starting point
    if greedy_score > initial_score:
        best_path = greedy_path
    else:
        best_path = initial_path
    
    # Improve with 2-opt
    best_path = _two_opt_improve(best_path, overlap_matrix, max_iterations=100, 
                                  time_limit=time_limit - (time.time() - start_time),
                                  start_time=start_time)
    
    # Further improve with OR-opt if time remains
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        best_path = _or_opt_improve(best_path, overlap_matrix, max_iterations=30,
                                     time_limit=remaining_time - 5,
                                     start_time=start_time)
    
    # Construct the superpermutation
    word = _construct_from_order([perms_6[i] for i in best_path], 7)
    
    return word

def construct(n):
    if n <= 0:
        return ""
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1312131"
    if n == 4:
        return "1413121413124131241"
    if n == 5:
        return "".join(map(str, _standard_construct(5)))
    if n == 6:
        return "".join(map(str, _standard_construct(6)))
    
    if n == 7:
        # Try optimized construction, fall back to standard if time runs out
        start_time = time.time()
        deadline = start_time + 140.0  # safety margin
        
        try:
            word = _construct_n7_optimized()
            # Verify it's valid
            if _verify_superpermutation(word, 7):
                return "".join(map(str, word))
        except Exception:
            pass
        
        # Fallback to standard construction
        if time.time() < deadline:
            return "".join(map(str, _standard_construct(7)))
        else:
            return "".join(map(str, _standard_construct(7)))
    
    # For n > 7, use standard construction
    return "".join(map(str, _standard_construct(n)))
