import time
import sys
import random
import numpy as np

N = 9
TOTAL_VERTS = 1 << N  # 512

# Precompute bit_count for all 512-bit values (0 to 2^9-1)
# Actually we need bit_count for XOR results which are also 9-bit values
_BIT_COUNT = [i.bit_count() for i in range(TOTAL_VERTS)]

def construct():
    """
    Construct the longest induced path in Q_9 using DFS backtracking
    with Warnsdorff-style heuristic and time-limited search.
    """
    deadline = time.time() + 118  # 118s to be safe (120s limit)
    
    # Precompute neighbors for each vertex
    neighbors = [[] for _ in range(TOTAL_VERTS)]
    for v in range(TOTAL_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    def dfs_backtrack(start_vertex, max_depth=None):
        """DFS backtracking to find longest induced path starting from start_vertex."""
        best_path = [start_vertex]
        used = {start_vertex}
        
        # For each vertex, track which path vertices it's adjacent to (for quick validity check)
        # adj_count[v] = number of path vertices adjacent to v (excluding the last one which is expected)
        adj_count = [0] * TOTAL_VERTS
        for nb in neighbors[start_vertex]:
            adj_count[nb] += 1
        
        def extend(path, used, adj_count, depth):
            nonlocal best_path
            cur = path[-1]
            
            # Get candidate next vertices
            candidates = []
            for nb in neighbors[cur]:
                if nb in used:
                    continue
                # nb is adjacent to cur (by construction)
                # Check if nb is adjacent to any other path vertex
                # adj_count[nb] should be 0 (no other path vertices adjacent to nb)
                if adj_count[nb] > 0:
                    continue
                # Count options from nb for Warnsdorff heuristic
                opts = 0
                for nb2 in neighbors[nb]:
                    if nb2 not in used and adj_count[nb2] == 0:
                        opts += 1
                candidates.append((opts, nb))
            
            if not candidates:
                # Dead end - update best if longer
                if len(path) > len(best_path):
                    best_path = path.copy()
                return False
            
            # Sort by number of options (Warnsdorff: prefer vertices with fewer options)
            candidates.sort()
            
            found = False
            for opts, nb in candidates:
                # Extend
                path.append(nb)
                used.add(nb)
                
                # Update adj_count for neighbors of nb
                for nb2 in neighbors[nb]:
                    if nb2 != cur:  # cur is already in used, we don't care about its count
                        adj_count[nb2] += 1
                
                found = extend(path, used, adj_count, depth + 1)
                
                # Backtrack
                path.pop()
                used.remove(nb)
                for nb2 in neighbors[nb]:
                    if nb2 != cur:
                        adj_count[nb2] -= 1
                
                if found:
                    return True
            
            return False
        
        # Try to extend from start
        extend(best_path, used, adj_count, 0)
        return best_path
    
    # Try multiple starting vertices with different randomizations
    best_overall = [0]
    best_len = 1
    
    # First, try a few deterministic starts
    for start in [0, 1, 255, 256]:
        if time.time() > deadline - 5:
            break
        path = dfs_backtrack(start)
        if len(path) > best_len:
            best_overall = path
            best_len = len(path)
    
    # Now try random starts with limited depth to find good seeds
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline - 2:
            break
        start = rng.randint(0, TOTAL_VERTS - 1)
        
        # Limited DFS with early termination
        path = [start]
        used = {start}
        adj_count = [0] * TOTAL_VERTS
        for nb in neighbors[start]:
            adj_count[nb] += 1
        
        # Simple greedy extension with randomization
        while True:
            cur = path[-1]
            candidates = []
            for nb in neighbors[cur]:
                if nb not in used and adj_count[nb] == 0:
                    opts = 0
                    for nb2 in neighbors[nb]:
                        if nb2 not in used and adj_count[nb2] == 0:
                            opts += 1
                    candidates.append((opts, nb))
            
            if not candidates:
                break
            
            # Randomly pick among top candidates
            candidates.sort()
            # Pick from top 3 with some randomness
            top_k = min(3, len(candidates))
            _, nb = candidates[rng.randint(0, top_k - 1)]
            
            path.append(nb)
            used.add(nb)
            for nb2 in neighbors[nb]:
                if nb2 != cur:
                    adj_count[nb2] += 1
        
        if len(path) > best_len:
            best_overall = path
            best_len = len(path)
    
    # Local improvement: try to insert vertices into the best path
    def try_insert_vertex(path, v):
        """Try to insert v into path at any valid position."""
        L = len(path)
        for i in range(L - 1):
            a, b = path[i], path[i+1]
            if _BIT_COUNT[v ^ a] != 1 or _BIT_COUNT[v ^ b] != 1:
                continue
            # Check v against all other path vertices
            ok = True
            for j in range(L):
                if j == i or j == i+1:
                    continue
                if _BIT_COUNT[v ^ path[j]] == 1:
                    ok = False
                    break
            if ok:
                return path[:i+1] + [v] + path[i+1:]
        return None
    
    # Try to improve the best path by inserting vertices
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        path = best_overall
        used = set(path)
        
        # Find candidate vertices
        candidates = set()
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            diff_bit = (a ^ b).bit_length() - 1
            for l in range(N):
                if l == diff_bit:
                    continue
                v = b ^ (1 << l)
                if v not in used:
                    candidates.add(v)
        
        for v in list(candidates):
            if time.time() > deadline - 0.5:
                break
            new_path = try_insert_vertex(path, v)
            if new_path is not None:
                best_overall = new_path
                best_len = len(new_path)
                improved = True
                break
        
        # Also try to extend from ends
        if not improved:
            # Extend from end
            cur = path[-1]
            for nb in neighbors[cur]:
                if nb in used:
                    continue
                # Check validity
                ok = True
                for j in range(len(path) - 1):
                    if _BIT_COUNT[nb ^ path[j]] == 1:
                        ok = False
                        break
                if ok:
                    best_overall = path + [nb]
                    best_len += 1
                    improved = True
                    break
        
        # Extend from beginning
        if not improved:
            cur = path[0]
            for nb in neighbors[cur]:
                if nb in used:
                    continue
                ok = True
                for j in range(2, len(path)):
                    if _BIT_COUNT[nb ^ path[j]] == 1:
                        ok = False
                        break
                if ok:
                    best_overall = [nb] + path
                    best_len += 1
                    improved = True
                    break
    
    return best_overall
