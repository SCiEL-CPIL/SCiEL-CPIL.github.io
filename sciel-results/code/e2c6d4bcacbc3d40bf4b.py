"""
Snake-in-the-Box in Q9 using Warnsdorff's rule with backtracking and randomized restarts.

Structural change: replace MILP formulation with a depth-first search using
Warnsdorff's heuristic (prefer moves to vertices with fewest remaining options),
combined with randomized restarts and aggressive pruning. This exploits the
combinatorial structure of the hypercube more directly than a generic MILP solver.
"""
import time
import random
import sys
import numpy as np

N = 9
TOTAL_VERTS = 1 << N  # 512

def _precompute_neighbors():
    neighbors = [[] for _ in range(TOTAL_VERTS)]
    for v in range(TOTAL_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    return neighbors

_NEIGHBORS = _precompute_neighbors()

def _is_valid_extension(path, used_mask, candidate):
    """Check if candidate can be appended to path."""
    if used_mask[candidate]:
        return False
    # Check that candidate is not adjacent to any vertex in path except the last
    last = path[-1]
    for w in _NEIGHBORS[candidate]:
        if w != last and used_mask[w]:
            return False
    return True

def _count_options(v, used_mask, exclude_last=None):
    """Count how many valid extensions are available from vertex v."""
    count = 0
    for w in _NEIGHBORS[v]:
        if used_mask[w]:
            continue
        if w == exclude_last:
            continue
        # Check if w can be added (not adjacent to any used vertex except v)
        valid = True
        for x in _NEIGHBORS[w]:
            if x != v and used_mask[x]:
                valid = False
                break
        if valid:
            count += 1
    return count

def _warnsdorff_dfs(start, time_limit, rng):
    """
    Depth-first search using Warnsdorff's rule.
    Returns the longest path found within time_limit.
    """
    deadline = time.time() + time_limit
    best_path = []
    best_len = 0
    
    # State: path as list, used_mask as bytearray
    path = [start]
    used_mask = bytearray(TOTAL_VERTS)
    used_mask[start] = 1
    
    # Stack of (path_copy, used_mask_copy, next_bit_to_try)
    # We use iterative DFS to avoid recursion depth issues
    
    # For Warnsdorff's rule, at each step we sort candidates by number of options
    # and try them in that order
    
    def _dfs_recursive(path, used_mask, depth):
        nonlocal best_path, best_len, deadline
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        if time.time() > deadline:
            return
        
        cur = path[-1]
        
        # Get all valid candidates
        candidates = []
        for w in _NEIGHBORS[cur]:
            if not used_mask[w]:
                # Check validity
                valid = True
                for x in _NEIGHBORS[w]:
                    if x != cur and used_mask[x]:
                        valid = False
                        break
                if valid:
                    # Count options for Warnsdorff's rule
                    opts = 0
                    for y in _NEIGHBORS[w]:
                        if not used_mask[y] and y != cur:
                            # Check if y is valid (not adjacent to any used except w)
                            y_valid = True
                            for z in _NEIGHBORS[y]:
                                if z != w and used_mask[z]:
                                    y_valid = False
                                    break
                            if y_valid:
                                opts += 1
                    candidates.append((opts, w))
        
        if not candidates:
            return
        
        # Sort by Warnsdorff's rule (fewest options first)
        candidates.sort(key=lambda x: (x[0], rng.random()))
        
        for _, w in candidates:
            if time.time() > deadline:
                return
            
            used_mask[w] = 1
            path.append(w)
            
            _dfs_recursive(path, used_mask, depth + 1)
            
            path.pop()
            used_mask[w] = 0
            
            if time.time() > deadline:
                return
    
    _dfs_recursive(path, used_mask, 0)
    
    return best_path

def _warnsdorff_dfs_iterative(start, time_limit, rng):
    """
    Iterative depth-first search using Warnsdorff's rule.
    Uses explicit stack to avoid recursion limits.
    """
    deadline = time.time() + time_limit
    best_path = []
    best_len = 0
    
    # Stack entries: (path, used_mask, candidates, candidate_idx)
    # We'll use a simpler approach: maintain path and used_mask, 
    # and at each level compute candidates sorted by Warnsdorff
    
    path = [start]
    used_mask = bytearray(TOTAL_VERTS)
    used_mask[start] = 1
    
    # For iterative DFS with backtracking, we need to save state at each level
    # Let's use a stack of (candidate_list, current_index)
    
    def _compute_candidates(cur):
        """Compute sorted candidates for current vertex."""
        candidates = []
        for w in _NEIGHBORS[cur]:
            if not used_mask[w]:
                valid = True
                for x in _NEIGHBORS[w]:
                    if x != cur and used_mask[x]:
                        valid = False
                        break
                if valid:
                    opts = 0
                    for y in _NEIGHBORS[w]:
                        if not used_mask[y] and y != cur:
                            y_valid = True
                            for z in _NEIGHBORS[y]:
                                if z != w and used_mask[z]:
                                    y_valid = False
                                    break
                            if y_valid:
                                opts += 1
                    candidates.append((opts, w))
        candidates.sort(key=lambda x: (x[0], rng.random()))
        return [w for _, w in candidates]
    
    # Stack: list of (candidates, index) for each level
    stack = []
    
    while True:
        if time.time() > deadline:
            break
        
        cur = path[-1]
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Get candidates for current vertex
        candidates = _compute_candidates(cur)
        
        if not candidates:
            # Dead end, backtrack
            if not stack:
                break
            # Undo last move
            path.pop()
            used_mask[path[-1] if path else 0]  # just clear
            # Actually, we need to properly backtrack
            # Let me restructure this
            
            # Pop from stack
            if stack:
                prev_candidates, prev_idx = stack.pop()
                if path:
                    path.pop()
                    used_mask[path[-1]] = 0
                # Continue with next candidate from prev level
                # This is getting complex, let's use a cleaner approach
            break
        else:
            # Try first candidate
            stack.append((candidates, 0))
            w = candidates[0]
            used_mask[w] = 1
            path.append(w)
    
    return best_path

def _warnsdorff_search(start, time_budget, rng):
    """
    Perform Warnsdorff-guided DFS from start vertex with time_budget seconds.
    Returns the best path found.
    """
    deadline = time.time() + time_budget
    best_path = []
    best_len = 0
    
    path = [start]
    used_mask = bytearray(TOTAL_VERTS)
    used_mask[start] = 1
    
    # Iterative DFS with explicit state management
    # Stack stores: (list_of_candidates, next_index_to_try)
    stack = []
    
    while time.time() < deadline:
        cur = path[-1]
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Compute candidates for current vertex
        candidates = []
        for w in _NEIGHBORS[cur]:
            if not used_mask[w]:
                # Check if w is a valid extension
                valid = True
                for x in _NEIGHBORS[w]:
                    if x != cur and used_mask[x]:
                        valid = False
                        break
                if valid:
                    # Count options from w for Warnsdorff's rule
                    opts = 0
                    for y in _NEIGHBORS[w]:
                        if not used_mask[y] and y != cur:
                            # Check if y would be valid
                            y_valid = True
                            for z in _NEIGHBORS[y]:
                                if z != w and used_mask[z]:
                                    y_valid = False
                                    break
                            if y_valid:
                                opts += 1
                    candidates.append((opts, w))
        
        if not candidates:
            # Dead end - backtrack
            if not stack:
                break
            # Pop last state
            prev_cands, prev_idx = stack.pop()
            # Remove last vertex from path
            last_v = path.pop()
            used_mask[last_v] = 0
            # Try next candidate from previous level
            if prev_idx + 1 < len(prev_cands):
                stack.append((prev_cands, prev_idx + 1))
                w = prev_cands[prev_idx + 1]
                used_mask[w] = 1
                path.append(w)
            # else: continue backtracking
        else:
            # Sort by Warnsdorff's rule
            candidates.sort(key=lambda x: (x[0], rng.random()))
            # Try first candidate
            stack.append((candidates, 0))
            w = candidates[0]
            used_mask[w] = 1
            path.append(w)
    
    return best_path

def _greedy_extend(path, used_mask=None):
    """Extend a path greedily from both ends."""
    if used_mask is None:
        used_mask = bytearray(TOTAL_VERTS)
        for v in path:
            used_mask[v] = 1
    
    # Try extending from the end
    while True:
        cur = path[-1]
        found = False
        for b in range(N):
            v = cur ^ (1 << b)
            if not used_mask[v]:
                valid = True
                for w in _NEIGHBORS[v]:
                    if w != cur and used_mask[w]:
                        valid = False
                        break
                if valid:
                    path.append(v)
                    used_mask[v] = 1
                    found = True
                    break
        if not found:
            break
    
    # Try extending from the beginning
    while True:
        cur = path[0]
        found = False
        for b in range(N):
            v = cur ^ (1 << b)
            if not used_mask[v]:
                valid = True
                for w in _NEIGHBORS[v]:
                    if w != cur and used_mask[w]:
                        valid = False
                        break
                if valid:
                    path.insert(0, v)
                    used_mask[v] = 1
                    found = True
                    break
        if not found:
            break
    
    return path

def _try_insert(path, vertex):
    """Try to insert a vertex into the path."""
    used_mask = bytearray(TOTAL_VERTS)
    for v in path:
        used_mask[v] = 1
    
    if used_mask[vertex]:
        return None
    
    # Try inserting between consecutive vertices
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        if (vertex ^ u).bit_count() == 1 and (vertex ^ v).bit_count() == 1:
            valid = True
            for w in _NEIGHBORS[vertex]:
                if w != u and w != v and used_mask[w]:
                    valid = False
                    break
            if valid:
                new_path = path[:i+1] + [vertex] + path[i+1:]
                return new_path
    
    # Try prepending
    if (vertex ^ path[0]).bit_count() == 1:
        valid = True
        for w in _NEIGHBORS[vertex]:
            if w != path[0] and used_mask[w]:
                valid = False
                break
        if valid:
            return [vertex] + path
    
    # Try appending
    if (vertex ^ path[-1]).bit_count() == 1:
        valid = True
        for w in _NEIGHBORS[vertex]:
            if w != path[-1] and used_mask[w]:
                valid = False
                break
        if valid:
            return path + [vertex]
    
    return None

def construct():
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Multiple randomized restarts with Warnsdorff's rule
    # Each restart gets a budget of time
    num_restarts = 20
    restart_budget = 6.0  # seconds per restart
    
    for restart in range(num_restarts):
        remaining_time = deadline - time.time()
        if remaining_time < 2.0:
            break
        
        # Choose a random start vertex
        start = rng.randint(0, TOTAL_VERTS - 1)
        
        # Allocate time for this restart
        alloc_time = min(restart_budget, remaining_time - 1.0)
        if alloc_time < 0.5:
            break
        
        path = _warnsdorff_search(start, alloc_time, rng)
        
        if path and len(path) > best_len:
            # Try to extend greedily
            path = _greedy_extend(path)
            best_path = path
            best_len = len(path)
        
        # Try local search: insert random vertices
        if time.time() < deadline - 2.0 and best_path:
            used_mask = bytearray(TOTAL_VERTS)
            for v in best_path:
                used_mask[v] = 1
            
            insert_attempts = 500
            for _ in range(insert_attempts):
                if time.time() > deadline - 2.0:
                    break
                v = rng.randint(0, TOTAL_VERTS - 1)
                if used_mask[v]:
                    continue
                new_path = _try_insert(best_path, v)
                if new_path:
                    best_path = new_path
                    best_len = len(new_path)
                    # Update used_mask
                    used_mask = bytearray(TOTAL_VERTS)
                    for x in best_path:
                        used_mask[x] = 1
    
    # Final greedy extension
    if best_path:
        best_path = _greedy_extend(best_path)
        best_len = len(best_path)
    
    # If we haven't found a good path, fall back to simple greedy
    if best_len < 50:
        greedy_path = [0]
        greedy_mask = bytearray(TOTAL_VERTS)
        greedy_mask[0] = 1
        while True:
            cur = greedy_path[-1]
            found = False
            for b in range(N):
                v = cur ^ (1 << b)
                if not greedy_mask[v]:
                    valid = True
                    for w in _NEIGHBORS[v]:
                        if w != cur and greedy_mask[w]:
                            valid = False
                            break
                    if valid:
                        greedy_path.append(v)
                        greedy_mask[v] = 1
                        found = True
                        break
            if not found:
                break
        if len(greedy_path) > best_len:
            best_path = greedy_path
            best_len = best_len
    
    return best_path

# EVOLVE-BLOCK-START
# The construct function is the entry point.
# EVOLVE-BLOCK-END
