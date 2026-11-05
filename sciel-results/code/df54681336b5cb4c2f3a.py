"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys
from collections import deque

_N = 24

# Precompute the edge order from the given data
_EDGE_DATA = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build arrival index: edge_index[(u,v)] = 1-based position in the shuffled list
_EDGE_INDEX = {}
for i, (u, v) in enumerate(_EDGE_DATA):
    if u > v:
        u, v = v, u
    _EDGE_INDEX[(u, v)] = i + 1

def _edge_score(u, v):
    """Get the arrival index (1-based) for edge (u,v)."""
    if u > v:
        u, v = v, u
    return _EDGE_INDEX.get((u, v), float('inf'))

def _cycle_score(cycle):
    """Compute the score of a cycle: max arrival index of its edges."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        s = _edge_score(u, v)
        if s > max_idx:
            max_idx = s
    return max_idx

def _find_hamiltonian_cycle_threshold(threshold, deadline=None):
    """
    Find a Hamilton cycle where all edges have arrival index <= threshold.
    Uses DFS with pruning: degree >= 2, connectivity check.
    Returns the cycle as a list of vertices, or None if not found.
    """
    # Build adjacency list with only edges <= threshold
    adj = [[] for _ in range(_N)]
    for i in range(_N):
        for j in range(i + 1, _N):
            s = _edge_score(i, j)
            if s <= threshold:
                adj[i].append(j)
                adj[j].append(i)
    
    # Check if each vertex has degree >= 2
    for i in range(_N):
        if len(adj[i]) < 2:
            return None
    
    # Check connectivity
    visited = [False] * _N
    queue = deque([0])
    visited[0] = True
    count = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    if count < _N:
        return None
    
    # DFS to find Hamilton cycle
    # Fix vertex 0 as start, and fix the first neighbor to reduce symmetry
    start = 0
    path = [start]
    used = [False] * _N
    used[start] = True
    
    def dfs():
        if len(path) == _N:
            # Check if we can close the cycle
            last = path[-1]
            if _edge_score(last, start) <= threshold:
                return True
            return False
        
        current = path[-1]
        # Try neighbors in order of increasing edge score
        neighbors = sorted(adj[current], key=lambda x: _edge_score(current, x))
        
        for nxt in neighbors:
            if used[nxt]:
                continue
            
            # Pruning: if nxt is the last vertex (path length N-1), check closing edge
            if len(path) == _N - 1:
                if _edge_score(nxt, start) > threshold:
                    continue
            
            # Pruning: don't allow visiting start until the end
            if nxt == start and len(path) < _N:
                continue
            
            # Connectivity pruning: after adding nxt, check that remaining 
            # unvisited vertices can still form a path
            if len(path) < _N - 1:
                # Check that the unvisited vertices (excluding start) are 
                # connected to the current path in the subgraph
                unvisited = [v for v in range(_N) if not used[v] and v != start]
                if unvisited:
                    # Check if any unvisited vertex has no edge to current path
                    # or to other unvisited vertices
                    path_set = set(path)
                    for u in unvisited:
                        has_edge_to_path = False
                        for v in path_set:
                            if _edge_score(u, v) <= threshold:
                                has_edge_to_path = True
                                break
                        if not has_edge_to_path:
                            # This vertex would be isolated from the path
                            # unless it connects to other unvisited vertices
                            # which also need to connect back
                            # Simple check: does u have any edge to unvisited or path?
                            has_any = False
                            for v in unvisited:
                                if v != u and _edge_score(u, v) <= threshold:
                                    has_any = True
                                    break
                            if not has_any and not has_edge_to_path:
                                return False
                            break
            
            path.append(nxt)
            used[nxt] = True
            if dfs():
                return True
            path.pop()
            used[nxt] = False
        
        return False
    
    if dfs():
        return path
    return None

def _binary_search_threshold(deadline):
    """
    Binary search for the minimum threshold K such that a Hamilton cycle exists
    with all edges having arrival index <= K.
    """
    # The maximum possible edge index is len(_EDGE_DATA)
    max_idx = len(_EDGE_DATA)
    
    lo = _N  # Minimum possible is _N (need at least N edges)
    hi = max_idx
    
    best_cycle = None
    best_threshold = max_idx + 1
    
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        
        cycle = _find_hamiltonian_cycle_threshold(mid, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            best_threshold = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_cycle, best_threshold

def _local_search(cycle, rng, num_iterations, deadline=None):
    """Improve a cycle using 2-opt and 1-swap moves."""
    n = len(cycle)
    best_score = _cycle_score(cycle)
    best_cycle = list(cycle)
    
    for it in range(num_iterations):
        if deadline is not None and time.time() > deadline:
            break
            
        move_type = it % 2
        
        if move_type == 0:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i > j:
                i, j = j, i
            if i == j:
                continue
            if i == 0 and j == n - 1:
                continue
            
            new_cycle = best_cycle[:i] + list(reversed(best_cycle[i:j+1])) + best_cycle[j+1:]
        else:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            new_cycle = list(best_cycle)
            new_cycle[i], new_cycle[j] = best_cycle[j], best_cycle[i]
        
        new_score = _cycle_score(new_cycle)
        if new_score < best_score:
            best_score = new_score
            best_cycle = new_cycle
    
    return best_cycle, best_score

def _simulated_annealing(cycle, rng, iterations, start_temp, end_temp, deadline=None):
    """Improve a cycle using simulated annealing with 2-opt and 1-swap moves."""
    n = len(cycle)
    current_cycle = list(cycle)
    current_score = _cycle_score(current_cycle)
    best_cycle = list(cycle)
    best_score = current_score
    
    for it in range(iterations):
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 3:
                break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max(1, iterations - 1))
        
        move_type = it % 2
        
        if move_type == 0:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i > j:
                i, j = j, i
            if i == j:
                continue
            if i == 0 and j == n - 1:
                continue
            
            new_cycle = current_cycle[:i] + list(reversed(current_cycle[i:j+1])) + current_cycle[j+1:]
        else:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            new_cycle = list(current_cycle)
            new_cycle[i], new_cycle[j] = current_cycle[j], current_cycle[i]
        
        new_score = _cycle_score(new_cycle)
        delta = new_score - current_score
        
        if delta < 0:
            current_cycle = new_cycle
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_cycle = list(new_cycle)
        else:
            if temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10)):
                current_cycle = new_cycle
                current_score = new_score
    
    return best_cycle, best_score

def _multi_start_local_search(rng, num_starts, iters_per_start, deadline):
    """Try multiple random starts with local search."""
    best_cycle = None
    best_score = float('inf')
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        perm = list(range(_N))
        rng.shuffle(perm)
        improved_cycle, improved_score = _local_search(perm, rng, iters_per_start, deadline)
        if improved_score < best_score:
            best_score = improved_score
            best_cycle = list(improved_cycle)
    
    return best_cycle, best_score

def _greedy_cycle(rng, start_vertex=None):
    """Build a cycle greedily by choosing the next vertex with the earliest edge."""
    if start_vertex is None:
        start_vertex = rng.randint(0, _N - 1)
    
    visited = [False] * _N
    visited[start_vertex] = True
    cycle = [start_vertex]
    current = start_vertex
    
    for _ in range(_N - 1):
        best_next = -1
        best_score = float('inf')
        for v in range(_N):
            if not visited[v]:
                s = _edge_score(current, v)
                if s < best_score:
                    best_score = s
                    best_next = v
                elif s == best_score and rng.random() < 0.5:
                    best_next = v
        
        if best_next == -1:
            return None
        cycle.append(best_next)
        visited[best_next] = True
        current = best_next
    
    return cycle

def _greedy_multi_start(rng, num_starts, deadline):
    """Try greedy construction from multiple starts, then local search."""
    best_cycle = None
    best_score = float('inf')
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        start_v = rng.randint(0, _N - 1)
        g_cycle = _greedy_cycle(rng, start_v)
        if g_cycle is None:
            continue
        g_score = _cycle_score(g_cycle)
        improved_cycle, improved_score = _local_search(g_cycle, rng, 500, deadline)
        if improved_score < best_score:
            best_score = improved_score
            best_cycle = list(improved_cycle)
    
    return best_cycle, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    rng = random.Random(12345)
    
    best_cycle = None
    best_score = float('inf')
    
    # Phase 1: Exact DFS with binary search on threshold
    t = time.time()
    dfs_cycle, dfs_threshold = _binary_search_threshold(deadline)
    if dfs_cycle is not None:
        best_score = dfs_threshold
        best_cycle = list(dfs_cycle)
        print(f"Phase 1 (DFS+binary): {time.time()-t:.2f}s, threshold={dfs_threshold}")
    
    # Phase 2: Quick random starts with local search
    t = time.time()
    num_quick = 30
    iters_quick = 500
    cycle, score = _multi_start_local_search(rng, num_quick, iters_quick, deadline)
    if cycle is not None and score < best_score:
        best_score = score
        best_cycle = cycle
    print(f"Phase 2 (quick LS): {time.time()-t:.2f}s, best={best_score}")
    
    # Phase 3: Greedy multi-start
    t = time.time()
    cycle, score = _greedy_multi_start(rng, 50, deadline)
    if cycle is not None and score < best_score:
        best_score = score
        best_cycle = cycle
    print(f"Phase 3 (greedy): {time.time()-t:.2f}s, best={best_score}")
    
    # Phase 4: Simulated annealing on best cycle
    if best_cycle is not None:
        t = time.time()
        remaining_budget = deadline - time.time()
        if remaining_budget > 10:
            sa_iters = int(remaining_budget * 2000)
            sa_iters = max(sa_iters, 1000)
            improved_cycle, improved_score = _simulated_annealing(
                best_cycle, rng, sa_iters, 5.0, 0.5, deadline
            )
            if improved_score < best_score:
                best_score = improved_score
                best_cycle = improved_cycle
        print(f"Phase 4 (SA): {time.time()-t:.2f}s, best={best_score}")
    
    # Phase 5: More local search passes if time remains
    if best_cycle is not None:
        while time.time() < deadline - 15:
            t = time.time()
            improved_cycle, improved_score = _local_search(best_cycle, rng, 3000, deadline)
            if improved_score < best_score:
                best_score = improved_score
                best_cycle = improved_cycle
            else:
                break
            print(f"Phase 5 iter: best={best_score}, time={time.time()-start_time:.1f}s")
    
    # Phase 6: Final SA polish
    if best_cycle is not None and time.time() < deadline - 5:
        t = time.time()
        remaining = deadline - time.time()
        if remaining > 5:
            sa_iters = int(remaining * 3000)
            improved_cycle, improved_score = _simulated_annealing(
                best_cycle, rng, sa_iters, 3.0, 0.3, deadline
            )
            if improved_score < best_score:
                best_score = improved_score
                best_cycle = improved_cycle
        print(f"Phase 6 (SA polish): time={time.time()-start_time:.1f}s, best={best_score}")
    
    # Final local search pass
    if best_cycle is not None and time.time() < deadline - 3:
        improved_cycle, improved_score = _local_search(best_cycle, rng, 3000, deadline)
        if improved_score < best_score:
            best_score = improved_score
            best_cycle = improved_cycle
        print(f"Final LS: best={best_score}, time={time.time()-start_time:.1f}s")
    
    if best_cycle is None:
        return list(range(_N))
    
    return best_cycle
# EVOLVE-BLOCK-END
