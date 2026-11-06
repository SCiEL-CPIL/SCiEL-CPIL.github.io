import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_longest_cycle_dfs(adj, n, start, deadline):
    """
    Finds the longest simple cycle starting from 'start' using iterative DFS.
    Returns the cycle as a list of vertices, or [] if none found.
    """
    best_cycle = []
    best_len = 0
    
    # Stack items: (path, visited_set)
    # We use a list of tuples to simulate recursion
    # To optimize, we can use a "color" array, but for N=30, set is fine.
    
    stack = [([start], {start})]
    
    while stack:
        if time.time() > deadline:
            break
        
        path, visited = stack.pop()
        u = path[-1]
        
        # Check if we can close the cycle
        if len(path) >= 3 and start in adj[u]:
            if len(path) > best_len:
                best_len = len(path)
                best_cycle = list(path)
                
        # Pruning: Can we possibly beat the best?
        # Max possible extension is n - len(visited)
        if len(path) + (n - len(visited)) <= best_len:
            continue
            
        # Get neighbors
        # Heuristic: Sort neighbors to find long cycles faster
        # Prefer neighbors that are NOT start, to keep path open
        # Prefer high degree neighbors?
        neighbors = []
        for v in adj[u]:
            if v not in visited:
                neighbors.append(v)
        
        if not neighbors:
            continue
            
        # Sort neighbors: 
        # 1. If v is start, it's a dead end for extension, but closes cycle. 
        #    We already handled closing. So we only consider v != start for extension.
        #    Wait, the loop above `if v not in visited` excludes start because start is in visited.
        #    So `neighbors` does NOT contain start.
        #    This means we will NEVER close the cycle in the DFS extension?
        #    Correct. The cycle is only detected when we check `start in adj[u]` at the top.
        #    So we just extend the path.
        
        # Sort by degree descending to explore "hub" nodes first?
        # Or randomize to find diverse cycles?
        # Let's sort by degree descending.
        neighbors.sort(key=lambda v: -len(adj[v]))
        
        # Push to stack in reverse order so that the best neighbor is popped first
        for v in reversed(neighbors):
            new_path = path + [v]
            new_visited = visited | {v}
            stack.append((new_path, new_visited))
            
    return best_cycle

def _decompose(edges, n, seed, time_limit=45.0):
    if not edges:
        return []

    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initial Greedy Decomposition
    remaining = set(edges)
    adj = defaultdict(list)
    for u, v in remaining:
        adj[u].append(v)
        adj[v].append(u)
        
    cycles = []
    singles = []
    
    # Greedy Phase
    while remaining and time.time() < deadline:
        # Pick a start vertex: highest degree
        vertices = [v for v in adj if adj[v]]
        if not vertices:
            break
        deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
        
        best_cycle = []
        # Try top 3 vertices
        for start in deg_sorted[:3]:
            c = _find_longest_cycle_dfs(adj, n, start, deadline)
            if len(c) > len(best_cycle):
                best_cycle = c
                
        if best_cycle and len(best_cycle) >= 3:
            # Remove cycle edges
            cycle_edges = set()
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(u, v), max(u, v))
                cycle_edges.add(e)
                
            # Verify
            if all(e in remaining for e in cycle_edges):
                for e in cycle_edges:
                    u, v = e
                    remaining.discard(e)
                    if v in adj[u]: adj[u].remove(v)
                    if u in adj[v]: adj[v].remove(u)
                cycles.append(best_cycle)
            else:
                # Error handling
                break
        else:
            # No cycle found, take a single edge
            # Pick edge with most common neighbors?
            best_edge = None
            best_score = -1
            for e in remaining:
                u, v = e
                common = len(set(adj[u]) & set(adj[v]))
                score = common * 3 + len(adj[u]) + len(adj[v])
                if score > best_score:
                    best_score = score
                    best_edge = e
                    
            if best_edge is None:
                best_edge = next(iter(remaining))
                
            u, v = best_edge
            remaining.discard(best_edge)
            if v in adj[u]: adj[u].remove(v)
            if u in adj[v]: adj[v].remove(u)
            singles.append([u, v])
            
    # Add remaining as singles
    for e in remaining:
        u, v = e
        if v in adj[u]: adj[u].remove(v)
        if u in adj[v]: adj[v].remove(u)
        singles.append([u, v])
        
    # Local Search Phase
    # Objective: Maximize sum(len(c) - 1)
    # Move: Replace a cycle C with a longer cycle C_new formed from C's edges + some singles?
    # For simplicity, we only allow C_new to be formed from C's edges.
    # This means we are just trying to find a longer cycle within the subgraph induced by C's edges.
    # But C was already the longest cycle found in the *whole* graph.
    # So this move will rarely succeed unless the initial greedy was very bad.
    
    # Better Move: 
    # Remove C. Add its edges to `remaining`.
    # Try to find a cycle C_new in the *entire* remaining graph (which now includes C's edges).
    # If C_new is longer than C, accept.
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        if iterations > 100: # Safety break
            break
            
        # Shuffle cycles to randomize order
        shuffled_cycles = cycles[:]
        random.Random(seed + iterations).shuffle(shuffled_cycles)
        
        for C in shuffled_cycles:
            if time.time() > deadline:
                break
                
            # Remove C
            cycles.remove(C)
            C_edges = set()
            for i in range(len(C)):
                u = C[i]
                v = C[(i + 1) % len(C)]
                e = (min(u, v), max(u, v))
                C_edges.add(e)
                
            # Add C's edges to remaining and adj
            for e in C_edges:
                u, v = e
                remaining.add(e)
                adj[u].append(v)
                adj[v].append(u)
                
            # Try to find a better cycle
            # We can try to find ANY cycle in the current graph that is longer than C
            # But finding the *longest* cycle is expensive.
            # Let's try to find a cycle starting from a vertex in C.
            best_new_cycle = []
            for start in C:
                if time.time() > deadline:
                    break
                c = _find_longest_cycle_dfs(adj, n, start, deadline)
                if len(c) > len(best_new_cycle):
                    best_new_cycle = c
                    
            if len(best_new_cycle) > len(C):
                # Accept new cycle
                new_edges = set()
                for i in range(len(best_new_cycle)):
                    u = best_new_cycle[i]
                    v = best_new_cycle[(i + 1) % len(best_new_cycle)]
                    e = (min(u, v), max(u, v))
                    new_edges.add(e)
                    
                # Verify all edges are in remaining
                if all(e in remaining for e in new_edges):
                    for e in new_edges:
                        u, v = e
                        remaining.discard(e)
                        if v in adj[u]: adj[u].remove(v)
                        if u in adj[v]: adj[v].remove(u)
                    cycles.append(best_new_cycle)
                    improved = True
                else:
                    # Restore C
                    for e in C_edges:
                        u, v = e
                        remaining.add(e)
                        adj[u].append(v)
                        adj[v].append(u)
            else:
                # Restore C
                for e in C_edges:
                    u, v = e
                    remaining.add(e)
                    adj[u].append(v)
                    adj[v].append(u)
                    
    # Final pieces
    pieces = []
    for c in cycles:
        pieces.append(c)
    for s in singles:
        pieces.append(s)
        
    # Add any remaining edges that weren't processed (should be none)
    for e in remaining:
        pieces.append(list(e))
        
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    start_time = time.time()
    total_deadline = start_time + 140

    for n, p, seed in specs:
        remaining_time = total_deadline - time.time()
        if remaining_time < 10:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _instance(n, p, seed)
        time_limit = min(remaining_time * 0.3, 50.0)
        pieces = _decompose(edges, n, seed, time_limit=time_limit)
        results.append(pieces)
        
    return results
