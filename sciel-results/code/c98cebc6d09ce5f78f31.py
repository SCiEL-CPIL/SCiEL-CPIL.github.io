"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import defaultdict

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose_greedy(edges_list, deadline):
    """
    Decompose graph into cycles and edges using a greedy strategy:
    1. Build adjacency from edges.
    2. Repeatedly find a long cycle (using DFS with heuristics to prefer longer paths).
    3. Remove cycle edges.
    4. Remaining edges become single-edge pieces.
    
    We use a modified DFS that tries to find the longest possible cycle from a start node
    by exploring neighbors in order of degree (high to low) to encourage longer paths.
    """
    if not edges_list:
        return []
    
    adj = defaultdict(set)
    for u, v in edges_list:
        adj[u].add(v)
        adj[v].add(u)
    
    # Work on a copy of the edge set for quick removal
    remaining = set((min(u, v), max(u, v)) for u, v in edges_list)
    
    def remove_edge(u, v):
        key = (min(u, v), max(u, v))
        if key in remaining:
            remaining.discard(key)
            adj[u].discard(v)
            adj[v].discard(u)
    
    pieces = []
    
    # Try to find cycles
    # We will iteratively find cycles until no more can be found
    # To find "long" cycles, we use a heuristic DFS
    
    # Sort nodes by degree for potential starting points
    # But we restart the search after each removal, so we just check all nodes with deg >= 2
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Find a node with degree >= 2
        start_candidates = [u for u in adj if len(adj[u]) >= 2]
        if not start_candidates:
            break
        
        # Sort by degree descending to find potentially longer cycles
        start_candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        
        found_cycle = None
        best_cycle_len = 0
        
        # Try each candidate start
        for start in start_candidates:
            if len(adj[start]) < 2:
                continue
            
            # DFS to find a cycle back to start
            # We want to maximize the length of the cycle.
            # We use a recursive DFS with backtracking, but we limit the search depth or time
            # to avoid exponential blowup. Since n<=30, it's usually fine, but we need to be careful.
            # Actually, finding the LONGEST cycle is NP-hard. We settle for a heuristic:
            # DFS that prefers unvisited nodes with higher degree.
            
            # Let's do a simple DFS that finds the first cycle, but we can try multiple starts.
            # A better heuristic: 
            # 1. Start from 'start'.
            # 2. Always move to a neighbor with the highest degree that hasn't been visited in the current path.
            # 3. If stuck, backtrack.
            
            # This is essentially a heuristic search for a Hamiltonian cycle in the subgraph induced by remaining edges?
            # No, just a long cycle.
            
            # Let's try a few different strategies for finding a good cycle from 'start':
            # Strategy: DFS with neighbor ordering by degree.
            
            best_path_for_start = []
            
            def dfs_heuristic(node, path):
                nonlocal best_path_for_start
                if time.time() > deadline:
                    return
                if len(path) > len(best_path_for_start):
                    if node == start and len(path) >= 4: # cycle length >= 3
                        # Check if closing edge exists
                        last = path[-2]
                        if (min(last, start), max(last, start)) in remaining:
                            best_path_for_start = path[:]
                            # Don't return immediately, maybe a longer one exists?
                            # But backtracking can be expensive.
                            # Let's just record it and continue if we have time.
                            # For simplicity, let's just return the first good one found with this heuristic?
                            # Or continue to see if we can find a longer one.
                            # Given the small size, let's try to continue a bit.
                            pass
                            
                # Order neighbors by degree descending
                neighbors = sorted(adj[node], key=lambda x: len(adj[x]), reverse=True)
                for neighbor in neighbors:
                    if neighbor == start and len(path) >= 3:
                        # Found a cycle
                        if len(path) > len(best_path_for_start):
                            last = path[-1]
                            if (min(last, start), max(last, start)) in remaining:
                                best_path_for_start = path[:]
                                # Continue searching?
                                # To save time, let's just break out of this neighbor loop?
                                # No, we want the longest.
                                continue
                    if neighbor not in path:
                        dfs_heuristic(neighbor, path + [neighbor])
            
            dfs_heuristic(start, [start])
            
            if len(best_path_for_start) > best_cycle_len:
                best_cycle_len = len(best_path_for_start)
                found_cycle = best_path_for_start
            
            # If we found a very long cycle, we might stop early?
            # But we don't know the max possible.
        
        if found_cycle:
            # Verify and add
            valid = True
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i + 1) % len(found_cycle)]
                if (min(u, v), max(u, v)) not in remaining:
                    valid = False
                    break
            if valid and len(found_cycle) >= 3:
                pieces.append(found_cycle)
                for i in range(len(found_cycle)):
                    u = found_cycle[i]
                    v = found_cycle[(i + 1) % len(found_cycle)]
                    remove_edge(u, v)
                continue
            else:
                # Should not happen if logic is correct, but just in case
                break
        
        # No more cycles found
        break
    
    # Remaining edges are single pieces
    for u, v in remaining:
        pieces.append([u, v])
    
    return pieces

def _decompose_improved(edges_list, deadline):
    """
    Improved decomposition:
    1. Find all cycles using a more robust method.
    2. Use a "path extension" greedy approach:
       - Find any cycle.
       - Try to extend it? No, cycles are fixed.
    
    Alternative: 
    Use a simple greedy cycle packing:
    1. While there is a cycle in the graph:
       a. Find a cycle (e.g., using DFS).
       b. Remove it.
    2. To improve quality, try to find LONGER cycles.
    
    The previous _decompose_greedy does this. 
    Let's try a different structural approach:
    **Minimum Feedback Arc Set / Cycle Basis?** No.
    
    **Local Search / Simulated Annealing?**
    Start with a valid decomposition (e.g., all edges).
    Try to merge two paths into a cycle?
    Or merge a cycle and a path?
    
    Let's stick to the greedy cycle finding but make it faster and try multiple random restarts/orderings if time permits.
    """
    if not edges_list:
        return []
    
    # Try the greedy approach
    result = _decompose_greedy(edges_list, deadline)
    
    # If we have time, try to improve?
    # For now, just return the greedy result.
    return result

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    start_time = time.time()
    total_deadline = start_time + 140.0  # Safety margin from 150s limit
    
    for spec in specs:
        # Allocate time for each graph
        # 3 graphs, so ~46s each
        graph_deadline = min(total_deadline - 5.0, start_time + (total_deadline - start_time) / 3.0)
        
        edges = _instance(*spec)
        pieces = _decompose_improved(edges, graph_deadline)
        results.append(pieces)
        
        # Update start_time for next allocation? No, just use absolute deadline.
        # But we should ensure we don't exceed total_deadline.
    
    return results
# EVOLVE-BLOCK-END
