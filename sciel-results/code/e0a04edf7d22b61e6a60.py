import time
import random
import sys
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycles_in_matching(n, matching_edges):
    """
    Given a set of directed edges (u, v) representing a partial permutation,
    find all directed cycles.
    Returns a list of cycles, where each cycle is a list of vertices.
    """
    # Build adjacency from matching edges
    # Each node has out-degree <= 1 and in-degree <= 1
    next_node = [-1] * n
    prev_node = [-1] * n
    for u, v in matching_edges:
        next_node[u] = v
        prev_node[v] = u
    
    visited = [False] * n
    cycles = []
    
    for start in range(n):
        if visited[start]:
            continue
        if next_node[start] == -1:
            # Not part of a cycle (start of a path or isolated)
            # Mark as visited to skip
            visited[start] = True
            continue
            
        # Trace the path
        path = []
        current = start
        while not visited[current]:
            visited[current] = True
            path.append(current)
            current = next_node[current]
            if current == -1:
                break
        
        # Check if we formed a cycle
        # If current == start, we closed the loop.
        # But the loop might have started earlier in the path.
        # Actually, since we mark visited, if we hit a visited node, 
        # it means we entered an existing cycle or a path that ended.
        
        # Let's re-evaluate.
        # In a functional graph (out-degree <= 1), every component is a cycle with trees feeding into it.
        # Since in-degree <= 1 as well (it's a matching), the "trees" are just paths.
        # So components are either:
        # 1. A cycle
        # 2. A path (no cycle)
        
        # If the path we traced leads back to a node in the current path, we found a cycle.
        # But we marked visited.
        
        # Better approach for matching:
        # Just follow the chain. If we return to a node in the current chain, it's a cycle.
        # If we hit -1 or a visited node not in current chain, it's a path.
        
        # Let's restart the logic for clarity.
        pass

    # Simpler implementation:
    # For each node, follow next_node.
    # If we hit a node already in the current trail, extract the cycle.
    
    visited_global = [False] * n
    for start in range(n):
        if visited_global[start]:
            continue
        if next_node[start] == -1:
            visited_global[start] = True
            continue
            
        trail = []
        current = start
        while not visited_global[current]:
            visited_global[current] = True
            trail.append(current)
            current = next_node[current]
            if current == -1:
                break
            
        if current != -1:
            # We hit a visited node.
            # If current is in trail, we found a cycle.
            if current in trail:
                idx = trail.index(current)
                cycle = trail[idx:]
                if len(cycle) >= 2: # In directed matching, len 2 is a 2-cycle (edge)
                    cycles.append(cycle)
    
    return cycles

def _decompose_2factor(edges, n, deadline, rng=None):
    """
    Decompose using iterative maximum bipartite matching to find 2-factors (sets of disjoint cycles).
    """
    if rng is None:
        rng = random.Random(42)
        
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    pieces = []
    remaining_edges = set(tuple(sorted(e)) for e in edges)
    
    while remaining_edges:
        if time.time() > deadline:
            break
            
        # Build cost matrix for Hungarian Algorithm
        # Cost 0 if edge exists, 1 if not.
        # We want to minimize cost, which maximizes the number of edges used in the matching.
        cost = [[1 for _ in range(n)] for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if v in adj[u]:
                    cost[u][v] = 0
                    
        # Find optimal assignment
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except:
            # Fallback if Hungarian fails
            break
            
        matched_edges = []
        for u, v in zip(row_ind, col_ind):
            if cost[u][v] == 0:
                matched_edges.append((u, v))
                
        if not matched_edges:
            # No edges matched, meaning no 2-factor structure found?
            # This shouldn't happen if there are edges, unless the graph is very sparse 
            # and the matching prefers non-edges? No, cost 0 is better than 1.
            # If there are edges, at least one should be matched.
            # If matched_edges is empty, it means no edge was chosen.
            # This implies the graph has no edges? But remaining_edges is not empty.
            # Ah, linear_sum_assignment returns a perfect matching.
            # If the graph is not complete, it will pick non-edges (cost 1).
            # We filter for cost 0.
            # If no cost 0 edges are picked, it means no edge in the graph was part of the optimal assignment?
            # This is possible if the graph is a matching itself (disjoint edges).
            # In that case, the optimal assignment picks all edges (cost 0).
            # So matched_edges should not be empty if remaining_edges is not empty.
            
            # Fallback: Find any cycle greedily
            best_cycle = None
            # Simple greedy cycle finder
            for start in range(n):
                if not adj[start]:
                    continue
                # DFS
                stack = [(start, [start])]
                visited_dfs = {start}
                while stack:
                    node, path = stack.pop()
                    for neighbor in list(adj[node]):
                        if neighbor == start and len(path) >= 3:
                            if not best_cycle or len(path) > len(best_cycle):
                                best_cycle = path[:]
                            break
                        elif neighbor not in visited_dfs:
                            visited_dfs.add(neighbor)
                            stack.append((neighbor, path + [neighbor]))
                
            if best_cycle:
                m = len(best_cycle)
                for i in range(m):
                    a = best_cycle[i]
                    b = best_cycle[(i + 1) % m]
                    edge = tuple(sorted((a, b)))
                    remaining_edges.discard(edge)
                    adj[a].discard(b)
                    adj[b].discard(a)
                pieces.append(best_cycle)
            else:
                # No cycle, take a single edge
                if remaining_edges:
                    u, v = next(iter(remaining_edges))
                    remaining_edges.discard((u, v))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    pieces.append([u, v])
                else:
                    break
            continue
            
        # Find cycles in the matched edges
        cycles = _find_cycles_in_matching(n, matched_edges)
        
        if not cycles:
             # The matching formed only paths, no cycles.
             # This can happen if the graph is a tree or similar.
             # In that case, we cannot extract any cycles from this matching.
             # We should fall back to greedy or single edge.
             # But wait, if the graph has a cycle, a maximum matching should find it?
             # Not necessarily. A maximum matching maximizes the number of edges, not the number of cycles.
             # If we have a path 1-2-3-4, matching might be (1,2), (3,4). No cycles.
             # In this case, we should just take a single edge or use greedy.
             
             # Let's use greedy to find one cycle if matching yields none.
             best_cycle = None
             for start in range(n):
                 if not adj[start]:
                     continue
                 stack = [(start, [start])]
                 visited_dfs = {start}
                 while stack:
                     node, path = stack.pop()
                     for neighbor in list(adj[node]):
                         if neighbor == start and len(path) >= 3:
                             if not best_cycle or len(path) > len(best_cycle):
                                 best_cycle = path[:]
                             break
                         elif neighbor not in visited_dfs:
                             visited_dfs.add(neighbor)
                             stack.append((neighbor, path + [neighbor]))
                             
             if best_cycle:
                 m = len(best_cycle)
                 for i in range(m):
                     a = best_cycle[i]
                     b = best_cycle[(i + 1) % m]
                     edge = tuple(sorted((a, b)))
                     remaining_edges.discard(edge)
                     adj[a].discard(b)
                     adj[b].discard(a)
                 pieces.append(best_cycle)
             else:
                 if remaining_edges:
                     u, v = next(iter(remaining_edges))
                     remaining_edges.discard((u, v))
                     adj[u].discard(v)
                     adj[v].discard(u)
                     pieces.append([u, v])
                 else:
                     break
             continue
             
        # Extract cycles
        for cycle in cycles:
            # Check if it's a 2-cycle (edge) or longer
            # In the directed matching, a 2-cycle is u->v and v->u.
            # This corresponds to the undirected edge (u,v).
            # A cycle of length k >= 3 is a valid cycle.
            
            # Remove edges from graph
            m = len(cycle)
            for i in range(m):
                a = cycle[i]
                b = cycle[(i + 1) % m]
                edge = tuple(sorted((a, b)))
                # Only remove if it exists in remaining_edges
                # It should, because it was in the matching which was based on adj
                if edge in remaining_edges:
                    remaining_edges.discard(edge)
                    adj[a].discard(b)
                    adj[b].discard(a)
                    
            if m == 2:
                pieces.append([cycle[0], cycle[1]])
            else:
                pieces.append(cycle)
                
    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    # Try 2-factor decomposition
    if time.time() < deadline - 2.0:
        pieces = _decompose_2factor(edges, n, deadline - 1.0)
        if pieces:
            best_count = len(pieces)
            best = pieces
        else:
            # Fallback
            best = [[u, v] for u, v in edges]
            best_count = len(best)
            
    # If 2-factor didn't work well or didn't run, try greedy
    # But 2-factor is generally superior.
    # We can add a greedy pass to see if it's better, but 2-factor is hard to beat.
    
    return best

def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results
