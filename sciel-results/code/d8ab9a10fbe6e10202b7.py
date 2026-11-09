"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import random
import heapq
from collections import defaultdict, deque

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        pieces = []
        remaining_edges = set(edges)
        
        while True:
            # Build adjacency list from remaining edges
            adj = defaultdict(list)
            for u, v in remaining_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Identify active nodes (degree > 0)
            active_nodes = [n for n in adj if adj[n]]
            if not active_nodes:
                break
            
            # Heuristic: Try to find a long cycle
            # Sort nodes by degree descending to prioritize high-degree starts
            active_nodes_sorted = sorted(active_nodes, key=lambda x: len(adj[x]), reverse=True)
            
            best_cycle_found = None
            
            for start in active_nodes_sorted:
                if not adj[start]:
                    continue
                
                # DFS to find a cycle, keeping track of the path
                visited = set()
                parent = {}
                path = []
                
                def dfs(u):
                    nonlocal best_cycle_found
                    visited.add(u)
                    path.append(u)
                    
                    for v in adj[u]:
                        if v not in visited:
                            parent[v] = u
                            dfs(v)
                            # Early exit if we found a sufficiently long cycle
                            if best_cycle_found and len(best_cycle_found) > 12:
                                return True
                        else:
                            # If v is already in the current path, we found a cycle
                            if v in path:
                                idx = path.index(v)
                                cyc = path[idx:]
                                if len(cyc) >= 3:
                                    if best_cycle_found is None or len(cyc) > len(best_cycle_found):
                                        best_cycle_found = cyc[:]
                    
                    path.pop()
                    visited.remove(u)
                    return False
                
                dfs(start)
                
                # If we found a good cycle, stop searching for others
                if best_cycle_found and len(best_cycle_found) > 10:
                    break
            
            if best_cycle_found:
                cycle = best_cycle_found
            else:
                # Fallback: Find any cycle if the heuristic didn't find a "long" one
                # This can happen if the graph has cycles but they are all short
                visited = set()
                parent = {}
                found = None
                
                def dfs_any(u):
                    nonlocal found
                    visited.add(u)
                    for v in adj[u]:
                        if v not in visited:
                            parent[v] = u
                            if dfs_any(v):
                                return True
                        elif v != parent.get(u, -1):
                            # Found a cycle
                            cyc = [u]
                            curr = u
                            while curr != v:
                                curr = parent[curr]
                                cyc.append(curr)
                            cyc.reverse()
                            found = cyc
                            return True
                    return False
                
                for n in active_nodes:
                    if n not in visited:
                        if dfs_any(n):
                            break
                
                if found:
                    cycle = found
                else:
                    break # No more cycles
            
            # Remove edges in the found cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                remaining_edges.discard((min(u, v), max(u, v)))
            
            pieces.append(cycle)
        
        # Add remaining edges as individual pieces
        for u, v in remaining_edges:
            pieces.append([u, v])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
