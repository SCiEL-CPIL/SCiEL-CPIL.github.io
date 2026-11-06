import time
import random
import sys
from collections import defaultdict
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]
    
    def decompose_graph(edges, time_limit):
        if not edges:
            return []
        
        # Build adjacency structure
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        pieces = []
        
        def get_cycle_from_path(path):
            # path is a list of vertices [v0, v1, ..., vk]
            # Check if edge (vk, v0) exists in remaining_edges
            if len(path) < 3:
                return None
            u, v = path[0], path[-1]
            # The closing edge must be in remaining_edges
            if (u, v) not in remaining_edges and (v, u) not in remaining_edges:
                return None
            
            # Verify all edges in path exist in remaining_edges
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                if (a, b) not in remaining_edges and (b, a) not in remaining_edges:
                    return None
            return path

        def remove_cycle_edges(cycle):
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i+1]
                if (u, v) in remaining_edges:
                    remaining_edges.remove((u, v))
                elif (v, u) in remaining_edges:
                    remaining_edges.remove((v, u))
            u, v = cycle[-1], cycle[0]
            if (u, v) in remaining_edges:
                remaining_edges.remove((u, v))
            elif (v, u) in remaining_edges:
                remaining_edges.remove((v, u))
            
            # Update adjacency
            for u in cycle:
                adj[u] = set()
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)

        def find_best_cycle():
            """
            Find a cycle that maximizes the number of edges removed.
            We prioritize longer cycles.
            Uses DFS with backtracking to find the longest cycle within time limits.
            """
            if not remaining_edges:
                return None
            
            # If the graph is small, we can try to find the longest cycle more aggressively.
            # For larger graphs, we rely on greedy heuristics.
            
            # Identify candidate start vertices: those with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None
            
            # Sort candidates by degree (descending) to find large cycles faster
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            best_cycle = None
            best_len = 0
            
            # Time check inside the search
            search_deadline = time.time() + 0.5 # 0.5s per cycle extraction attempt
            
            for start in candidates[:10]: # Limit starts to avoid explosion
                if time.time() > search_deadline:
                    break
                
                # DFS to find cycles starting from 'start'
                # We want the longest cycle.
                # Path: [start, ..., current]
                # To close, current must connect to start.
                
                visited = {start}
                path = [start]
                
                # Stack for DFS: (current, next_index, path_copy)
                # To save memory, we modify path in place and backtrack
                stack = [(start, 0)]
                # We need to track which edges are used in the path to avoid reusing them?
                # No, in a simple cycle, vertices are distinct, so edges are distinct.
                # We just need to ensure we don't visit a vertex twice (except start at the end).
                
                # Standard DFS for longest cycle is NP-hard, but N is small.
                # We will do a limited DFS.
                
                # Optimization: Only try to close the cycle if the path length is >= 3
                # and the closing edge exists.
                
                # Instead of full DFS which might be too slow, we do a "Greedy Longest Path"
                # that backtracks only if it hits a dead end.
                
                def dfs(current, path):
                    nonlocal best_cycle, best_len, search_deadline
                    
                    if time.time() > search_deadline:
                        return
                    
                    # Check if we can close the cycle
                    if len(path) >= 3:
                        if (current, start) in remaining_edges or (start, current) in remaining_edges:
                            # Valid cycle found
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = list(path)
                            # We could continue to find even longer cycles, 
                            # but for performance, we might stop or continue.
                            # Let's continue to try to find longer ones if time permits.
                    
                    # Extend path
                    for neighbor in adj[current]:
                        if neighbor == start:
                            # Only allowed if we are closing the cycle (handled above)
                            # If we haven't reached min length, we can't close.
                            # If we can close, we did.
                            # We should not add start to path again.
                            continue
                        if neighbor not in visited:
                            # Check if edge (current, neighbor) is in remaining_edges
                            # It must be, because adj is built from remaining_edges.
                            visited.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor, path)
                            path.pop()
                            visited.remove(neighbor)
                            if time.time() > search_deadline:
                                return

                # Reset state for this start
                visited = {start}
                path = [start]
                dfs(start, path)
                
            if best_cycle:
                return best_cycle
            
            # Fallback: Find any cycle (e.g., triangle)
            # If no cycle found by DFS (maybe time ran out or graph is a tree), 
            # check for simple triangles.
            for u in list(adj.keys()):
                if len(adj[u]) < 2:
                    continue
                for v in list(adj[u]):
                    if v <= u: continue
                    for w in list(adj[v]):
                        if w == u or w == v: continue
                        if (u, w) in remaining_edges or (w, u) in remaining_edges:
                            # Triangle u-v-w-u
                            cycle = [u, v, w]
                            return cycle
                            
            return None

        # Main loop: extract cycles
        while remaining_edges:
            if time.time() > time_limit:
                break
                
            cycle = find_best_cycle()
            
            if cycle:
                pieces.append(cycle)
                remove_cycle_edges(cycle)
            else:
                # No cycle found. 
                # Check if there are any edges left that form a cycle we missed?
                # If not, we must take edges.
                # Strategy: Take the edge that is part of the fewest potential cycles?
                # Or just take any edge.
                # To minimize pieces, we should group edges if possible.
                # If no cycle exists, the graph is a forest (or has isolated edges).
                # In a forest, we cannot form cycles. We must take edges individually.
                # Wait, if the graph is not a forest, there is a cycle.
                # So if find_best_cycle returns None, either:
                # 1. Time limit for search expired (unlikely if graph is small)
                # 2. The remaining graph is a forest.
                
                # If it's a forest, every edge is a separate piece.
                # We can just take all remaining edges as pieces.
                for e in list(remaining_edges):
                    pieces.append(list(e))
                    remaining_edges.remove(e)
                break

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time proportional to complexity
        # Graph 1 (N=20): Easy
        # Graph 2 (N=25): Medium
        # Graph 3 (N=30): Harder
        # Give more time to later graphs? Or equal?
        # Let's give 30s, 30s, 70s? No, 140s total.
        # Let's try 20s, 30s, 80s.
        time_limits = [20.0, 30.0, 80.0]
        
        current_limit = time_limits[i]
        # Ensure we don't exceed global deadline
        global_limit = deadline - time.time()
        if global_limit < current_limit:
            current_limit = max(1.0, global_limit)
            
        time_limit = time.time() + current_limit
        
        best_pieces = decompose_graph(edges, time_limit)
        results.append(best_pieces)
        
        # If time is critical, fallback to trivial for remaining
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break
            
    return results

if __name__ == "__main__":
    res = construct()
    # Quick sanity check
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        # Re-generate to check
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)
        
        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                # Check distinct vertices
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                # Check edges
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break
                
        if valid and used_edges != edge_set:
            valid = False
            
        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
