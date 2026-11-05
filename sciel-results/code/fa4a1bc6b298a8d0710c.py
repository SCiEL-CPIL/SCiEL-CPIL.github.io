import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycles(adj, n, max_len=10):
        cycles = []
        # adj: list of sets of neighbors
        # We need to find all simple cycles up to max_len
        # To avoid duplicates, we can enforce that the smallest vertex in the cycle is the start
        # and the second smallest is the second vertex? No, just standard DFS with visited set and start node constraint.
        
        for start in range(n):
            # DFS from start
            # Path: [start]
            # We only consider cycles where start is the smallest vertex in the cycle
            # to avoid counting the same cycle multiple times.
            
            def dfs(u, path, visited):
                if time.time() > deadline:
                    return
                if len(path) > max_len:
                    return
                # If we can close the cycle
                if len(path) >= 3 and start in adj[u]:
                    # Check if start is the smallest vertex in the cycle
                    if min(path) == start:
                        # Add cycle
                        cycle_edges = set()
                        for i in range(len(path)):
                            u1 = path[i]
                            u2 = path[(i+1)%len(path)]
                            cycle_edges.add(tuple(sorted((u1, u2))))
                        cycles.append(frozenset(cycle_edges))
                
                for v in adj[u]:
                    if v == start and len(path) < 3:
                        continue
                    if v not in visited:
                        # Only proceed if start is the smallest vertex so far
                        # This is a heuristic to reduce duplicates, but might miss some if not careful.
                        # Better: just find all and deduplicate later.
                        # For small n, finding all is fine.
                        visited.add(v)
                        path.append(v)
                        dfs(v, path, visited)
                        path.pop()
                        visited.remove(v)
                    elif v == start and len(path) >= 3:
                        # Close cycle
                        pass # Handled above
        
        # The above DFS is a bit tricky to implement correctly for all cycles without duplicates.
        # Let's use a simpler approach for small n:
        # Generate all subsets of vertices of size 3 to max_len? No, too slow.
        # Let's use NetworkX if available? No, standard library only.
        
        # Let's implement a standard cycle enumeration.
        cycles_set = set()
        for start in range(n):
            # Only start from the smallest vertex in the cycle to avoid duplicates
            # We will do a DFS and only record the cycle if start is the min vertex.
            
            def dfs2(u, path, visited):
                if time.time() > deadline:
                    return
                if len(path) > max_len:
                    return
                
                for v in adj[u]:
                    if v == start and len(path) >= 3:
                        # Found a cycle
                        if min(path) == start:
                            cycle_edges = set()
                            for i in range(len(path)):
                                u1 = path[i]
                                u2 = path[(i+1)%len(path)]
                                cycle_edges.add(tuple(sorted((u1, u2))))
                            cycles_set.add(frozenset(cycle_edges))
                    elif v > start and v not in visited:
                        # Ensure we don't go to vertices smaller than start, 
                        # and we don't revisit.
                        # This ensures start is the smallest vertex in the path.
                        visited.add(v)
                        path.append(v)
                        dfs2(v, path, visited)
                        path.pop()
                        visited.remove(v)
            
            dfs2(start, [start], {start})
        
        return list(cycles_set)

    def optimize_decomposition(edges, adj, n):
        if not edges:
            return []
        
        # Find candidate cycles
        candidates = find_cycles(adj, n, max_len=12)
        
        # Filter out cycles that use edges not in the graph? 
        # The graph is defined by edges, so adj is built from edges.
        # So all cycles found are valid.
        
        # Calculate weight for each candidate: len(cycle) - 1
        # We want to maximize sum of weights.
        
        # Greedy initialization
        selected = []
        used_edges = set()
        
        # Sort candidates by weight descending
        # Weight is len(cycle) - 1. Longer cycles are better.
        candidates.sort(key=lambda c: len(c), reverse=True)
        
        for c in candidates:
            if time.time() > deadline:
                break
            # Check if c conflicts with used_edges
            if c.isdisjoint(used_edges):
                selected.append(c)
                used_edges.update(c)
        
        # Local Search
        # Try to improve by swapping
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any unselected cycle
            for c in candidates:
                if c in selected:
                    continue
                if c.isdisjoint(used_edges):
                    # Can add
                    selected.append(c)
                    used_edges.update(c)
                    improved = True
                    # Restart loop or continue? Continue is fine, but might miss better swaps.
                    # Let's break and restart for simplicity
                    break
            
            if improved:
                continue
            
            # Try to swap one selected with one unselected
            # This is O(N^2) which might be slow.
            # For small graphs, it's fine.
            
            # We can also try to remove a cycle and see if we can add more.
            # For now, simple swap.
            
            # To speed up, we can pre-filter candidates that conflict.
            
            # Let's try a simple hill climb:
            # For each selected cycle S, try to remove it and see if we can add any unselected cycle U
            # such that the new total weight is higher.
            
            # Actually, since we want to maximize weight, and weight is monotonic with length,
            # replacing a short cycle with a longer one is good.
            
            # Let's stick to the "add if possible" and "swap if better" logic.
            # Swap logic:
            # For each i in selected:
            #   For each j in candidates (not in selected):
            #     If selected[i] and candidates[j] are disjoint? No, we remove selected[i] first.
            #     Let new_used = used_edges - selected[i]
            #     If candidates[j].isdisjoint(new_used):
            #        old_weight = len(selected[i]) - 1
            #        new_weight = len(candidates[j]) - 1
            #        If new_weight > old_weight:
            #           Swap
            
            # This is expensive. Let's limit the number of iterations.
            
            # A simpler improvement: 
            # If we have a cycle of length 3, and we can replace it with a cycle of length 4, that's good.
            
            # Given the time limit, the greedy + add-if-possible is likely sufficient for a big jump.
            # The baseline is 1 piece per edge.
            # Even a few cycles will reduce the count significantly.
            
            # Let's add a time check inside the swap loop if we implement it.
            
            # For now, let's just do the greedy and the "add if possible" loop.
            # The "add if possible" loop will run until no more can be added.
            # This is a maximal packing.
            
            # To get a better packing, we can randomize the order of candidates and repeat.
            
            # Let's implement a simple restart strategy.
            pass
            
        # Convert selected cycles to list of vertices
        result = []
        for c in selected:
            # Reconstruct vertex list from edge set
            # This is non-trivial. 
            # We stored frozenset of edges.
            # We need to return a list of vertices [v1, v2, ..., vk].
            # We can just return the edges? No, the task says "cycle given by at least three distinct vertices".
            # So we must return vertices.
            
            # To get vertices from edges, we can build a small graph and find the cycle.
            # Since it's a simple cycle, each vertex has degree 2 in this subgraph.
            edges_list = list(c)
            if not edges_list:
                continue
            
            # Build adjacency for this cycle
            adj_c = {u: [] for u in range(n)}
            for u, v in edges_list:
                adj_c[u].append(v)
                adj_c[v].append(u)
            
            # Find a vertex with degree 2 (all should be)
            start_v = None
            for v in range(n):
                if len(adj_c[v]) == 2:
                    start_v = v
                    break
            
            if start_v is None:
                continue # Should not happen
            
            # Trace the cycle
            path = [start_v]
            prev = -1
            curr = start_v
            while True:
                nexts = [x for x in adj_c[curr] if x != prev]
                if not nexts:
                    break
                next_v = nexts[0]
                if next_v == start_v and len(path) > 1:
                    break
                path.append(next_v)
                prev = curr
                curr = next_v
            
            if len(path) >= 3:
                result.append(path)
        
        # Add remaining edges as single-edge pieces
        for e in edges:
            if e not in used_edges:
                result.append(list(e))
                
        return result

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        res = optimize_decomposition(edges, adj, n)
        results.append(res)
        
        if time.time() > deadline:
            # If we run out of time, return what we have
            break
            
    # If we broke early, fill remaining with baseline
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([list(e) for e in edges])
        
    return results

from random import Random
