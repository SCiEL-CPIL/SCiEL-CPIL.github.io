import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]
    
    def decompose_randomized(edges, time_limit):
        if not edges:
            return []
        
        best_pieces = None
        best_count = float('inf')
        
        # We will run multiple randomized greedy attempts
        # Each attempt:
        # 1. Shuffles the order in which we consider edges
        # 2. Greedily tries to close cycles
        
        attempts = 0
        while time.time() < time_limit:
            attempts += 1
            if attempts % 100 == 0 and time.time() > time_limit - 5:
                break
                
            # Randomize the "priority" of edges
            # We want to prioritize edges that are "hard" to place later?
            # Or just random order?
            # Let's sort edges by a random key, but bias towards high-degree vertices
            edge_list = list(edges)
            random.shuffle(edge_list)
            
            remaining = set(edges)
            pieces = []
            
            # Pre-compute adjacency for the current remaining set? 
            # That's expensive if we do it every step. 
            # Instead, we maintain a set of remaining edges and check membership.
            
            # To make the greedy cycle finding fast, we need to be able to 
            # quickly find a path from v back to u using only remaining edges.
            # This is a reachability problem.
            
            # Simple Greedy Cycle Finder:
            # For each edge (u, v) in the shuffled list:
            #   If (u, v) is still in remaining:
            #     Try to find a path from v to u in the graph of remaining edges (excluding (u,v))
            #     If found, and length >= 2 (so cycle length >= 3):
            #        Extract this cycle.
            #     Else:
            #        Keep (u, v) for now? Or take it as a singleton?
            #        Taking as singleton is safe but might be suboptimal.
            #        Better: Try to extend (u, v) to a path, and see if it can be closed later?
            #        That's complex.
            
            # Let's use a simpler heuristic:
            # 1. Identify all cycles in the current remaining graph using a fast heuristic (e.g., find any cycle).
            # 2. Remove the largest one.
            # This is what the current code does.
            
            # The "Randomized" part: 
            # Instead of finding the *longest* cycle (expensive), find a *random* cycle 
            # biased towards longer ones, or just find *any* cycle quickly and remove it.
            # Repeating this with different randomizations might lead to better global structures.
            
            # Even better: 
            # Use a "2-Swap" local search on the set of pieces?
            
            # Let's try: **Greedy Cycle Packing with Randomized Tie-Breaking**
            # 1. Build adjacency.
            # 2. While edges remain:
            #    a. Find a cycle. To do this, pick a random edge (u,v).
            #    b. Try to find a path from v to u using DFS/BFS on remaining edges.
            #    c. If found, remove that cycle.
            #    d. If not found, this edge cannot be part of a cycle in the current remaining graph?
            #       No, it might be part of a cycle that requires removing other edges first? 
            #       No, if no path exists from v to u in the remaining graph, then (u,v) is a bridge 
            #       or in a tree component. It MUST be a singleton.
            #       So: if no path v->u exists, take (u,v) as singleton.
            #    e. If path exists, we have a cycle. 
            #       Which cycle to take? The one found by the random DFS?
            #       We can run multiple DFSs and pick the longest one found within a small time budget.
            
            # Implementation of one attempt:
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            current_remaining = set(edges)
            current_pieces = []
            
            # To speed up path finding, we can use BFS/DFS.
            # Since n is small (<=30), BFS is very fast.
            
            while current_remaining:
                if time.time() > time_limit:
                    break
                    
                # Pick a random edge to start with? 
                # Or pick an edge incident to a high-degree vertex?
                # Let's pick a random edge from the remaining set.
                if not current_remaining:
                    break
                    
                # Heuristic: Pick an edge that is likely to be in a cycle.
                # Edges in bridges cannot be in cycles.
                # We can't easily identify bridges without Tarjan's, which is O(E).
                # Let's just pick a random edge.
                e = random.choice(list(current_remaining))
                u, v = e
                
                # Try to find a path from v to u in the graph formed by current_remaining - {e}
                # If such a path exists, it forms a cycle.
                # We want the *shortest* path? No, shortest path gives smallest cycle.
                # We want a cycle. Any cycle is fine for the count (1 piece per cycle).
                # But removing a small cycle might leave a large cycle intact.
                # Removing a large cycle might break up other potential cycles.
                # Usually, removing *any* cycle is good. 
                # However, if we remove a triangle, we might leave a 5-cycle. 
                # If we remove the 5-cycle, we might leave the triangle.
                # Both are 1 piece. 
                # The problem arises when cycles share edges. 
                # If we pick a cycle that shares edges with many other potential cycles, 
                # we might destroy them.
                
                # Let's find *all* simple cycles passing through (u,v) up to a small limit? 
                # Too expensive.
                
                # Let's just find *one* cycle. 
                # To be safe, let's find the *shortest* cycle through (u,v). 
                # Why? Because it's the most "local" and least likely to disrupt distant structure?
                # Or the *longest*? 
                # Let's try finding the shortest cycle via BFS.
                
                # BFS from v to u in G - (u,v)
                target = u
                start = v
                visited = {start}
                queue = [(start, [start])]
                found_path = None
                
                # Limit BFS depth to avoid huge searches? 
                # Max depth is n.
                
                while queue:
                    node, path = queue.pop(0)
                    if node == target:
                        found_path = path
                        break
                    for neighbor in adj[node]:
                        if neighbor in visited:
                            continue
                        # Check if edge (node, neighbor) is in remaining
                        edge_check = (min(node, neighbor), max(node, neighbor))
                        if edge_check == e:
                            continue # Cannot use the edge we are trying to close
                        if edge_check not in current_remaining:
                            continue
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
                        
                if found_path:
                    # Cycle is found_path + [u] (which is start of path's end? No, path ends at u)
                    # Path is v -> ... -> u.
                    # Cycle is v -> ... -> u -> v.
                    # Vertices: found_path (v...u)
                    # Edges: (v, next), ..., (last, u), (u, v)
                    cycle = found_path
                    # Remove edges of this cycle from current_remaining
                    for i in range(len(cycle) - 1):
                        a, b = cycle[i], cycle[i+1]
                        ce = (min(a, b), max(a, b))
                        if ce in current_remaining:
                            current_remaining.remove(ce)
                    # Remove (u, v)
                    if e in current_remaining:
                        current_remaining.remove(e)
                        
                    # Update adjacency
                    for a, b in [(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)] + [(u, v)]:
                        if b in adj[a]: adj[a].discard(b)
                        if a in adj[b]: adj[b].discard(a)
                        
                    current_pieces.append(cycle)
                else:
                    # No cycle found through this edge. 
                    # This edge is a bridge or in a tree.
                    # It must be a singleton.
                    current_pieces.append([u, v])
                    current_remaining.remove(e)
                    if v in adj[u]: adj[u].discard(v)
                    if u in adj[v]: adj[v].discard(u)
                    
            if len(current_pieces) < best_count:
                best_count = len(current_pieces)
                best_pieces = current_pieces
                
        return best_pieces if best_pieces else [list(e) for e in edges]

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time proportional to complexity
        time_limits = [30.0, 40.0, 60.0]
        
        current_limit = time_limits[i]
        # Ensure we don't exceed global deadline
        global_limit = deadline - time.time()
        if global_limit < current_limit:
            current_limit = max(1.0, global_limit)
            
        time_limit = time.time() + current_limit
        
        best_pieces = decompose_randomized(edges, time_limit)
        results.append(best_pieces)
        
        # If time is critical, fallback to trivial for remaining
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break
            
    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
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
                if len(set(piece)) != len(piece):
                    valid = False
                    break
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
