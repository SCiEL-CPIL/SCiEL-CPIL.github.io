"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_sa(edges, time_limit=10.0):
        if not edges:
            return []
        
        m = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        
        # Build adjacency for quick neighbor checks
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Vertices involved
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
            
        # Initial assignment: each edge gets a unique path ID
        # This is a valid decomposition (m paths)
        color = np.arange(m, dtype=int)
        
        def get_paths(c):
            """Reconstruct paths from color array. Returns list of paths (vertex lists)."""
            # Group edges by color
            color_edges = defaultdict(list)
            for i in range(m):
                color_edges[c[i]].append(edges[i])
                
            paths = []
            for col, eds in color_edges.items():
                # Build graph for this color class
                # Check validity: each vertex degree <= 2, no cycles
                deg = defaultdict(int)
                adj_col = defaultdict(list)
                for u, v in eds:
                    deg[u] += 1
                    deg[v] += 1
                    adj_col[u].append(v)
                    adj_col[v].append(u)
                
                # Check degree constraint
                valid = True
                for v in deg:
                    if deg[v] > 2:
                        valid = False
                        break
                if not valid:
                    # This state is invalid. In SA, we handle this by penalty or rejection.
                    # For reconstruction in valid states, this won't happen.
                    # But we need to return *something* if called on invalid state?
                    # We only call get_paths on the best valid state found.
                    pass
                    
                # Extract paths from valid color class
                # Find start vertices (degree 1) or any vertex if cycle (but cycles are invalid)
                # Since no cycles and deg<=2, components are paths.
                visited_edges = set()
                for u, v in eds:
                    if (u, v) in visited_edges or (v, u) in visited_edges:
                        continue
                    # Start a new path from u
                    # We need to traverse.
                    # Find endpoint: a vertex with degree 1 in this component.
                    # Or just walk from u.
                    # Let's find a vertex with degree 1 in the whole component.
                    # Simpler: walk from u.
                    # Since it's a path, one end is u or v.
                    # Let's just do a DFS/BFS on the component.
                    
                    # Get component vertices
                    comp_vertices = set()
                    stack = [u]
                    while stack:
                        cur = stack.pop()
                        if cur in comp_vertices:
                            continue
                        comp_vertices.add(cur)
                        for nb in adj_col[cur]:
                            if nb not in comp_vertices:
                                stack.append(nb)
                    
                    # Find start node (degree 1)
                    start_node = None
                    for v in comp_vertices:
                        if deg[v] == 1:
                            start_node = v
                            break
                    if start_node is None:
                        # Should not happen for valid path decomposition (no cycles)
                        # But if it does, pick any
                        start_node = next(iter(comp_vertices))
                        
                    # Walk
                    path = [start_node]
                    prev = None
                    cur = start_node
                    while True:
                        # Find next node
                        next_node = None
                        for nb in adj_col[cur]:
                            if nb != prev:
                                next_node = nb
                                break
                        if next_node is None:
                            break
                        path.append(next_node)
                        prev, cur = cur, next_node
                        
                    # Mark edges as visited
                    for i in range(len(path)-1):
                        e1 = (path[i], path[i+1])
                        e2 = (path[i+1], path[i])
                        if e1 in edge_index:
                            visited_edges.add(e1)
                        if e2 in edge_index:
                            visited_edges.add(e2)
                            
                    paths.append(path)
            return paths

        def is_valid(c):
            """Check if color assignment c forms a valid path decomposition."""
            # 1. Each color class must have max degree <= 2
            # 2. Each color class must be acyclic
            # We can check this efficiently by maintaining counts? 
            # For simplicity in SA, we check on the fly or use a penalty function.
            # Let's use a penalty function approach for speed, and only validate the final best.
            # Actually, checking validity for every step is expensive.
            # Alternative: Move-based validation.
            # When we change color of edge e from c1 to c2:
            # - Check if c1 remains valid (removing edge might break connectivity? No, paths can be split. Removing an edge from a path splits it into two paths or leaves one. Degree drops. So c1 remains valid if it was valid before, UNLESS we created a cycle? No, removing edges doesn't create cycles. So c1 is always valid after removal if it was valid before.)
            # - Check if c2 becomes valid (adding edge). Adding edge e=(u,v) to color c2:
            #   - deg[u] and deg[v] must be < 2 in c2.
            #   - Adding e must not create a cycle in c2.
            
            # So we can maintain validity incrementally.
            # However, implementing incremental validity in SA with numpy arrays is tricky.
            # Given m is small (< 200), we can just re-validate the affected color classes.
            
            # Let's implement a fast validator for a specific color class given the color array.
            # This is O(m) per check if we just scan.
            # Better: maintain degree counts per color per vertex?
            # Let's stick to a simpler penalty-based SA that allows invalid states but penalizes them heavily,
            # and only accepts moves that don't increase penalty if possible, or uses temperature.
            
            # Actually, let's try a different approach: 
            # Local Search with strict validity.
            # Only accept moves that keep the solution valid.
            
            pass

        # Let's implement a strict local search / SA with validity checks.
        
        # Precompute edge endpoints
        e_u = np.array([e[0] for e in edges])
        e_v = np.array([e[1] for e in edges])
        
        # State: color array
        # Validity check for a move: change color of edge i from c_old to c_new.
        # 1. c_old: removing edge i.
        #    - Degrees of u,v decrease. Validity is preserved (no new cycles, degrees drop).
        # 2. c_new: adding edge i.
        #    - Degrees of u,v increase. Must be <= 2.
        #    - Must not create a cycle in the subgraph of color c_new.
        
        # To check cycle in c_new quickly:
        # We need to know if u and v are already connected in the subgraph of color c_new.
        # This requires connectivity checks.
        # For small graphs, we can do BFS/DFS on the subgraph of color c_new.
        # Size of subgraph for a color is small (path length).
        
        def check_add_valid(c, i):
            """Check if adding edge i to color c[i] is valid."""
            u = edges[i][0]
            v = edges[i][1]
            
            # Check degrees
            # Count edges in color c[i] incident to u and v
            # This is slow if done from scratch.
            # Let's maintain deg[color][vertex] counts?
            # That's complex.
            
            # Given the small size, let's just extract the edges of color c[i] and check.
            mask = (color == c)
            # Exclude edge i itself if it's already in that color? 
            # Wait, in the move, we are CHANGING color.
            # So currently color[i] is c_old. We want to set color[i] = c_new.
            # So we check validity of c_new WITH edge i added.
            # Current state has color[i] = c_old.
            # So the current subgraph of c_new does NOT include edge i.
            
            # Find all edges j != i such that color[j] == c_new
            # This is O(m).
            # Then check if u and v are connected in this subgraph.
            # And check degrees of u and v in this subgraph.
            
            # Optimization: Precompute adjacency lists for each color?
            # Too much overhead.
            
            # Let's just do it. m is ~100-200.
            # Number of moves per step: 1.
            # Steps: ~10000.
            # 10000 * 200 * 200 (BFS worst case) = 400M ops? Too slow in Python.
            
            # We need a faster validity check.
            # Maintain deg_count[color][vertex] and DSU for each color?
            # DSU with rollback is hard.
            
            # Alternative: 
            # Use a penalty function. 
            # Energy = num_paths + lambda * (num_violations)
            # Violations: vertices with deg > 2 in a color, or cycles in a color.
            # Checking cycles is the hard part.
            
            # Let's try a different strategy: 
            # Greedy with backtracking / Beam Search was the seed.
            # What about Integer Linear Programming?
            # No, no external solvers.
            
            # Let's go back to the Beam Search idea but improve it.
            # The seed's beam search was O(m * beam * n).
            # Can we make it faster?
            # 
            # Actually, let's look at the "Inspiration" note:
            # "Greedy extraction ... leveraging Warnsdorff's heuristic ... Measured: 1 -> 8.606"
            # This suggests a greedy approach with a good heuristic (Warnsdorff) works well.
            # The seed program used a simple beam search.
            # Let's implement a Greedy approach with Warnsdorff's rule.
            # Warnsdorff's rule: move to the neighbor with the fewest onward moves.
            # This is typically for Hamiltonian paths, but can be adapted for longest path in sparse graphs.
            
            return [] # Placeholder, will rewrite

    def solve_greedy_warnsdorff(edges, max_restarts=50, time_limit=10.0):
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        for restart in range(max_restarts):
            if time.time() > start_time + time_limit:
                break
                
            rng = random.Random(restart * 1234 + 5678)
            current_edges = set(edges)
            paths = []
            
            while current_edges:
                # Choose start vertex
                # Heuristic: Vertex with highest degree in current_edges
                # Or random?
                # Let's try highest degree, tie-break random.
                degs = defaultdict(int)
                for u, v in current_edges:
                    degs[u] += 1
                    degs[v] += 1
                
                if not degs:
                    break
                    
                max_deg = max(degs.values())
                candidates = [v for v, d in degs.items() if d == max_deg]
                start_v = rng.choice(candidates)
                
                # Build path using Warnsdorff's heuristic
                path = [start_v]
                visited = {start_v}
                last_v = start_v
                
                while True:
                    # Find neighbors in current_edges not visited
                    neighbors = []
                    for nb in adj[last_v]:
                        if nb not in visited:
                            e = tuple(sorted((last_v, nb)))
                            if e in current_edges:
                                # Calculate onward moves for nb
                                # Number of unvisited neighbors of nb in current_edges
                                onward = 0
                                for nn in adj[nb]:
                                    if nn not in visited and nn != last_v:
                                        e2 = tuple(sorted((nb, nn)))
                                        if e2 in current_edges:
                                            onward += 1
                                neighbors.append((onward, nb))
                    
                    if not neighbors:
                        break
                    
                    # Sort by onward moves (ascending)
                    neighbors.sort(key=lambda x: x[0])
                    
                    # Pick the one with fewest onward moves
                    # Tie-break: random or highest degree?
                    # Let's pick randomly among the best few if tied?
                    # Or just the first one.
                    # To add diversity, we can pick randomly from the top 2 if tied.
                    min_onward = neighbors[0][0]
                    best_nbs = [nb for on, nb in neighbors if on == min_onward]
                    next_v = rng.choice(best_nbs)
                    
                    path.append(next_v)
                    visited.add(next_v)
                    e = tuple(sorted((last_v, next_v)))
                    current_edges.discard(e)
                    last_v = next_v
                
                if len(path) > 1:
                    paths.append(path)
                else:
                    # Isolated vertex? Shouldn't happen if edges exist.
                    # But if we get stuck immediately, it means start_v had no valid moves?
                    # No, start_v was chosen from degs, so it has edges.
                    # Wait, if start_v has degree 1, and we pick it, we move to its neighbor.
                    # Then from neighbor, if no onward moves, we stop. Path length 2.
                    # So len(path) >= 2 always if edges exist.
                    break
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                
            # Early stop if we hit a very good score?
            # Lower bound: ceil(sum(deg(v)-1) / 2) ? No.
            # Gallai: n/2 + 1.
            # For n=20, LB=11. For n=25, LB=13. For n=30, LB=15.
            # Total LB = 39? No, 11+13+15 = 39.
            # Wait, Gallai's conjecture says p(G) <= floor(n/2).
            # For n=20, <= 10.
            # For n=25, <= 12.
            # For n=30, <= 15.
            # Total <= 37.
            # The prompt says "Gallai's conjecture predicts at most 10+13+15 = 38".
            # Let's check the prompt's numbers.
            # n=20 -> 10. n=25 -> 13? floor(25/2) = 12. Why 13?
            # Maybe the prompt implies something else or I am misremembering.
            # "Gallai's conjecture predicts at most 10+13+15 = 38"
            # If we can get close to 38, that's good.
            # If we get < 38, that's great.
            
            if best_count <= 38:
                # We are within the conjecture bound.
                # Can we do better?
                # Let's continue searching a bit more if time allows.
                pass
                
        if best_paths is None:
            # Fallback: simple greedy
            return greedy_simple(edges)
            
        return best_paths

    def greedy_simple(edges):
        # Same as seed's greedy_solve but cleaned up
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        current_edges = set(edges)
        paths = []
        
        while current_edges:
            degs = defaultdict(int)
            for u, v in current_edges:
                degs[u] += 1
                degs[v] += 1
            
            if not degs:
                break
                
            max_deg = max(degs.values())
            candidates = [v for v, d in degs.items() if d == max_deg]
            start_v = candidates[0] # Deterministic for fallback
            
            path = [start_v]
            visited = {start_v}
            last_v = start_v
            
            while True:
                next_v = None
                for nb in adj[last_v]:
                    if nb not in visited:
                        e = tuple(sorted((last_v, nb)))
                        if e in current_edges:
                            next_v = nb
                            break
                if next_v is None:
                    break
                path.append(next_v)
                visited.add(next_v)
                current_edges.discard(tuple(sorted((last_v, next_v))))
                last_v = next_v
            
            if len(path) > 1:
                paths.append(path)
            else:
                for e in list(current_edges):
                    paths.append(list(e))
                current_edges = set()
                break
                
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    # Allocate time per graph
    # Total time 130s. 3 graphs. ~40s each.
    # But greedy is fast. Let's give 10s each for search, rest for safety.
    
    for i, spec in enumerate(specs):
        g_edges = instance(*spec)
        # Adjust restarts based on remaining time
        time_left = deadline - time.time()
        if time_left < 5:
            # Emergency: use simple greedy
            paths = greedy_simple(g_edges)
        else:
            # Use greedy with Warnsdorff and multiple restarts
            # Time limit per graph: min(10, time_left/3)
            tl = min(10.0, time_left / 3.0)
            paths = solve_greedy_warnsdorff(g_edges, max_restarts=100, time_limit=tl)
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-END
