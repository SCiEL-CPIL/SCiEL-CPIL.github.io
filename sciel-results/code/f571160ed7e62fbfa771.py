<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    all_vertices = set(range(n))
    
    # We'll use backtracking to find Hamiltonian cycles
    # Key insight: fix vertex 0 as start to avoid rotations
    # Use symmetry breaking: the neighbor of 0 in the cycle must be less than the other neighbor
    
    def find_hamiltonian_cycle(available_edges, start=0):
        """Find a Hamiltonian cycle using only available_edges."""
        # available_edges is a set of frozenset({u,v}) or similar
        # Build available adjacency
        avail_adj = defaultdict(set)
        for e in available_edges:
            u, v = e
            avail_adj[u].add(v)
            avail_adj[v].add(u)
        
        # Check: every vertex must have degree >= 2 in available graph
        for v in all_vertices:
            if len(avail_adj[v]) < 2:
                return None
        
        # Backtracking DFS
        # Path starts at 0, ends must return to 0
        # To break symmetry, fix the first step: neighbor of 0 must be < last neighbor
        # Actually, let's fix: the two neighbors of 0 in the cycle, let them be a < b
        # Then the cycle is 0 -> ... -> a and 0 -> ... -> b (two paths from 0)
        
        # Simpler: just do DFS from 0, track visited, and check at end if last connects to 0
        # But this is exponential. Let's use a smarter approach.
        
        # Degree-2 constraint propagation
        # If a vertex has degree exactly 2, both edges must be in the cycle
        
        def solve_with_propagation():
            # Work with a copy
            remaining = set(available_edges)
            forced = set()
            
            # Build initial degree counts
            deg = {v: 0 for v in all_vertices}
            incident = defaultdict(set)
            for e in remaining:
                u, v = e
                deg[u] += 1
                deg[v] += 1
                incident[u].add(e)
                incident[v].add(e)
            
            changed = True
            while changed:
                changed = False
                for v in all_vertices:
                    if deg[v] == 2:
                        # Both edges incident to v must be in cycle
                        for e in list(incident[v]):
                            if e not in forced:
                                forced.add(e)
                                changed = True
                                # Remove the other edge incident to each endpoint
                                u, w = e
                                for other_e in list(incident[u]):
                                    if other_e != e and other_e in remaining and other_e not in forced:
                                        remaining.discard(other_e)
                                        incident[u].discard(other_e)
                                        incident[w].discard(other_e) if w != v else None
                                        # Update degrees
                                        for vertex in (u, w):
                                            if other_e in incident[vertex]:
                                                pass
                                        # Need to be more careful
                                        break
                                break
                    elif deg[v] < 2:
                        return None
                        # Dead end
            
            # This propagation is getting complex. Let me use a simpler DFS with pruning.
            pass
        
        # Let's use a straightforward backtracking with good heuristics
        # Fix start at 0. The cycle is a permutation of all vertices starting and ending at 0.
        # To reduce symmetry: the second vertex must be less than the second-to-last vertex.
        
        best = None
        
        # Order neighbors of 0
        neighbors_of_0 = sorted(avail_adj[0])
        if len(neighbors_of_0) < 2:
            return None
        
        # We'll try different first neighbors
        # For each choice of first neighbor, do DFS
        # Use time limit
        
        import time
        start_time = time.time()
        TIME_LIMIT = 50  # seconds per cycle search
        
        def dfs(path, visited, remaining_edges_local):
            nonlocal best
            if time.time() - start_time > TIME_LIMIT:
                return None
            
            if len(path) == n:
                # Check if last connects to 0
                last = path[-1]
                if 0 in avail_adj[last] and (min(last,0), max(last,0)) in remaining_edges_local:
                    return path[:]
                return None
            
            curr = path[-1]
            # Try neighbors of curr that are unvisited
            # Heuristic: prefer vertices with fewer unvisited neighbors (dead-end first)
            candidates = []
            for nb in avail_adj[curr]:
                if nb not in visited:
                    # Check edge is still available
                    e = (min(curr, nb), max(curr, nb))
                    if e in remaining_edges_local:
                        # Count unvisited neighbors of nb
                        unvisited_count = sum(1 for x in avail_adj[nb] if x not in visited)
                        candidates.append((unvisited_count, nb))
            
            candidates.sort()
            
            for _, nb in candidates:
                e = (min(curr, nb), max(curr, nb))
                if e not in remaining_edges_local:
                    continue
                # Remove edge
                remaining_edges_local.discard(e)
                visited.add(nb)
                path.append(nb)
                
                result = dfs(path, visited, remaining_edges_local)
                if result is not None:
                    return result
                
                # Backtrack
                path.pop()
                visited.remove(nb)
                remaining_edges_local.add(e)
            
            return None
        
        # Try different first neighbors for vertex 0
        for first_nb in neighbors_of_0:
            e0 = (0, first_nb)
            if e0 not in available_edges:
                continue
            
            # Start DFS
            path = [0, first_nb]
            visited = {0, first_nb}
            rem = set(available_edges)
            rem.discard(e0)
            
            # Also need to ensure the last vertex connects to 0
            # The last vertex must be a neighbor of 0 (other than first_nb potentially)
            
            result = dfs(path, visited, rem)
            if result is not None:
                return result
        
        return None
    
    # Find cycles iteratively
    cycles = []
    remaining = set((min(u,v), max(u,v)) for u,v in edges)
    
    for i in range(3):  # Max 3 cycles (72 edges / 21 per cycle)
        cycle = find_hamiltonian_cycle(remaining)
        if cycle is None:
            break
        # Convert to list of vertices
        cycle_list = list(cycle)
        # Remove the implicit closing: cycle is [0, v1, v2, ..., v20] and closing edge is v20-0
        # The representation should be distinct vertices with closing edge implicit
        # So [0, v1, v2, ..., v20] is correct
        cycles.append(cycle_list)
        # Remove edges used in this cycle
        for j in range(len(cycle_list)):
            u = cycle_list[j]
            v = cycle_list[(j+1) % len(cycle_list)]
            e = (min(u,v), max(u,v))
            remaining.discard(e)
    
    if not cycles:
        # Fallback: return the trivial cycle if possible
        # Check if 0-1-2-...-20-0 is valid
        trivial = list(range(n))
        valid = True
        for j in range(n):
            u = trivial[j]
            v = trivial[(j+1) % n]
            e = (min(u,v), max(u,v))
            if e not in set((min(a,b), max(a,b)) for a,b in edges):
                valid = False
                break
        if valid:
            return [trivial]
        return [list(range(n))]  # This might be invalid but better than empty
    
    # Ensure types are correct
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    return result

# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    all_vertices = list(range(n))
    
    def find_hamiltonian_cycle(available_edges):
        """Find a Hamiltonian cycle using only available_edges via backtracking with strong pruning."""
        # Build available adjacency
        avail_adj = defaultdict(set)
        for e in available_edges:
            u, v = e
            avail_adj[u].add(v)
            avail_adj[v].add(u)
        
        # Check: every vertex must have degree >= 2 in available graph
        for v in all_vertices:
            if len(avail_adj[v]) < 2:
                return None
        
        # Use DFS with pruning
        start = 0
        neighbors_of_start = sorted(avail_adj[start])
        if len(neighbors_of_start) < 2:
            return None
        
        # Symmetry breaking: first neighbor must be < last neighbor
        # We'll try all first neighbors but enforce that the second-to-last > first
        
        def dfs(path, visited, avail_adj_local, edge_count_local):
            if len(path) == n:
                last = path[-1]
                if start in avail_adj_local[last]:
                    return path[:]
                return None
            
            curr = path[-1]
            # Get unvisited neighbors
            candidates = []
            for nb in avail_adj_local[curr]:
                if nb not in visited:
                    # If we're at position n-2 (second to last), check if nb can connect to start
                    if len(path) == n - 2:
                        if start in avail_adj_local[nb]:
                            candidates.append(nb)
                    else:
                        # Prune: nb must have at least one unvisited neighbor (or be connectable to start if it's the last)
                        unvisited_neighbors = [x for x in avail_adj_local[nb] if x not in visited]
                        if len(path) == n - 3:
                            # nb will be second-to-last or last
                            if start in avail_adj_local[nb] or len(unvisited_neighbors) >= 1:
                                candidates.append(nb)
                        else:
                            if len(unvisited_neighbors) >= 1:
                                candidates.append(nb)
            
            # Sort by degree heuristic (fewest options first)
            candidates.sort(key=lambda x: len([y for y in avail_adj_local[x] if y not in visited]))
            
            for nb in candidates:
                # Remove edges from curr to nb and nb to others to maintain consistency
                # Actually, we just track visited and use avail_adj_local
                new_avail = {v: set(neighbors) for v, neighbors in avail_adj_local.items()}
                new_avail[curr].discard(nb)
                new_avail[nb].discard(curr)
                
                # Check all remaining vertices still have degree >= 2
                valid = True
                for v in all_vertices:
                    if v in visited:
                        continue
                    if len(new_avail[v]) < 2:
                        valid = False
                        break
                if not valid:
                    continue
                
                visited.add(nb)
                path.append(nb)
                
                result = dfs(path, visited, new_avail, edge_count_local)
                if result is not None:
                    return result
                
                path.pop()
                visited.remove(nb)
            
            return None
        
        # Try different first neighbors
        for first_nb in neighbors_of_start:
            # Start path: [0, first_nb]
            path = [start, first_nb]
            visited = {start, first_nb}
            
            # Remove the edge between start and first_nb from consideration
            # We need to track which edges are used
            # For simplicity, we'll use the avail_adj and just ensure we don't reuse
            # Actually, let's track used edges separately
            
            used_edges = set()
            used_edges.add((min(start, first_nb), max(start, first_nb)))
            
            # Build local adjacency excluding used edges
            local_adj = defaultdict(set)
            for v in all_vertices:
                for nb in avail_adj[v]:
                    e = (min(v, nb), max(v, nb))
                    if e not in used_edges:
                        local_adj[v].add(nb)
            
            # Also remove start from local_adj[first_nb] since we came from start
            # And remove first_nb from local_adj[start]
            local_adj[start].discard(first_nb)
            local_adj[first_nb].discard(start)
            
            # Check degrees
            valid = True
            for v in all_vertices:
                if v in visited:
                    continue
                if len(local_adj[v]) < 2:
                    valid = False
                    break
            if not valid:
                continue
            
            result = dfs(path, visited, local_adj, used_edges)
            if result is not None:
                return result
        
        return None
    
    # Find cycles iteratively
    cycles = []
    remaining_edges = set(edge_set)
    
    for i in range(3):
        cycle = find_hamiltonian_cycle(remaining_edges)
        if cycle is None:
            break
        
        cycles.append(list(cycle))
        
        # Remove edges used in this cycle
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            remaining_edges.discard(e)
    
    # Ensure types are correct
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    return result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize my new approach still has the same fundamental issue - copying the adjacency dictionary at each DFS step is expensive. Let me think of a better approach.

Since the maximum is 3 cycles and the current program already finds 3, let me try a completely different strategy: use a greedy approach with local search, or better yet, try to find cycles using a more efficient algorithm.

Actually, let me try a different approach: use the fact that we can precompute some structure. Let me try using a different search strategy - instead of full backtracking, use a constraint satisfaction approach or try specific cycle patterns.

Let me try yet another approach: since the graph is fixed, I can try to find 3 Hamiltonian cycles by using a randomized or heuristic approach that's faster.

Actually, the simplest improvement: the current program's DFS copies `remaining_edges_local` which is a set - that's expensive. Let me use a different representation.

Let me try a cleaner implementation that uses a bitset or a different data structure for tracking available edges.

