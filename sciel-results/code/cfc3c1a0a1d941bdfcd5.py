"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build full adjacency list
    adj_full = defaultdict(list)
    for u, v in edges:
        adj_full[u].append(v)
        adj_full[v].append(u)
        
    # Precompute edge keys for fast lookup
    def ekey(u, v):
        if u < v:
            return (u, v)
        else:
            return (v, u)
            
    deadline = time.time() + 55.0  # Safety margin for 60s limit (or 90s if that's the real one, but 55 is safe for both)
    
    def find_hamiltonian_cycle(adj_cur, deadline):
        """
        Find a Hamiltonian cycle in the graph defined by adj_cur.
        Uses randomized DFS with Warnsdorff-like heuristic.
        Returns list of vertices or None.
        """
        # Check basic feasibility: all degrees >= 2
        for i in range(n):
            if len(adj_cur[i]) < 2:
                return None
        
        # Start node 0 to fix symmetry
        start = 0
        path = [start]
        visited = {start}
        
        # Heuristic score for next move: prefer neighbors with fewer unvisited neighbors (Warnsdorff)
        # But also consider connectivity of remaining graph? Too expensive.
        # Let's just use degree in remaining graph.
        
        def dfs(u):
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                # Check if we can close the cycle
                if start in adj_cur[u]:
                    # Ensure the edge (u, start) is available
                    # adj_cur only contains available edges, so if start is in adj_cur[u], it's available.
                    return list(path)
                return None
            
            neighbors = adj_cur[u]
            # Filter unvisited
            unvisited_neighbors = [v for v in neighbors if v not in visited]
            
            if not unvisited_neighbors:
                return None
            
            # Sort by heuristic: prefer vertices with fewer options
            # Calculate "degree" in the remaining unvisited subgraph + 1 (for u)
            # Actually, standard Warnsdorff: pick neighbor with fewest unvisited neighbors.
            def heuristic(v):
                count = 0
                for w in adj_cur[v]:
                    if w not in visited and w != u:
                        count += 1
                return count
            
            unvisited_neighbors.sort(key=heuristic)
            
            # Add some randomness to break ties or explore other branches
            # We can shuffle the list and then stable sort? Or just randomize order slightly.
            # Let's shuffle first, then sort by (heuristic, random_key) to break ties randomly.
            # Or simpler: if we want to be fast, strict Warnsdorff often works.
            # Let's add a small random factor to the key to explore different paths on retries.
            random.shuffle(unvisited_neighbors)
            unvisited_neighbors.sort(key=heuristic)
            
            for v in unvisited_neighbors:
                path.append(v)
                visited.add(v)
                
                res = dfs(v)
                if res is not None:
                    return res
                
                path.pop()
                visited.remove(v)
                
            return None
        
        return dfs(start)

    best_cycles = []
    
    # We try to find 3 edge-disjoint Hamiltonian cycles.
    # Since the graph is 6-regular on 21 vertices, 3 HC partition the edges.
    # If we find 3, we are done (max possible is 3 for HC on 21 vertices? 
    # Actually, could we find more cycles if they are not Hamiltonian?
    # The task says "All cycles must use exactly the same vertex set".
    # If the vertex set is size 21, they must be Hamiltonian.
    # If the vertex set is smaller, say size 3, we could potentially find more?
    # "Maximize number of cycles".
    # If we use a subset of vertices of size k, max cycles is bounded by min(10, E_sub // k).
    # For k=21, max is 3 (since 63 edges // 21 = 3).
    # For k=14, 70 edges total. Subgraph induced by 14 vertices might have ~14*6/2 * (14/21)^2 ? 
    # Average degree in induced subgraph of size k is approx 6 * (k-1)/(n-1).
    # Edges approx k * 6 * (k-1)/(2*(n-1)).
    # Cycles approx Edges / k = 3 * (k-1)/(n-1).
    # For k=21, 3 * 20/20 = 3.
    # For k=14, 3 * 13/20 = 1.95 -> 1 cycle?
    # Wait, if we pick a specific subset, the density might be higher.
    # But Hamiltonian cycles on the full set seem to yield 3.
    # Can we yield more than 3?
    # If we use a smaller vertex set, say size 7.
    # Max edges in 7-vertex subset: 7*6 = 42 (if all connected to outside, but induced subgraph).
    # Induced subgraph of 7 vertices in 6-reg graph.
    # Max edges ~ 7 * 6 / 2 * (something).
    # If we can find 10 cycles of size 3?
    # 10 cycles * 3 edges = 30 edges.
    # We have 63 edges.
    # If we partition into triangles?
    # 21 vertices / 3 = 7 triangles. 7 cycles.
    # Can we find 10 edge-disjoint triangles on the same 21 vertices?
    # No, "same vertex set" means ALL cycles use the SAME set of vertices.
    # So if I pick vertices {0,1,2}, every cycle must be a cycle on {0,1,2}.
    # The only simple cycle on 3 vertices is the triangle 0-1-2-0.
    # It uses edges (0,1), (1,2), (2,0).
    # I can only have 1 such cycle because edges must be disjoint.
    # So for k=3, max cycles is 1.
    # For k=4, cycles are 4-cycles.
    # Max edges in 4-set is 6. Cycle uses 4.
    # Can I fit 2 disjoint 4-cycles on same 4 vertices?
    # No, 4 edges each, 8 edges needed, only 6 available.
    # So max 1.
    # For k=5, 5-cycle uses 5 edges. Max edges 10.
    # Can I fit 2? 10 edges needed. Yes, if K5.
    # But our graph is sparse.
    # Generally, for small k, max cycles is small.
    # For k=21, we found 3.
    # Is it possible to get > 3?
    # Only if we use a smaller vertex set with higher edge density relative to size?
    # Or if the "vertex set" constraint allows something else?
    # "All cycles must use exactly the same vertex set of size at least three".
    # If I choose a set S, all cycles are in G[S].
    # Max cycles = max number of edge-disjoint cycles in G[S].
    # For S=V, it's 3 (if 3-edge-colorable).
    # If I pick S such that G[S] is dense?
    # The host graph is 6-regular.
    # Maybe a subset of 10 vertices is very dense?
    # But 3 is a decent number.
    # Let's stick to finding 3 Hamiltonian cycles on the full set.
    # If we can't find 3, we return fewer.
    
    for attempt in range(500):
        if time.time() > deadline - 2.0:
            break
            
        current_used = set()
        current_cycles = []
        
        success = True
        for k in range(3):
            # Build adjacency for remaining edges
            adj_cur = defaultdict(list)
            for u, v in edges:
                e = ekey(u, v)
                if e not in current_used:
                    adj_cur[u].append(v)
                    adj_cur[v].append(u)
                    
            # Check feasibility
            possible = True
            for i in range(n):
                if len(adj_cur[i]) < 2:
                    possible = False
                    break
                    
            if not possible:
                success = False
                break
                
            hc = find_hamiltonian_cycle(adj_cur, deadline)
            
            if hc is None:
                success = False
                break
                
            current_cycles.append(hc)
            
            # Mark edges as used
            for i in range(len(hc)):
                u = hc[i]
                v = hc[(i + 1) % len(hc)]
                current_used.add(ekey(u, v))
                
        if success and len(current_cycles) > len(best_cycles):
            best_cycles = current_cycles
            
        if len(best_cycles) == 3:
            break
            
    # If we found 3, return them.
    # If we found less, return what we have.
    # If we found 0, we need a fallback.
    # The task requires a nonempty list.
    # So we must find at least 1.
    
    if not best_cycles:
        # Fallback: find any single Hamiltonian cycle
        adj_cur = defaultdict(list)
        for u, v in edges:
            adj_cur[u].append(v)
            adj_cur[v].append(u)
        hc = find_hamiltonian_cycle(adj_cur, deadline)
        if hc:
            best_cycles = [hc]
        else:
            # Absolute fallback: return a trivial valid cycle if possible?
            # Or just return an empty list? No, "nonempty list".
            # If we can't find a HC, maybe find a smaller cycle?
            # But "same vertex set".
            # If we return a cycle on a subset, that's valid as long as all cycles use that subset.
            # If we return 1 cycle on subset S, it's valid.
            # Let's try to find any cycle.
            # Find a triangle.
            for u in range(n):
                for v in adj_full[u]:
                    if u < v:
                        for w in adj_full[v]:
                            if w > v and w in adj_full[u]:
                                # Triangle u-v-w
                                # Check if edges exist (they do by construction)
                                best_cycles = [[u, v, w]]
                                break
                        if best_cycles:
                            break
                if best_cycles:
                    break
                    
    # Final validation and type conversion
    result = []
    for c in best_cycles:
        # Ensure integers
        result.append([int(v) for v in c])
        
    return result

# EVOLVE-BLOCK-END
