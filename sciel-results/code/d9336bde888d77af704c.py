"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Start with Paley graph on 17 vertices
    def make_paley(n):
        if n == 17:
            squares = {x * x % 17 for x in range(1, 17)}
            edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
            return edges
        return None
    
    edges = make_paley(17)
    n = 17
    
    # Helper: check if adding vertex v with given neighbor set creates K4 or independent 6-set
    def can_add_vertex(n, edges, new_vertex, neighbors):
        """Check if adding new_vertex with given neighbors to graph on n vertices is valid."""
        # Build adjacency for existing vertices
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check no K4: for any pair (a,b) both in neighbors, if a~b and both adjacent to new_vertex,
        # and there's a common neighbor c of a and b (not new_vertex), then {a,b,c,new} forms K4
        # Actually simpler: for any triangle (a,b,c) in the subgraph induced by neighbors,
        # if new_vertex connects to all three, that's a K4.
        # Also: if a~b and both in neighbors, and a has a common neighbor c with b, 
        # and c is also in neighbors, then {a,b,c,new} is K4.
        
        # More efficient: check all triples in neighbors
        neighbor_list = list(neighbors)
        for i in range(len(neighbor_list)):
            a = neighbor_list[i]
            for j in range(i + 1, len(neighbor_list)):
                b = neighbor_list[j]
                if b not in adj[a]:
                    continue  # a and b not adjacent, can't form K4 with new
                # a~b and both connect to new. Check if there's c in neighbors adjacent to both a and b
                for c in neighbor_list:
                    if c != a and c != b and c in adj[a] and c in adj[b]:
                        return False  # K4 formed
        
        # Check no independent 6-set: the non-neighbors of new_vertex (including new_vertex)
        # Non-neighbors are vertices 0..n-1 not in neighbors, plus new_vertex itself
        non_neighbors = [v for v in range(n) if v not in neighbors]
        # We need to check if there's an independent set of size 5 in non_neighbors
        # (which with new_vertex would make 6)
        # Or an independent set of size 6 entirely within non_neighbors (but new isn't in that set)
        # Actually: independent set of 6 could be:
        # 1. 5 from non_neighbors + new_vertex
        # 2. 6 from non_neighbors (new not involved)
        
        # Case 2: check if non_neighbors has independent set of size 6
        # This is expensive; use a quick check
        # For case 1: check if non_neighbors has independent set of size 5
        
        # Quick heuristic: if |non_neighbors| < 5, case 1 impossible
        if len(non_neighbors) >= 5:
            # Check for independent set of size 5 in non_neighbors
            # Use a simple greedy/backtracking check
            if _has_independent_set(non_neighbors, adj, 5):
                return False
        
        # Case 2: independent set of size 6 in non_neighbors
        if len(non_neighbors) >= 6:
            if _has_independent_set(non_neighbors, adj, 6):
                return False
        
        return True
    
    def _has_independent_set(vertices, adj, size):
        """Check if there's an independent set of given size among vertices."""
        if len(vertices) < size:
            return False
        if size <= 1:
            return True
        
        # Use a simple recursive check with pruning
        # This is exponential but we limit the search
        vertices = list(vertices)
        
        def backtrack(remaining, current_size, current_set):
            if current_size == size:
                return True
            if len(remaining) < size - current_size:
                return False
            if not remaining:
                return False
            
            # Try including remaining[0]
            v = remaining[0]
            rest = remaining[1:]
            
            # Include v: remove all neighbors of v from consideration
            new_remaining = [u for u in rest if u not in adj[v]]
            if backtrack(new_remaining, current_size + 1, current_set | {v}):
                return True
            
            # Exclude v
            if backtrack(rest, current_size, current_set):
                return True
            
            return False
        
        return backtrack(vertices, 0, set())
    
    # Try to add vertices one by one
    max_n = 40
    while n < max_n:
        if time.time() > deadline:
            break
        
        new_vertex = n
        found = False
        
        # Try different neighbor sets for the new vertex
        # Start with greedy: try to connect to many vertices while maintaining validity
        # Use a randomized local search approach
        
        best_neighbors = None
        best_size = -1
        
        # Try several random neighbor sets
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random subset of existing vertices
            import random
            random.seed(n * 1000 + attempt)  # deterministic for reproducibility
            neighbors = set(random.sample(range(n), random.randint(n // 2, n - 1)))
            
            if can_add_vertex(n, edges, new_vertex, neighbors):
                if len(neighbors) > best_size:
                    best_size = len(neighbors)
                    best_neighbors = neighbors.copy()
        
        if best_neighbors is not None:
            # Add the vertex
            for v in best_neighbors:
                edges.append([new_vertex, v])
            n += 1
            found = True
        
        if not found:
            # Try with fewer connections
            success = False
            for attempt in range(30):
                if time.time() > deadline:
                    break
                import random
                random.seed(n * 2000 + attempt)
                neighbors = set(random.sample(range(n), random.randint(0, n // 2)))
                
                if can_add_vertex(n, edges, new_vertex, neighbors):
                    for v in neighbors:
                        edges.append([new_vertex, v])
                    n += 1
                    success = True
                    break
            
            if not success:
                break
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
