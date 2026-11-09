import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    n = P['n']
    # Normalize edges to (min, max) tuples for consistent lookup
    edges = set()
    for u, v in P['edges']:
        if u > v:
            u, v = v, u
        edges.add((u, v))
    
    remaining_edges = set(edges)
    bicliques = []
    
    # Pre-compute full adjacency for potential fallback or initial heuristics if needed, 
    # but we operate on remaining_edges for the greedy steps.
    
    def get_neighbors_in_remaining(vertex, rem_edges):
        """Get neighbors of vertex in the subgraph defined by rem_edges."""
        neighbors = set()
        for u, v in rem_edges:
            if u == vertex:
                neighbors.add(v)
            elif v == vertex:
                neighbors.add(u)
        return neighbors

    def expand_biclique(L, R, rem_edges):
        """
        Expand L and R to form a maximal biclique within rem_edges.
        L and R are sets of vertices.
        """
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Expand L: find vertices connected to ALL in R
            new_L = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                # Check if w is connected to all vertices in R
                is_connected = True
                for r in R:
                    e = (min(w, r), max(w, r))
                    if e not in rem_edges:
                        is_connected = False
                        break
                if is_connected:
                    new_L.add(w)
            if new_L:
                L |= new_L
                changed = True
            
            # Expand R: find vertices connected to ALL in L
            new_R = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                # Check if w is connected to all vertices in L
                is_connected = True
                for l in L:
                    e = (min(w, l), max(w, l))
                    if e not in rem_edges:
                        is_connected = False
                        break
                if is_connected:
                    new_R.add(w)
            if new_R:
                R |= new_R
                changed = True
        
        return L, R

    def find_best_biclique(rem_edges):
        """
        Heuristically find a large biclique in rem_edges.
        Strategy: Try expanding each edge in rem_edges, pick the one with largest area.
        To speed up, limit candidates if rem_edges is large, but n=30 is small.
        """
        if not rem_edges:
            return None, None
        
        best_L = None
        best_R = None
        best_area = 0
        
        # To avoid O(E^2 * N^2) if E is large, we can sample or prioritize high-degree edges.
        # With ~150 edges, trying all is 150 * (expansion cost). Expansion is O(N^2) roughly.
        # 150 * 900 = 135,000 ops. Very fast.
        
        # Sort edges by degree sum to prioritize "central" edges? 
        # Let's just iterate through all edges.
        for u, v in rem_edges:
            # Initial biclique is just the edge (u,v)
            L, R = expand_biclique({u}, {v}, rem_edges)
            area = len(L) * len(R)
            
            if area > best_area:
                best_area = area
                best_L = L
                best_R = R
        
        return best_L, best_R

    while remaining_edges:
        if time.time() > deadline:
            # Fallback: cover remaining edges individually to ensure validity
            for u, v in remaining_edges:
                bicliques.append([list([u]), list([v])])
            break
        
        L, R = find_best_biclique(remaining_edges)
        
        if L is None:
            # Should not happen if remaining_edges is not empty
            break
        
        # Remove edges covered by this biclique
        edges_to_remove = set()
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                if e in remaining_edges:
                    edges_to_remove.add(e)
        
        remaining_edges -= edges_to_remove
        
        # Ensure sides are non-empty and disjoint (expansion logic ensures disjointness by skipping w in L/R)
        # Convert to lists for return format
        bicliques.append([list(L), list(R)])
        
    # Final check and type conversion
    # The problem states: "integers must be Python int... sequences must be list or tuple"
    # Our L and R are sets of ints (from range(n)), so converting to list is fine.
    
    return bicliques
# EVOLVE-BLOCK-END
