import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = set(tuple(sorted(e)) for e in P['edges'])
    
    # Adjacency list for current remaining edges
    # We will maintain a set of remaining edges and derive neighbors on the fly or update adj lists.
    # Given N=30, deriving neighbors is O(E) which is small.
    
    def get_neighbors(u, rem_edges):
        neighbors = set()
        for e in rem_edges:
            if e[0] == u:
                neighbors.add(e[1])
            elif e[1] == u:
                neighbors.add(e[0])
        return neighbors

    bicliques = []
    rem_edges = set(edges)
    
    while rem_edges:
        best_L = None
        best_R = None
        best_count = -1
        
        # Try each vertex as a seed for a maximal biclique
        for u in range(n):
            N_u = get_neighbors(u, rem_edges)
            if not N_u:
                continue
            
            L = {u}
            R = set(N_u)
            
            # Expand L and R to be a maximal biclique
            # A vertex v can be in L if it is connected to all in R
            # A vertex w can be in R if it is connected to all in L
            changed = True
            while changed:
                changed = False
                # Expand L
                new_L = set(L)
                for v in range(n):
                    if v in R or v in L:
                        continue
                    # Check if v is connected to all in R
                    is_connected = True
                    for w in R:
                        if (min(v,w), max(v,w)) not in rem_edges:
                            is_connected = False
                            break
                    if is_connected:
                        new_L.add(v)
                if new_L != L:
                    L = new_L
                    changed = True
                
                # Expand R
                new_R = set(R)
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # Check if w is connected to all in L
                    is_connected = True
                    for v in L:
                        if (min(v,w), max(v,w)) not in rem_edges:
                            is_connected = False
                            break
                    if is_connected:
                        new_R.add(w)
                if new_R != R:
                    R = new_R
                    changed = True
            
            # Count edges in L x R present in rem_edges
            count = 0
            for v in L:
                for w in R:
                    if v != w: # Biclique sides are disjoint? 
                        # The task says "nonempty disjoint sides". 
                        # My expansion logic ensures L and R are disjoint because 
                        # we skip v if v in R and w if w in L.
                        # Also, a vertex cannot be connected to itself in a simple graph.
                        e = (min(v,w), max(v,w))
                        if e in rem_edges:
                            count += 1
            
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R
        
        if best_count == -1:
            # Should not happen if rem_edges is not empty
            break
        
        # Remove edges covered by this biclique
        for v in best_L:
            for w in best_R:
                if v != w:
                    e = (min(v,w), max(v,w))
                    if e in rem_edges:
                        rem_edges.remove(e)
        
        # Convert to lists of ints
        bicliques.append([list(best_L), list(best_R)])
        
    return bicliques
# EVOLVE-BLOCK-END
