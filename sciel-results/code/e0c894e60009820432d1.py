import time
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initialize remaining edges as a set of frozensets or tuples for hashability
    # Using tuple (u, v) with u < v for consistency
    remaining_edges = set()
    for u, v in P['edges']:
        if u > v:
            u, v = v, u
        remaining_edges.add((u, v))
        
    bicliques = []
    
    # Helper to check if time is up
    def time_up():
        return time.time() > deadline

    while remaining_edges:
        if time_up():
            break
            
        # Get active vertices
        active_vertices = set()
        for u, v in remaining_edges:
            active_vertices.add(u)
            active_vertices.add(v)
            
        active_vertices_list = list(active_vertices)
        
        best_score = 0
        best_left = None
        best_right = None
        
        # Try subsets of size 1, 2, 3, 4
        # Size 1
        for u in active_vertices_list:
            # Neighbors of u in remaining graph
            neighbors = set()
            for v in active_vertices_list:
                if u == v: continue
                if (u, v) in remaining_edges or (v, u) in remaining_edges:
                    neighbors.add(v)
            if not neighbors:
                continue
            # Biclique [ {u}, neighbors ]
            # Check if all edges exist (they do by construction of neighbors)
            score = len(neighbors)
            if score > best_score:
                best_score = score
                best_left = [u]
                best_right = sorted(neighbors)
                
        # Size 2
        for u, v in itertools.combinations(active_vertices_list, 2):
            # Common neighbors of u and v
            # Must be in remaining_edges for both
            common = set()
            for w in active_vertices_list:
                if w == u or w == v: continue
                # Check (u,w) and (v,w)
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                if e1 in remaining_edges and e2 in remaining_edges:
                    common.add(w)
            if not common:
                continue
            score = 2 * len(common)
            if score > best_score:
                best_score = score
                best_left = [u, v]
                best_right = sorted(common)
                
        # Size 3
        for u, v, w in itertools.combinations(active_vertices_list, 3):
            common = set()
            for x in active_vertices_list:
                if x in (u, v, w): continue
                e1 = (u, x) if u < x else (x, u)
                e2 = (v, x) if v < x else (x, v)
                e3 = (w, x) if w < x else (x, w)
                if e1 in remaining_edges and e2 in remaining_edges and e3 in remaining_edges:
                    common.add(x)
            if not common:
                continue
            score = 3 * len(common)
            if score > best_score:
                best_score = score
                best_left = [u, v, w]
                best_right = sorted(common)
                
        # Size 4
        for u, v, w, x in itertools.combinations(active_vertices_list, 4):
            common = set()
            for y in active_vertices_list:
                if y in (u, v, w, x): continue
                e1 = (u, y) if u < y else (y, u)
                e2 = (v, y) if v < y else (y, v)
                e3 = (w, y) if w < y else (y, w)
                e4 = (x, y) if x < y else (y, x)
                if e1 in remaining_edges and e2 in remaining_edges and e3 in remaining_edges and e4 in remaining_edges:
                    common.add(y)
            if not common:
                continue
            score = 4 * len(common)
            if score > best_score:
                best_score = score
                best_left = [u, v, w, x]
                best_right = sorted(common)
                
        if best_left is None:
            # Fallback: pick any single edge
            # This should not happen if remaining_edges is not empty
            edge = next(iter(remaining_edges))
            best_left = [edge[0]]
            best_right = [edge[1]]
            best_score = 1
            
        # Remove edges covered by best_biclique
        for u in best_left:
            for v in best_right:
                e = (u, v) if u < v else (v, u)
                if e in remaining_edges:
                    remaining_edges.remove(e)
                    
        bicliques.append([best_left, best_right])
        
    return bicliques
