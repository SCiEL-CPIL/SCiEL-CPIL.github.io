import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # 5s margin
    
    n = 18
    vertices = list(range(n))
    
    # Helper: Check if a set of edges is K4-free
    # Edges represented as a set of tuples (u, v) with u < v
    def is_k4_free(edges):
        # For each 4-subset, check if all 6 edges are present
        # 3060 subsets, 6 checks each
        # Use a set for fast lookup
        edge_set = edges
        for quad in combinations(vertices, 4):
            a, b, c, d = quad
            if ((a,b) in edge_set and (a,c) in edge_set and (a,d) in edge_set and
                (b,c) in edge_set and (b,d) in edge_set and (c,d) in edge_set):
                return False
        return True

    # Helper: Calculate max triangle-free induced subgraph size
    # Returns the size
    def max_triangle_free_size(edges):
        edge_set = edges
        # Precompute adjacency lists or just use set
        # We iterate k from n down to 1
        for k in range(n, 0, -1):
            found = False
            for subset in combinations(vertices, k):
                is_tf = True
                # Check all triples in subset
                for i in range(len(subset)):
                    u = subset[i]
                    for j in range(i+1, len(subset)):
                        v = subset[j]
                        if (u, v) in edge_set or (v, u) in edge_set:
                            # If edge exists, check for w
                            for w in subset[j+1:]:
                                if ((u, w) in edge_set or (w, u) in edge_set) and \
                                   ((v, w) in edge_set or (w, v) in edge_set):
                                    is_tf = False
                                    break
                        # Note: if (u,v) not in edge_set, no triangle can use u,v,w
                        # Wait, triangle needs 3 edges.
                        # If (u,v) is missing, u-v-w is not a triangle.
                        # So we only need to check w if (u,v) is present.
                        # But we also need to check if (u,w) and (v,w) are present.
                        # My loop above:
                        # If (u,v) is edge:
                        #   Check w. If (u,w) and (v,w) are edges -> Triangle.
                        # This covers all triangles.
                        if not is_tf:
                            break
                    if not is_tf:
                        break
                if is_tf:
                    return k
            if found:
                break
        return 0

    # Initial State: K6,6,6
    def get_k666_edges():
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        edges = set()
        for i in range(3):
            for j in range(i+1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        edges.add((min(u,v), max(u,v)))
        return edges

    current_edges = get_k666_edges()
    current_score = max_triangle_free_size(current_edges)
    best_edges = current_edges.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    # Pre-generate all possible edges
    all_edges = list(combinations(vertices, 2))
    
    while time.time() < deadline:
        # Generate neighbor
        new_edges = current_edges.copy()
        if random.random() < 0.5:
            # Add edge
            e = random.choice(all_edges)
            if e not in new_edges:
                new_edges.add(e)
        else:
            # Remove edge
            if new_edges:
                e = random.choice(list(new_edges))
                new_edges.remove(e)
        
        # Check K4-free
        if not is_k4_free(new_edges):
            continue
            
        new_score = max_triangle_free_size(new_edges)
        
        # Accept if better or with probability
        if new_score < current_score:
            current_edges = new_edges
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges.copy()
        else:
            # Metropolis criterion
            # Delta = new_score - current_score (we want to minimize, so positive is bad)
            delta = new_score - current_score
            if delta == 0:
                # Accept equal to explore
                if random.random() < 0.5:
                    current_edges = new_edges
            else:
                # delta > 0
                prob = 1.0 / (1.0 + delta * T)
                if random.random() < prob:
                    current_edges = new_edges
                    current_score = new_score
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    return [[u, v] for u, v in best_edges]
