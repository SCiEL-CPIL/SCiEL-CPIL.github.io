import random
import time
import itertools
import numpy as np

def construct():
    n = 20
    # adj is a binary numpy matrix: adj[u,v] = 1 if edge exists
    adj = np.zeros((n, n), dtype=np.int32)
    # common[u][v] is the number of common neighbors of u and v
    common = np.zeros((n, n), dtype=np.int32)
    
    def can_add(u, v):
        if u == v:
            return False
        if adj[u, v] == 1:
            return False
        # Check 1: common neighbors of u and v must be <= 2
        if common[u, v] > 2:
            return False
        
        # Check 2: For all y in N(v), common[u][y] must be <= 1
        # Vectorized: check if any y in N(v) has common[u,y] > 1
        neighbors_v = np.where(adj[v] == 1)[0]
        if len(neighbors_v) > 0:
            mask = np.ones(n, dtype=bool)
            mask[v] = False  # exclude v itself
            mask[u] = False  # exclude u
            neighbors_v_masked = neighbors_v[mask[neighbors_v]]
            if len(neighbors_v_masked) > 0:
                if np.any(common[u, neighbors_v_masked] > 1):
                    return False
        
        # Check 3: For all y in N(u), common[v][y] must be <= 1
        neighbors_u = np.where(adj[u] == 1)[0]
        if len(neighbors_u) > 0:
            mask = np.ones(n, dtype=bool)
            mask[u] = False  # exclude u itself
            mask[v] = False  # exclude v
            neighbors_u_masked = neighbors_u[mask[neighbors_u]]
            if len(neighbors_u_masked) > 0:
                if np.any(common[v, neighbors_u_masked] > 1):
                    return False
        
        return True

    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        # Update common counts
        # For y in N(v) (excluding u): common[u][y] += 1
        neighbors_v = np.where(adj[v] == 1)[0]
        mask = np.ones(n, dtype=bool)
        mask[u] = False
        neighbors_v_masked = neighbors_v[mask[neighbors_v]]
        if len(neighbors_v_masked) > 0:
            common[u, neighbors_v_masked] += 1
            common[neighbors_v_masked, u] += 1
            
        # For y in N(u) (excluding v): common[v][y] += 1
        neighbors_u = np.where(adj[u] == 1)[0]
        mask = np.ones(n, dtype=bool)
        mask[v] = False
        neighbors_u_masked = neighbors_u[mask[neighbors_u]]
        if len(neighbors_u_masked) > 0:
            common[v, neighbors_u_masked] += 1
            common[neighbors_u_masked, v] += 1

    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0
        # Revert updates
        # For y in N(v) (excluding u): common[u][y] -= 1
        neighbors_v = np.where(adj[v] == 1)[0]
        if len(neighbors_v) > 0:
            common[u, neighbors_v] -= 1
            common[neighbors_v, u] -= 1
        # For y in N(u) (excluding v): common[v][y] -= 1
        neighbors_u = np.where(adj[u] == 1)[0]
        if len(neighbors_u) > 0:
            common[v, neighbors_u] -= 1
            common[neighbors_u, v] -= 1

    def get_edge_count():
        return int(adj.sum()) // 2

    def get_edges_list():
        """Get list of all edges as (u, v) with u < v"""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 1:
                    edges.append((u, v))
        return edges

    def build_clebsch():
        # Clebsch graph on vertices 0-15
        # Vertices are 4-bit strings.
        # Adjacency: Hamming distance 1 or 4.
        for u in range(16):
            for v in range(16):
                if u >= v:
                    continue
                # Calculate Hamming distance
                diff = u ^ v
                popcount = bin(diff).count('1')
                if popcount == 1 or popcount == 4:
                    add_edge(u, v)

    def build_initial_graph():
        # Build Clebsch graph
        build_clebsch()
        
        # Find all edges in Clebsch graph to use as neighbor sets for new vertices
        edges_in_clebsch = get_edges_list()
        
        # Greedily pick 4 disjoint edges
        used_vertices = set()
        selected_pairs = []
        for u, v in edges_in_clebsch:
            if u not in used_vertices and v not in used_vertices:
                selected_pairs.append((u, v))
                used_vertices.add(u)
                used_vertices.add(v)
                if len(selected_pairs) == 4:
                    break
            
        # If we didn't find 4 disjoint edges (unlikely for Clebsch), we can pick overlapping ones
        # but that might violate constraints if not careful. 
        # However, for now, let's assume we found 4.
        # Connect new vertices 16-19 to these pairs
        for i, (u, v) in enumerate(selected_pairs):
            new_v = 16 + i
            add_edge(new_v, u)
            add_edge(new_v, v)
            
        # Connect new vertices among themselves (K4)
        for i in range(4):
            for j in range(i + 1, 4):
                u = 16 + i
                v = 16 + j
                if can_add(u, v):
                    add_edge(u, v)

    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    
    # Build initial structure
    build_initial_graph()
    
    best_edges = [[u, v] for u, v in get_edges_list()]
    best_count = get_edge_count()
    
    # Local search with vectorized checks
    while time.time() - start_time < time_limit:
        # Try to add a random edge
        improved = False
        # Try many random additions
        for _ in range(2000):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if can_add(u, v):
                add_edge(u, v)
                improved = True
                break
        
        if not improved:
            # Stuck. Try to remove an edge to allow new ones
            curr_count = get_edge_count()
            if curr_count > 0:
                # Pick a random edge to remove
                edges_list = get_edges_list()
                if edges_list:
                    u, v = random.choice(edges_list)
                    remove_edge(u, v)
                    # Try to add edges again
                    for _ in range(2000):
                        u2 = random.randint(0, n - 1)
                        v2 = random.randint(0, n - 1)
                        if can_add(u2, v2):
                            add_edge(u2, v2)
                            improved = True
                            break
                    if improved:
                        curr_count = get_edge_count()
                        if curr_count > best_count:
                            best_count = curr_count
                            best_edges = [[u, v] for u, v in get_edges_list()]
            else:
                # Empty graph, restart
                pass

    # Final check: ensure the best_edges is valid
    if not best_edges:
        # Fallback to seed if something went wrong
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    # Verification step (optional but good for safety)
    adj_check = [set() for _ in range(n)]
    for u, v in best_edges:
        adj_check[u].add(v)
        adj_check[v].add(u)
    
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            common_count = len(adj_check[u] & adj_check[v])
            if common_count > 2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return best_edges
