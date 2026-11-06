import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 18
    # Initialize with K6,6,6
    # Parts: 0-5, 6-11, 12-17
    edges = set()
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.add((u, v))
    
    def is_k4_free(edge_set):
        # Check for K4
        # For each pair of edges sharing a vertex?
        # Simpler: iterate over all 4-subsets? 18C4 = 3060.
        # Check if all 6 edges exist.
        vertices = list(range(n))
        for combo in itertools.combinations(vertices, 4):
            a, b, c, d = combo
            if ((a,b) in edge_set or (b,a) in edge_set) and \
               ((a,c) in edge_set or (c,a) in edge_set) and \
               ((a,d) in edge_set or (d,a) in edge_set) and \
               ((b,c) in edge_set or (c,b) in edge_set) and \
               ((b,d) in edge_set or (d,b) in edge_set) and \
               ((c,d) in edge_set or (d,c) in edge_set):
                return False
        return True

    # Optimize is_k4_free using bitsets or adjacency matrix
    # Let's use adjacency matrix for speed
    def build_adj(edge_set):
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
        return adj

    def check_k4_adj(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    # Common neighbors of i and j
                    # If any common neighbor k has an edge to another common neighbor l?
                    # K4 is i,j,k,l.
                    # i-j edge.
                    # Need k,l such that i-k, j-k, i-l, j-l, k-l all exist.
                    common = [k for k in range(n) if adj[i][k] and adj[j][k]]
                    for x in range(len(common)):
                        for y in range(x+1, len(common)):
                            k, l = common[x], common[y]
                            if adj[k][l]:
                                return False
        return True

    def max_triangle_free_size(adj):
        # Branch and bound to find largest triangle-free induced subgraph
        # Triangle-free means no 3-clique in the induced subgraph.
        # Equivalently, the complement of the induced subgraph is K3-free? No.
        # We want S such that G[S] has no triangles.
        
        # Heuristic: Greedy set packing?
        # Or B&B on vertices.
        
        best_size = 0
        
        def dfs(current_set, remaining):
            nonlocal best_size
            if not remaining:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                return
            
            # Pruning: if len(current_set) + len(remaining) <= best_size: return
            
            # Pick a vertex v from remaining
            v = remaining[0]
            rest = remaining[1:]
            
            # Option 1: Exclude v
            dfs(current_set, rest)
            
            # Option 2: Include v
            # Check if adding v creates a triangle in current_set + {v}
            # Triangle exists if there are u, w in current_set such that u-v, w-v, u-w exist.
            # i.e., u, w are adjacent in G and both adjacent to v.
            # So v cannot be added if G[current_set] contains an edge (u,w) where u,v and w,v are edges.
            # Actually, if u,w in current_set are adjacent, and both connected to v, then u,w,v is a triangle.
            
            can_add = True
            neighbors_in_set = [u for u in current_set if adj[v][u]]
            for i in range(len(neighbors_in_set)):
                for j in range(i+1, len(neighbors_in_set)):
                    u, w = neighbors_in_set[i], neighbors_in_set[j]
                    if adj[u][w]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                dfs(current_set + [v], rest)

        dfs([], list(range(n)))
        return best_size

    # Local Search
    current_edges = edges.copy()
    current_adj = build_adj(current_edges)
    current_val = max_triangle_free_size(current_adj)
    best_edges = current_edges.copy()
    best_val = current_val
    
    # Randomly add some internal edges to break symmetry?
    # K6,6,6 has value 12.
    
    T = 1.0
    t_start = time.time()
    
    while time.time() < deadline:
        # Pick a random edge to flip
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        if (u, v) in current_edges:
            current_edges.remove((u, v))
            current_adj[u][v] = False
            current_adj[v][u] = False
        else:
            current_edges.add((u, v))
            current_adj[u][v] = True
            current_adj[v][u] = True
            
        if not check_k4_adj(current_adj):
            # Revert
            if (u, v) in current_edges:
                current_edges.remove((u, v))
                current_adj[u][v] = False
                current_adj[v][u] = False
            else:
                current_edges.add((u, v))
                current_adj[u][v] = True
                current_adj[v][u] = True
            continue
            
        new_val = max_triangle_free_size(current_adj)
        
        if new_val < best_val:
            best_val = new_val
            best_edges = current_edges.copy()
            
        # Accept/Reject
        if new_val <= current_val:
            current_val = new_val
        else:
            # Reject (or accept with prob)
            # Revert
            if (u, v) in current_edges:
                current_edges.remove((u, v))
                current_adj[u][v] = False
                current_adj[v][u] = False
            else:
                current_edges.add((u, v))
                current_adj[u][v] = True
                current_adj[v][u] = True
            current_val = max_triangle_free_size(current_adj) # Recalculate? Or keep track.
            # For simplicity, revert and keep current_val as is (approx)
            
    # Return best found
    return [list(e) for e in best_edges]

# Run construct to check
# print(construct())
