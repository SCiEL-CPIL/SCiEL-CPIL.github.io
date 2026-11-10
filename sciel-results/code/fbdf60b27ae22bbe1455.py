import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for 150s limit

    # 1. Construct the Clebsch Graph (Triangle-free, 16 vertices, 40 edges, chi=4)
    # Vertices are 5-bit strings (0..31).
    # Folded 5-cube: u ~ v if v = u XOR 31.
    # Map 32 nodes to 16 classes: class(x) = min(x, x^31)
    
    def get_class(x):
        return min(x, x ^ 31)
    
    class_to_id = {}
    id_to_class = []
    for i in range(32):
        c = get_class(i)
        if c not in class_to_id:
            class_to_id[c] = len(class_to_id)
            id_to_class.append(c)
            
    node_to_id = {}
    for i in range(32):
        node_to_id[i] = class_to_id[get_class(i)]
        
    edges_set = set()
    for i in range(32):
        u_id = node_to_id[i]
        for k in range(5):
            j = i ^ (1 << k)
            v_id = node_to_id[j]
            if u_id < v_id:
                edges_set.add((u_id, v_id))
                
    initial_edges = [list(e) for e in edges_set]
    
    # Build adjacency bitmasks
    def build_adj(edge_list):
        adj = [0] * 16
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    adj = build_adj(initial_edges)
    best_adj = adj[:]
    best_count = len(initial_edges)
    best_edges = initial_edges

    # 2. Verification Functions
    
    def is_triangle_free(adj):
        for u in range(16):
            for v in range(u + 1, 16):
                if adj[u] & (1 << v):
                    if adj[u] & adj[v]:
                        return False
        return True

    def is_not_3_colorable(adj):
        colors = [-1] * 16
        
        def get_available(v):
            avail = [0, 1, 2]
            mask = adj[v]
            for c in range(3):
                for j in range(16):
                    if mask & (1 << j):
                        if colors[j] == c:
                            if c in avail: avail.remove(c)
                            break
            return avail

        def backtrack(unassigned_mask):
            if unassigned_mask == 0:
                return True
            
            # MRV Heuristic
            best_v = -1
            best_avail = None
            min_count = 4
            
            m = unassigned_mask
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                avail = get_available(v)
                if len(avail) < min_count:
                    min_count = len(avail)
                    best_v = v
                    best_avail = avail
                    if min_count == 0:
                        return False
            
            for c in best_avail:
                colors[best_v] = c
                if backtrack(unassigned_mask & ~(1 << best_v)):
                    return True
                colors[best_v] = -1
            
            return False

        return not backtrack((1 << 16) - 1)

    # Initial check
    if not is_triangle_free(adj) or not is_not_3_colorable(adj):
        # Fallback to Grötzsch (20 edges) if Clebsch fails (should not happen)
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]

    # 3. Simulated Annealing / Local Search
    # Goal: Find graph with > 40 edges if possible.
    # Move: Randomly swap an edge with a non-edge, or add/remove edge.
    # Constraint: Must be triangle-free and not 3-colorable.
    
    all_edges = [(u, v) for u in range(16) for v in range(u + 1, 16)]
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(16)) // 2

    def edges_from_adj(adj):
        e = []
        for u in range(16):
            for v in range(u + 1, 16):
                if adj[u] & (1 << v):
                    e.append([u, v])
        return e

    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    current_adj = best_adj[:]
    current_edges = count_edges(current_adj)
    
    # Precompute non-edges for faster access?
    # We will generate random moves.
    
    while time.time() < deadline - 5:
        # Generate a random move:
        # 1. Pick a random edge to remove
        # 2. Pick a random non-edge to add
        # Or just pick a random pair and toggle?
        # Better: 2-opt swap. Remove (u1, v1), add (u2, v2).
        
        # Get list of current edges
        curr_edge_list = []
        for u in range(16):
            for v in range(u + 1, 16):
                if current_adj[u] & (1 << v):
                    curr_edge_list.append((u, v))
        
        # Get list of non-edges
        curr_non_edge_list = []
        for u in range(16):
            for v in range(u + 1, 16):
                if not (current_adj[u] & (1 << v)):
                    curr_non_edge_list.append((u, v))
        
        if not curr_edge_list or not curr_non_edge_list:
            continue
            
        u1, v1 = random.choice(curr_edge_list)
        u2, v2 = random.choice(curr_non_edge_list)
        
        # Apply move
        current_adj[u1] &= ~(1 << v1)
        current_adj[v1] &= ~(1 << u1)
        current_adj[u2] |= (1 << v2)
        current_adj[v2] |= (1 << u2)
        
        # Check validity
        valid = True
        if not is_triangle_free(current_adj):
            valid = False
        elif not is_not_3_colorable(current_adj):
            valid = False
            
        new_count = count_edges(current_adj)
        
        if valid:
            if new_count > best_count:
                best_count = new_count
                best_adj = current_adj[:]
                best_edges = edges_from_adj(current_adj)
            # Accept move if valid (hill climbing) or with probability exp(-delta/T)
            # Since we want max edges, and valid graphs are rare, we accept all valid moves.
            # This effectively becomes a random walk on the space of valid graphs.
        else:
            # Reject move, revert
            current_adj[u1] |= (1 << v1)
            current_adj[v1] |= (1 << u1)
            current_adj[u2] &= ~(1 << v2)
            current_adj[v2] &= ~(1 << u2)
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_edges
