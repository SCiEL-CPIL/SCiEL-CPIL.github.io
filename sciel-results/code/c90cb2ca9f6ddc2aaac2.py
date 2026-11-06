import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 18
    
    # Bitmask representation
    # adj[i] is an integer where bit j is 1 if (i, j) is an edge
    
    def make_edge_set_to_bitmask(edge_set):
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def bitmask_to_edge_list(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    def is_k4_free(adj):
        # Check if adding any edge would create K4? No, check current graph.
        # For each edge (u, v), if there is an edge (w, x) in N(u) ∩ N(v), then K4 exists.
        # N(u) ∩ N(v) is adj[u] & adj[v].
        # We need to check if the subgraph induced by adj[u] & adj[v] has any edge.
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Check if common neighborhood has any edge
                    # Iterate through vertices in common
                    c = common
                    while c:
                        # Get lowest set bit
                        w = (c & -c).bit_length() - 1
                        c &= c - 1
                        # Check if w has any neighbor in common
                        # (adj[w] & common) != 0
                        if adj[w] & common:
                            return False
        return True

    def max_triangle_free_size(adj, timeout=0.05):
        start = time.time()
        best = 0
        
        # Order vertices by degree? Or just 0..n-1
        # We use a recursive B&B
        # remaining: bitmask of vertices not yet decided
        # current: bitmask of vertices in the current triangle-free set
        
        def bt(remaining, current):
            nonlocal best
            if time.time() - start > timeout:
                return
            
            # Pruning: if current size + remaining count <= best, return
            if current.bit_count() + remaining.bit_count() <= best:
                return
            
            if not remaining:
                if current.bit_count() > best:
                    best = current.bit_count()
                return
            
            # Pick a vertex from remaining
            # Heuristic: pick vertex with highest degree in remaining?
            # Or just lowest index.
            # Let's pick lowest index for simplicity, but maybe heuristic helps.
            v = (remaining & -remaining).bit_length() - 1
            rest = remaining & ~(1 << v)
            
            # Option 1: Exclude v
            bt(rest, current)
            
            # Option 2: Include v
            # Check if v can be added to current
            # v is compatible with all u in current if:
            # For all u in current, (adj[u] & adj[v]) & current == 0 ?
            # No, triangle-free means no edge between any pair in {current} U {v}.
            # Edges within current are already checked (current is triangle-free).
            # We need to check edges between v and current, and triangles formed by v and two vertices in current.
            # Actually, if current is triangle-free, adding v is valid if:
            # 1. v is not connected to any u in current? No, v can be connected to u.
            #    But if v is connected to u, and u is connected to w in current, and v is connected to w, then triangle.
            #    So for all u in current such that (adj[v] >> u) & 1:
            #    (adj[u] & adj[v] & current) must be 0.
            #    Basically, N(v) ∩ N(u) ∩ current must be empty for all u in current.
            
            can_add = True
            # Check compatibility
            # Let C = current
            # For each u in C:
            # If edge (v, u) exists, then N(v) ∩ N(u) ∩ C must be empty.
            # If edge (v, u) does not exist, no constraint from this pair?
            # Wait, triangle is 3 vertices.
            # If we add v to C, new triangles can only involve v.
            # A triangle involving v and two vertices u, w in C exists if:
            # (v,u) is edge, (v,w) is edge, and (u,w) is edge.
            # So we need to check if there exists u, w in C such that
            # (adj[v] & (1<<u)) and (adj[v] & (1<<w)) and (adj[u] & (1<<w)).
            
            # Efficient check:
            # Let Nv_in_C = adj[v] & current
            # For each u in Nv_in_C:
            #   If (adj[u] & Nv_in_C) != 0, then there is a triangle.
            
            Nv_in_C = adj[v] & current
            c = Nv_in_C
            while c:
                u = (c & -c).bit_length() - 1
                c &= c - 1
                if adj[u] & Nv_in_C:
                    can_add = False
                    break
            
            if can_add:
                bt(rest, current | (1 << v))
        
        bt((1 << n) - 1, 0)
        return best

    # Initial Graph: K_{6,6,6}
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    init_edges = set()
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    init_edges.add((min(a, b), max(a, b)))
    
    current_adj = make_edge_set_to_bitmask(init_edges)
    current_score = max_triangle_free_size(current_adj)
    best_adj = current_adj[:]
    best_score = current_score
    
    all_edges = set((i, j) for i in range(n) for j in range(i + 1, n))
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.1
    cooling = 0.99
    
    random.seed(42)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random edge to flip
        u, v = random.choice(list(all_edges))
        
        new_adj = current_adj[:]
        
        if (new_adj[u] >> v) & 1:
            # Remove edge
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            # Always K4 free
            is_valid = True
        else:
            # Add edge
            # Check K4 constraint
            common = new_adj[u] & new_adj[v]
            valid = True
            c = common
            while c:
                w = (c & -c).bit_length() - 1
                c &= c - 1
                if new_adj[w] & common:
                    valid = False
                    break
            
            if not valid:
                T *= cooling
                continue
            
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            is_valid = True
        
        if is_valid:
            new_score = max_triangle_free_size(new_adj)
            
            if new_score < current_score:
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
            else:
                # Accept with probability exp(-delta / T)
                delta = new_score - current_score
                if delta == 0:
                    # Accept to explore
                    if random.random() < 0.5:
                        current_adj = new_adj
                        current_score = new_score
                else:
                    prob = pow(2.71828, -delta / T)
                    if random.random() < prob:
                        current_adj = new_adj
                        current_score = new_score
        
        T *= cooling
    
    return bitmask_to_edge_list(best_adj)
