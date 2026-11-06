import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    n = 16
    target_edges = 65
    
    # Initialize with K_{8,8} + (0,1)
    # Vertices 0-7 in A, 8-15 in B
    adj = [0] * n
    degs = [0] * n
    
    def add_edge(u, v):
        if u == v:
            return False
        if adj[u] & (1 << v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        degs[u] += 1
        degs[v] += 1
        return True
        
    def remove_edge(u, v):
        if u == v:
            return False
        if not (adj[u] & (1 << v)):
            return False
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        degs[u] -= 1
        degs[v] -= 1
        return True

    # Build K_{8,8}
    for u in range(8):
        for v in range(8, 16):
            add_edge(u, v)
            
    # Add one internal edge (0,1)
    add_edge(0, 1)
    
    # Check edge count
    current_edges = sum(degs) // 2
    if current_edges != target_edges:
        # Fallback to random if init failed
        adj = [0] * n
        degs = [0] * n
        edges_list = []
        while len(edges_list) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not (adj[u] & (1 << v)):
                add_edge(u, v)
                edges_list.append((u, v))

    def get_score():
        max_sum = 0
        # Iterate over edges
        for u in range(n):
            if degs[u] == 0: continue
            v_mask = adj[u]
            # Only consider v > u to avoid double counting? 
            # For triangles, we can just iterate all pairs in adj[u]
            # But we need degs.
            # Triangle (u, v, w) exists if v in adj[u], w in adj[v], w in adj[u]
            # And v != w.
            # To be efficient, iterate u < v < w
            pass
            
        # Better way: iterate u, v, w
        # n=16, so 16*16*16 = 4096 checks. Very fast.
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    continue
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                # Remove u and v from common
                common &= ~(1 << u)
                common &= ~(1 << v)
                # Iterate through common neighbors
                w_mask = common
                while w_mask:
                    # Get lowest set bit
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
        return max_sum

    best_score = get_score()
    best_adj = [x for x in adj]
    best_degs = [x for x in degs]
    
    # Local Search
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate a move
        # 50% chance: Edge Swap (preserve degrees)
        # 50% chance: Edge Replacement (change degrees)
        
        if random.random() < 0.5:
            # Edge Swap
            # Find two disjoint edges (u,v) and (w,z)
            # Such that (u,w) and (v,z) are not edges
            # And (u,z) and (v,w) are not edges (to avoid creating duplicate edges or self loops, though u!=w etc handles self loops)
            # Actually, we just need (u,w) and (v,z) to be non-edges.
            # We pick a random edge (u,v)
            u = random.randint(0, n-1)
            if degs[u] == 0: continue
            v_mask = adj[u]
            # Pick random v
            v_list = [i for i in range(n) if adj[u] & (1 << i)]
            if not v_list: continue
            v = random.choice(v_list)
            
            # Pick random w, z
            # w, z must be distinct from u, v
            candidates = [i for i in range(n) if i != u and i != v]
            if len(candidates) < 2: continue
            w, z = random.sample(candidates, 2)
            
            # Check if (w,z) is an edge
            if not (adj[w] & (1 << z)):
                continue # We need (w,z) to be an edge to swap with
            
            # Check if new edges (u,w) and (v,z) are valid (non-edges)
            if (adj[u] & (1 << w)) or (adj[v] & (1 << z)):
                continue
            # Also ensure (u,z) and (v,w) are not edges? 
            # If (u,z) is an edge, and we add (u,w), we are fine.
            # But if we add (u,w) and (v,z), we might create a multigraph? No, simple graph.
            # We just need to ensure we don't add an existing edge.
            # We checked (u,w) and (v,z) are non-edges.
            
            # Perform swap
            remove_edge(u, v)
            remove_edge(w, z)
            add_edge(u, w)
            add_edge(v, z)
            
            new_score = get_score()
            
            # Acceptance
            if new_score < best_score or (T > 0.1 and random.random() < math.exp(-(new_score - best_score) / T)):
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [x for x in adj]
                    best_degs = [x for x in degs]
            else:
                # Revert
                remove_edge(u, w)
                remove_edge(v, z)
                add_edge(u, v)
                add_edge(w, z)
                
        else:
            # Edge Replacement
            # Remove (u,v), Add (w,z)
            u = random.randint(0, n-1)
            if degs[u] == 0: continue
            v_list = [i for i in range(n) if adj[u] & (1 << i)]
            if not v_list: continue
            v = random.choice(v_list)
            
            # Pick non-edge (w,z)
            w = random.randint(0, n-1)
            z = random.randint(0, n-1)
            if w == z: continue
            if adj[w] & (1 << z): continue
            # Ensure we don't remove (u,v) and add (u,v) or (v,u)
            if (w == u and z == v) or (w == v and z == u):
                continue
                
            # Perform move
            remove_edge(u, v)
            add_edge(w, z)
            
            new_score = get_score()
            
            if new_score < best_score or (T > 0.1 and random.random() < math.exp(-(new_score - best_score) / T)):
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [x for x in adj]
                    best_degs = [x for x in degs]
            else:
                # Revert
                remove_edge(w, z)
                add_edge(u, v)
                
        T *= cooling

    # Convert best_adj back to edge list
    edges = []
    for u in range(n):
        v_mask = best_adj[u]
        while v_mask:
            lsb = v_mask & -v_mask
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            v_mask ^= lsb
            
    # Verify edge count
    if len(edges) != target_edges:
        # Fallback to seed
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
