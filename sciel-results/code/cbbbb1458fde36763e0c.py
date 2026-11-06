import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        return verts

    def is_valid(n, adj):
        # Check all subsets of size >= 6
        # We can iterate through all masks
        for mask in range(1 << n):
            # Quick check: if popcount < 6, skip
            # bit_count is fast
            if mask.bit_count() < 6:
                continue
            
            verts = get_vertices(mask)
            k = len(verts)
            # k is at least 6
            
            # Check if regular
            deg0 = (adj[verts[0]] & mask).bit_count()
            is_reg = True
            for v in verts[1:]:
                if (adj[v] & mask).bit_count() != deg0:
                    is_reg = False
                    break
            
            if is_reg:
                return False
        
        return True

    def generate_graph(n, p):
        adj = [0] * n
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        return adj, edges

    # Try n=21 first
    for n_target in [21, 20]:
        while time.time() < deadline:
            adj, edges = generate_graph(n_target, 0.5)
            if is_valid(n_target, adj):
                return {'n': n_target, 'edges': edges}
        
        # If we ran out of time for n_target, check if we have time for next
        if time.time() >= deadline:
            break

    # Fallback: empty graph on 5 vertices
    return {'n': 5, 'edges': []}
