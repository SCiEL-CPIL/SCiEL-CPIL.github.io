import time
import random
import numpy as np

def construct():
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # Adjacency as bitmasks for fast operations
    adj = [0] * N
    adj_list = [[] for _ in range(N)]

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        adj_list[u].append(v)
        adj_list[v].append(u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj_list[u].remove(v)
        adj_list[v].remove(u)

    def edge_count():
        return sum(bin(adj[i]).count('1') for i in range(N)) // 2

    def has_c6_fast(u, v):
        """
        Check if adding edge (u, v) creates a C6.
        We need to check if there is a path of length 4 from u to v in the existing graph
        (i.e., u-a-b-c-d-v where a,b,c,d are distinct and not u,v).
        
        Using bitmasks for efficiency:
        - N(u) = neighbors of u (excluding v for the path)
        - N(v) = neighbors of v (excluding u for the path)
        - We need: exists a in N(u), d in N(v), and a path a-b-c-d of length 3
        
        More precisely: exists a in N(u)\{v}, d in N(v)\{u}, b, c such that
        a-b, b-c, c-d are edges, all distinct, none are u or v.
        
        Optimized approach:
        1. For each a in N(u)\{v}, compute N2[a] = neighbors of a reachable in 2 steps
           (i.e., N2[a] = union of N(b) for b in N(a)\{u,v,a})
        2. Check if N2[a] intersects N(v)\{u} for any a
        3. But we also need to ensure the path is simple (no repeated vertices)
        
        Actually, let's use a more direct bitmask approach:
        - Let S1 = N(u) & ~(1<<v)  # neighbors of u except v
        - For each a in S1, let S2_a = N(a) & ~(1<<u) & ~(1<<v) & ~(1<<a)  # neighbors of a except u,v,a
        - For each b in S2_a, let S3_ab = N(b) & ~(1<<u) & ~(1<<v) & ~(1<<a) & ~(1<<b)  # neighbors of b except u,v,a,b
        - Check if S3_ab intersects N(v) & ~(1<<u)
        
        To speed this up, precompute for each node x the bitmask of its neighbors.
        """
        # Get neighbors of u and v, excluding each other
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        # For each a in N(u), check if there's a path a-b-c-d where d in N(v)
        # a-b: b in N(a)
        # b-c: c in N(b)  
        # c-d: d in N(c) and d in N(v)
        # So we need: exists a in N(u), b in N(a), c in N(b) such that N(c) intersects N(v)
        # And all of a,b,c,d are distinct and not u,v
        
        # Precompute: for each node c, does N(c) intersect N(v)?
        # Let's compute nv_mask = nv (neighbors of v)
        # For each c, check if adj[c] & nv != 0
        
        # Iterate over a in N(u)
        a_mask = nu
        while a_mask:
            lsb_a = a_mask & (-a_mask)
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            # Get neighbors of a, excluding u, v, a
            b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            while b_mask:
                lsb_b = b_mask & (-b_mask)
                b = lsb_b.bit_length() - 1
                b_mask ^= lsb_b
                
                # Get neighbors of b, excluding u, v, a, b
                c_mask = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                while c_mask:
                    lsb_c = c_mask & (-c_mask)
                    c = lsb_c.bit_length() - 1
                    c_mask ^= lsb_c
                    
                    # Check if c is adjacent to any d in N(v)
                    if adj[c] & nv:
                        # Found a path u-a-b-c-d-v, which with edge v-u forms a C6
                        return True
        
        return False

    def can_add(u, v):
        """Return True if adding edge (u,v) does NOT create a C6."""
        return not has_c6_fast(u, v)

    # Initialize with 8 disjoint K5s
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(base + a, base + b)
    
    current_edges = edge_count()
    best_edges = current_edges
    best_adj = adj[:]
    best_adj_list = [lst[:] for lst in adj_list]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    missing_edges = [p for p in all_pairs if not (adj[p[0]] & (1 << p[1]))]
    
    while time.time() < deadline:
        random.shuffle(missing_edges)
        added_in_pass = 0
        new_missing = []
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if can_add(u, v):
                add_edge(u, v)
                added_in_pass += 1
                current_edges += 1
            else:
                new_missing.append((u, v))
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            best_adj_list = [lst[:] for lst in adj_list]
        
        missing_edges = new_missing
        
        if added_in_pass == 0:
            if current_edges > 0:
                kick_size = random.randint(2, 5)
                existing_edges = [(u, v) for u in range(N) for v in adj_list[u] if u < v]
                random.shuffle(existing_edges)
                for _ in range(kick_size):
                    if not existing_edges:
                        break
                    u, v = existing_edges.pop()
                    remove_edge(u, v)
                    current_edges -= 1
                    missing_edges.append((u, v))
    
    final_edges = []
    for u in range(N):
        for v in best_adj_list[u]:
            if u < v:
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
