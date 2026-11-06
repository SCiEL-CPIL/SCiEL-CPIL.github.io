import time
import random

def construct():
    deadline = time.time() + 110
    n = 40
    adj = [0] * n
    masks = [1 << i for i in range(n)]
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def has_c6_with_edge(u, v):
        # Check if adding (u,v) creates a C6.
        # This is true if there is a path of length 5 from u to v in the current graph.
        # Path: u - a - b - c - d - v
        # We can swap u and v to iterate over the smaller neighbor set first.
        if bin(adj[u]).count('1') > bin(adj[v]).count('1'):
            u, v = v, u
            
        a_mask = adj[u] & ~masks[v]
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # b must be in N(a), not u, not v
            b_mask = adj[a] & ~masks[u] & ~masks[v]
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # c must be in N(b), not u, a, v
                c_mask = adj[b] & ~masks[u] & ~masks[a] & ~masks[v]
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # d must be in N(c) and in N(v), not u, a, b
                    d_mask = adj[c] & adj[v]
                    forbidden = masks[u] | masks[a] | masks[b] | masks[c]
                    if d_mask & ~forbidden:
                        return True
        return False

    def add_edge(u, v):
        adj[u] |= masks[v]
        adj[v] |= masks[u]
        
    def remove_edge(u, v):
        adj[u] &= ~masks[v]
        adj[v] &= ~masks[u]

    def count_edges():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def greedy_fill():
        # Try to add all possible edges in random order
        random.shuffle(all_edges)
        for u, v in all_edges:
            if (adj[u] & masks[v]) == 0:
                if not has_c6_with_edge(u, v):
                    add_edge(u, v)

    # Init: 8 disjoint K5s
    for k in range(8):
        start = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                add_edge(start + i, start + j)
                
    greedy_fill()
    best_edges = count_edges()
    best_adj = adj[:]
    
    while time.time() < deadline:
        improved = False
        
        # Try 1-remove/2-add swaps
        # This allows escaping local optima more effectively than 1-remove/1-add
        for _ in range(20): # Increase attempts
            if time.time() > deadline: break
            
            # Pick random edge (u,v) to remove
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if (adj[u] & masks[v]) == 0: continue
            
            # Pick two random non-edges
            x1 = random.randint(0, n-1)
            y1 = random.randint(0, n-1)
            x2 = random.randint(0, n-1)
            y2 = random.randint(0, n-1)
            
            if x1 == y1 or x2 == y2: continue
            if (adj[x1] & masks[y1]) != 0 or (adj[x2] & masks[y2]) != 0: continue
            if (x1, y1) == (x2, y2) or (x1, y1) == (y2, x2) or (x2, y2) == (x1, y1) or (x2, y2) == (y1, x2): continue
            # Also ensure the two edges are distinct from the removed one
            if (u, v) == (x1, y1) or (v, u) == (x1, y1) or (u, v) == (x2, y2) or (v, u) == (x2, y2):
                 continue

            # Remove (u,v)
            remove_edge(u, v)
            
            # Try to add (x1, y1)
            if not has_c6_with_edge(x1, y1):
                add_edge(x1, y1)
                
                # Try to add (x2, y2)
                if not has_c6_with_edge(x2, y2):
                    add_edge(x2, y2)
                    
                    # Greedy fill
                    greedy_fill()
                    
                    new_count = count_edges()
                    if new_count > best_edges:
                        best_edges = new_count
                        best_adj = adj[:]
                        improved = True
                        break # Accept and restart local search
                    else:
                        # Revert
                        remove_edge(x2, y2)
                        remove_edge(x1, y1)
                        add_edge(u, v)
                else:
                    # Revert x1, y1
                    remove_edge(x1, y1)
                    add_edge(u, v)
            else:
                # Revert
                add_edge(u, v)
                
        if not improved:
            # Perturb: Remove a few edges and greedy fill
            # Randomly remove 3 edges to break local structure
            for _ in range(3):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and (adj[u] & masks[v]):
                    remove_edge(u, v)
            greedy_fill()
            if count_edges() > best_edges:
                best_edges = count_edges()
                best_adj = adj[:]

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] & masks[j]):
                edges.append([i, j])
    return edges
