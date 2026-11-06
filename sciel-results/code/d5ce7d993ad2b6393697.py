import time
import random

def construct():
    deadline = time.time() + 110
    n = 40
    adj = [0] * n
    masks = [1 << i for i in range(n)]
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def has_c6_with_edge(u, v):
        a_mask = adj[u] & ~masks[v]
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            b_mask = adj[a] & ~masks[u] & ~masks[v]
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                c_mask = adj[b] & ~masks[u] & ~masks[a] & ~masks[v]
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
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
        # Try to add all possible edges
        # To make it deterministic for a given state, we can iterate in fixed order
        # or random. Fixed is faster to check? Random helps escape?
        # For local search, fixed order is fine if we do multiple passes?
        # Let's use random order for diversity in the repair step.
        random.shuffle(all_edges)
        for u, v in all_edges:
            if (adj[u] & masks[v]) == 0:
                if not has_c6_with_edge(u, v):
                    add_edge(u, v)

    # Init
    for k in range(8):
        start = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                add_edge(start + i, start + j)
                
    greedy_fill()
    best_edges = count_edges()
    best_adj = adj[:]
    
    while time.time() < deadline:
        # Local Search: Try swaps
        # Find a random edge to remove
        # Find a random non-edge to add
        # Check if swap is valid (no C6)
        
        # To speed up, we can try a few random swaps
        improved = False
        for _ in range(10): # Try 10 swaps per iteration
            if time.time() > deadline: break
            
            # Pick random edge (u,v)
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if (adj[u] & masks[v]) == 0: continue
            
            # Pick random non-edge (x,y)
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y: continue
            if (adj[x] & masks[y]) != 0: continue
            
            # Check if removing (u,v) and adding (x,y) is valid
            # 1. Remove (u,v)
            remove_edge(u, v)
            
            # 2. Check if adding (x,y) creates C6
            if not has_c6_with_edge(x, y):
                # Add (x,y)
                add_edge(x, y)
                
                # 3. Greedy fill
                greedy_fill()
                
                new_count = count_edges()
                if new_count > best_edges:
                    best_edges = new_count
                    best_adj = adj[:]
                    improved = True
                    break # Accept and restart local search
                else:
                    # Revert
                    remove_edge(x, y)
                    add_edge(u, v)
            else:
                # Revert
                add_edge(u, v)
                
        if not improved:
            # Perturb: Remove a few edges and greedy fill
            for _ in range(2):
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
