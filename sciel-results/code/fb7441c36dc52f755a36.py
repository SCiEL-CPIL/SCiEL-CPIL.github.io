import time
import random
import numpy as np

def construct():
    n = 40
    deadline = 115.0
    start_time = time.time()
    
    # Bitset representation of adjacency
    adj = [0] * n
    
    # A2[i] is an integer where bit j is 1 if there is a path of length 2 from i to j
    A2 = [0] * n
    
    def update_A2():
        for i in range(n):
            mask = 0
            a = adj[i]
            while a:
                lsb = a & -a
                k = lsb.bit_length() - 1
                mask |= adj[k]
                a ^= lsb
            A2[i] = mask

    def check_c6_add(u, v):
        # Returns True if adding (u,v) creates a C6
        a_u = adj[u]
        a_v = adj[v]
        
        x_mask = a_u
        while x_mask:
            lsb_x = x_mask & -x_mask
            x = lsb_x.bit_length() - 1
            x_mask ^= lsb_x
            
            w_mask = a_v
            while w_mask:
                lsb_w = w_mask & -w_mask
                w = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                
                # Check if there is a path of length 3 between x and w
                # This is equivalent to N(x) & A2[w] != 0
                if adj[x] & A2[w]:
                    return True
        
        return False

    # 1. Spectral Heuristic Initialization:
    # Construct a random regular graph with degree d ~ 5.25 (total edges ~ 105).
    # This aims to start near the optimal edge count with a "random-like" structure
    # that minimizes spectral norm, reducing the likelihood of immediate C6s.
    
    # Target degree 5 (100 edges) or 6 (120 edges). 
    # Since max is 105, let's try to build a 5-regular graph first, then add edges.
    # 5-regular on 40 nodes is possible.
    
    # Generate a random 5-regular graph
    # Method: Pair up stubs.
    # Each node has 5 stubs. Total 200 stubs.
    # Randomly pair them up, ensuring no self-loops or multi-edges.
    
    max_attempts = 100
    success = False
    
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
            
        # Reset adj
        for i in range(n):
            adj[i] = 0
            
        # Create list of stubs: [0,0,0,0,0, 1,1,1,1,1, ...]
        stubs = []
        for i in range(n):
            for _ in range(5):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        # Pair them up
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            
            if u == v:
                valid = False
                break
            if (adj[u] >> v) & 1:
                valid = False
                break
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        if valid:
            success = True
            break
    
    if not success:
        # Fallback: Disjoint K5s
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    update_A2()
    
    best_edges = sum(bin(a).count("1") for a in adj) // 2
    best_adj = adj[:]
    
    # 2. Exact Greedy Search
    # Try to add any possible edge.
    # We will iterate through all non-edges and try to add them if valid.
    # Since the initial graph is already dense and "random", many edges might be addable.
    # However, adding an edge might block others. A simple greedy "add if possible" might get stuck.
    # But given the time limit, we can afford a more sophisticated search:
    # Simulated Annealing / Randomized Greedy with backtracking?
    
    # Let's use a fast randomized greedy loop.
    # Pick random non-edge. If valid, add. If not, skip.
    # This is fast. To escape local optima, we can occasionally remove an edge (swap).
    
    # But first, let's see how far pure greedy gets.
    # To make it faster, we can maintain a list of "candidate" edges.
    # But updating candidates is expensive.
    
    # Let's stick to the SA-like approach but optimized.
    # The previous program's SA was slow because update_A2 was O(N^2) or worse per step.
    # Here, update_A2 is O(N * D^2) roughly, which is fast for D~5.
    # Actually, update_A2 is sum of degrees. For D=5, it's very fast.
    
    T = 5.0
    cooling = 0.999
    
    while time.time() < deadline:
        if T < 0.1:
            T = 5.0
            
        # 80% Add, 20% Swap
        if random.random() < 0.8:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj[u] >> v) & 1:
                continue
            
            if not check_c6_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                update_A2()
                current_edges = sum(bin(a).count("1") for a in adj) // 2
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
        else:
            # Swap
            # Pick random edge to remove
            edges_list = []
            for i in range(n):
                a = adj[i]
                while a:
                    lsb = a & -a
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges_list.append((i, j))
                    a ^= lsb
            
            if not edges_list:
                continue
                
            a, b = random.choice(edges_list)
            
            # Save state
            adj_backup = adj[:]
            A2_backup = A2[:]
            
            # Remove
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            update_A2()
            
            # Add random edge
            c = random.randint(0, n - 1)
            d = random.randint(0, n - 1)
            if c == d or (adj[c] >> d) & 1:
                adj = adj_backup
                A2 = A2_backup
                T *= cooling
                continue
                
            if not check_c6_add(c, d):
                adj[c] |= (1 << d)
                adj[d] |= (1 << c)
                update_A2()
                new_edges = sum(bin(x).count("1") for x in adj) // 2
                
                # Accept if same or better
                # If worse, accept with prob exp(-delta/T)
                # Here delta is usually 0 (swap).
                # If we remove 1 and add 1, edge count is same.
                # We accept to explore neighborhood.
                if new_edges >= best_edges: # Only keep if it's at least as good as best? No, we want to explore.
                     pass
                # Actually, for swaps, edge count doesn't change. 
                # We accept if it allows future improvements. 
                # Simple heuristic: Accept if it doesn't decrease edge count (which it doesn't).
                # So we always accept valid swaps.
                
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = adj[:]
            else:
                adj = adj_backup
                A2 = A2_backup
        
        T *= cooling

    # Extract edges from best_adj
    edges = []
    for u in range(n):
        a = best_adj[u]
        while a:
            lsb = a & -a
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            a ^= lsb
            
    return edges
