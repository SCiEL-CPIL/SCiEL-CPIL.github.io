import time
import random

def construct():
    deadline = time.time() + 110
    n = 40
    adj = [0] * n
    masks = [1 << i for i in range(n)]
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def has_c6_with_edge(u, v):
        # Check if adding (u, v) creates a C6.
        # Path: u - a - b - c - d - v
        # a in N(u), b in N(a)\{u,v}, c in N(b)\{u,v,a}, d in N(c)\{u,v,a,b} intersect N(v)
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
        # Use a random order to break ties and allow diversity in restarts
        random.shuffle(all_edges)
        for u, v in all_edges:
            if (adj[u] & masks[v]) == 0:
                if not has_c6_with_edge(u, v):
                    add_edge(u, v)

    def random_init():
        adj = [0] * n
        # Start with a random sparse graph to avoid immediate C6s
        # Or start with a known good structure?
        # Let's try starting with a random matching or sparse graph
        for _ in range(20):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and (adj[u] & masks[v]) == 0:
                add_edge(u, v)
        greedy_fill()
        return adj[:]

    best_adj = [0] * n
    best_edges = 0
    
    # Multi-start greedy
    # We have 110s. Let's do many fast restarts.
    # Each restart: init + greedy.
    # How fast is greedy?
    # n=40, edges ~ 800. C6 check is cheap if graph is sparse, expensive if dense.
    # Let's try 50-100 restarts.
    
    start_time = time.time()
    restarts = 0
    while time.time() < deadline - 20: # Leave 20s for polishing
        adj = random_init()
        count = count_edges()
        if count > best_edges:
            best_edges = count
            best_adj = adj[:]
            restarts += 1
        if time.time() > deadline - 20:
            break
            
    # Polish the best solution found
    adj = best_adj[:]
    
    # Local Search: Simulated Annealing style
    # We try to add edges. If stuck, we remove edges and try again.
    current_edges = best_edges
    temp = 10.0
    cooling = 0.99
    
    while time.time() < deadline - 5:
        improved = False
        # Try to find an add move
        # Randomly select non-edges and try to add them
        for _ in range(50):
            if time.time() > deadline - 5: break
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y: continue
            if (adj[x] & masks[y]) != 0: continue
            
            if not has_c6_with_edge(x, y):
                add_edge(x, y)
                current_edges += 1
                improved = True
                # Try to fill greedily after an improvement?
                # Maybe just continue adding single edges for speed
                # Or do a quick greedy pass
                # Let's do a quick greedy pass to catch easy wins
                greedy_fill()
                new_count = count_edges()
                if new_count > best_edges:
                    best_edges = new_count
                    best_adj = adj[:]
                current_edges = new_count
                break # Restart the inner loop to re-evaluate state
        
        if not improved:
            # Perturbation: Remove a random edge and try to add a different one
            # Or remove 2 edges and re-greedy
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and (adj[u] & masks[v]):
                remove_edge(u, v)
                current_edges -= 1
                greedy_fill()
                new_count = count_edges()
                if new_count > best_edges:
                    best_edges = new_count
                    best_adj = adj[:]
                current_edges = new_count
            temp *= cooling

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] & masks[j]):
                edges.append([i, j])
    return edges
