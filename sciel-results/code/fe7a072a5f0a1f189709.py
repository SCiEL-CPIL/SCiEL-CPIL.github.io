import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 100  # Tighter safety margin to ensure finish
    
    best_edges = 0
    best_graph = None
    
    def count_edges(G):
        return sum(G[i].bit_count() for i in range(n)) // 2
    
    def find_k33(G):
        """Find a K3,3 in G. Returns list of 6 edges forming K3,3 or None."""
        for i, j, k in itertools.combinations(range(n), 3):
            mask = G[i] & G[j] & G[k]
            if mask.bit_count() >= 3:
                bits = []
                m = mask
                while len(bits) < 3 and m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    m ^= lsb
                return [(i, l) for l in bits] + [(j, l) for l in bits] + [(k, l) for l in bits]
        return None
    
    def is_add_valid(G, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        A = G[v] & ~(1 << u)  # neighbors of v except u
        B = G[u] & ~(1 << v)  # neighbors of u except v
        
        # Check if bipartite graph between A and B contains K2,2
        # Iterate over pairs in A
        A_list = []
        m = A
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            A_list.append(idx)
            m ^= lsb
        
        for i in range(len(A_list)):
            a1 = A_list[i]
            for j in range(i+1, len(A_list)):
                a2 = A_list[j]
                common = G[a1] & G[a2] & B
                if common.bit_count() >= 2:
                    return False
        return True
    
    def remove_one_k33_edge(G, rng):
        """Find a K3,3 and remove one random edge from it. Returns True if removed."""
        k33 = find_k33(G)
        if k33 is None:
            return False
        u, v = rng.choice(k33)
        G[u] &= ~(1 << v)
        G[v] &= ~(1 << u)
        return True
    
    def greedy_add_edges(G, rng):
        """Try to add all possible edges greedily."""
        # Build list of missing edges
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not (G[i] & (1 << j)):
                    missing.append((i, j))
        
        # Keep a list of indices to avoid rebuilding 'missing' list repeatedly
        # We will iterate through the list and skip already added ones
        while missing:
            # Shuffle remaining missing edges
            rng.shuffle(missing)
            added_any = False
            new_missing = []
            for u, v in missing:
                # Check if still missing (might have been added by a previous edge in this pass? 
                # No, adding (u,v) only affects validity of other edges, it doesn't add (u,v) again.
                # But if (u,v) was added, it's no longer missing. 
                # Wait, if we add (u,v), we don't remove it from 'missing' immediately in this loop structure 
                # unless we check if it's already present.
                if G[u] & (1 << v):
                    continue # Already added
                if is_add_valid(G, u, v):
                    G[u] |= (1 << v)
                    G[v] |= (1 << u)
                    added_any = True
                    # Do not add to new_missing
                else:
                    new_missing.append((u, v))
            if not added_any:
                break
            missing = new_missing
            if time.time() > deadline:
                break
    
    def local_search_swap(G, rng):
        """Try edge swaps: remove edge (u,v), add edge (x,y) if valid."""
        swaps = 0
        all_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if G[i] & (1 << j):
                    all_edges.append((i, j))
        
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not (G[i] & (1 << j)):
                    missing.append((i, j))
        
        if not all_edges or not missing:
            return 0
            
        rng.shuffle(all_edges)
        rng.shuffle(missing)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            # Remove edge (u,v)
            G[u] &= ~(1 << v)
            G[v] &= ~(1 << u)
            
            # Try to add a missing edge
            found = False
            for x, y in missing:
                if (x == u and y == v) or (x == v and y == u):
                    continue
                if G[x] & (1 << y):
                    continue
                if is_add_valid(G, x, y):
                    G[x] |= (1 << y)
                    G[y] |= (1 << x)
                    found = True
                    swaps += 1
                    break
            
            if not found:
                # Put edge back
                G[u] |= (1 << v)
                G[v] |= (1 << u)
        
        return swaps
    
    # Multiple random restarts
    # We will run until time limit
    seed = 0
    while time.time() < deadline - 1:
        rng = random.Random(seed)
        
        # Initialize: start with random graph at varying densities
        # Higher density starts require more repair but might converge to denser graphs
        p = rng.uniform(0.5, 0.8)
        G = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    G[i] |= (1 << j)
                    G[j] |= (1 << i)
        
        # Phase 1: Remove edges to make K3,3-free
        max_removals = 100
        removals = 0
        while removals < max_removals:
            if time.time() > deadline:
                break
            if not remove_one_k33_edge(G, rng):
                break
            removals += 1
        
        # Phase 2: Greedy add edges
        greedy_add_edges(G, rng)
        
        # Phase 3: Local search with swaps
        # Limit local search time to ensure we have time for next restart
        local_start = time.time()
        local_limit = min(deadline - 2, local_start + 5) # Max 5s per local search phase
        while time.time() < local_limit:
            local_search_swap(G, rng)
        
        # Count edges
        ec = count_edges(G)
        if ec > best_edges:
            best_edges = ec
            best_graph = G[:]
            
        seed += 1
    
    # If no valid graph found, return a simple valid one
    if best_graph is None:
        # Return a star graph (valid, K3,3-free)
        edges = [[0, i] for i in range(1, n)]
        return edges
    
    edges = []
    for u in range(n):
        mask = best_graph[u]
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m ^= lsb
    return edges
