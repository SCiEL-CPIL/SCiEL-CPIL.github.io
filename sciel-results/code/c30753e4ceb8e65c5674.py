import time
import random

def construct():
    n = 20
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
        
    def can_add(u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Check common neighbors of u and v
        if popcount(adj[u] & adj[v]) > 2:
            return False
            
        # 2. Check neighbors of v: for each x in N(v), |N(u) & N(x)| <= 1
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            if popcount(adj[u] & adj[x]) > 1:
                return False
            mask_v ^= lsb
            
        # 3. Check neighbors of u: for each y in N(u), |N(v) & N(y)| <= 1
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            y = lsb.bit_length() - 1
            if popcount(adj[v] & adj[y]) > 1:
                return False
            mask_u ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges():
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    def get_edges_list():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def get_missing_edges():
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    missing.append((i, j))
        return missing

    def get_existing_edges():
        existing = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    existing.append((i, j))
                mask ^= lsb
        return existing

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_adj = [0] * n
    best_count = 0
    
    # Initialize with 5xK4 structure
    # Vertices 0-3, 4-7, 8-11, 12-15, 16-19
    groups = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]]
    for g in groups:
        for i in range(4):
            for j in range(i+1, 4):
                add_edge(g[i], g[j])
                
    current_count = count_edges()
    best_count = current_count
    best_adj = adj[:]
    
    # Local Search with Simulated Annealing
    temperature = 10.0
    cooling = 0.999
    min_temp = 0.1
    
    while time.time() < deadline and temperature > min_temp:
        # Try to add any possible edge
        added = False
        missing = get_missing_edges()
        random.shuffle(missing)
        for u, v in missing:
            if can_add(u, v):
                add_edge(u, v)
                added = True
        
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        if not added:
            # Try swap moves
            existing = get_existing_edges()
            if not existing:
                break
            
            move_made = False
            # Try random swap
            if random.random() < 0.5:
                u1, v1 = random.choice(existing)
                # Pick a random missing edge
                missing = get_missing_edges()
                if missing:
                    u2, v2 = random.choice(missing)
                    # Check if swap improves or is neutral
                    # Remove u1-v1
                    remove_edge(u1, v1)
                    if can_add(u2, v2):
                        add_edge(u2, v2)
                        new_count = count_edges()
                        # Swap is neutral in count, but might enable future adds
                        # For now, accept if it doesn't reduce count (it doesn't)
                        # But we need to ensure validity, which can_add checks.
                        move_made = True
                    else:
                        add_edge(u1, v1)
            
            if not move_made:
                # Try single edge removal to escape
                u1, v1 = random.choice(existing)
                remove_edge(u1, v1)
                move_made = True
                
        temperature *= cooling

    # Final greedy polish
    for _ in range(10):
        added = False
        missing = get_missing_edges()
        random.shuffle(missing)
        for u, v in missing:
            if can_add(u, v):
                add_edge(u, v)
                added = True
        if not added:
            break
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]

    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges
