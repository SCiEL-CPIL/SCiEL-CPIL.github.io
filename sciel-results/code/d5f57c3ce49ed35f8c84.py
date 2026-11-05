import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_edges = 0
    best_adj = None
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_add(adj, u, v):
        # Check if adding edge (u, v) keeps graph K2,3-free
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Check common neighbors of u and v
        # If |N(u) & N(v)| >= 3, then (u,v) would be a K2,3 center pair
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False
            
        # 2. Check neighbors of u
        # For any x in N(u), if x is also in N(v), then v becomes a new common neighbor of (u, x)
        # We need to ensure |N(u) & N(x)| < 2 for all such x
        # Note: N(u) does not include v yet in the mask, but we are simulating adding it.
        # The common neighbors of (u, x) in the new graph will be (N(u) \cup {v}) & N(x)
        # = (N(u) & N(x)) | ({v} & N(x))
        # If v in N(x), size increases by 1.
        # So we need current |N(u) & N(x)| <= 1 if v in N(x).
        
        # Iterate through neighbors of u
        nu = adj[u]
        # We only care about x that are neighbors of v, because only then v is added to common neighbors of (u,x)
        # Actually, if x is not neighbor of v, v is not in N(x), so no change.
        # So check x in N(u) & N(v)
        candidates_u = nu & adj[v]
        
        while candidates_u:
            lsb = candidates_u & -candidates_u
            x = lsb.bit_length() - 1
            # Check common neighbors of u and x
            common_ux = adj[u] & adj[x]
            if popcount(common_ux) >= 2:
                return False
            candidates_u ^= lsb
            
        # 3. Check neighbors of v
        nv = adj[v]
        candidates_v = nv & adj[u]
        
        while candidates_v:
            lsb = candidates_v & -candidates_v
            y = lsb.bit_length() - 1
            # Check common neighbors of v and y
            common_vy = adj[v] & adj[y]
            if popcount(common_vy) >= 2:
                return False
            candidates_v ^= lsb
            
        return True

    def greedy_fill(adj, order):
        # Try to add edges in given order
        changed = False
        for u, v in order:
            if is_valid_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                changed = True
        return changed

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    def local_search(adj):
        # Returns improved adj
        improved = True
        while improved:
            improved = False
            
            # 1. Try adding any missing edge
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_pairs)
            
            for u, v in all_pairs:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
            if improved:
                continue
                
            # 2. If stuck, try removing an edge and refilling
            # Pick a random edge to remove
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] >> j) & 1]
            if not edges:
                break
                
            # Try a few random removals
            for _ in range(20):
                if time.time() > deadline:
                    return adj
                u, v = random.choice(edges)
                if not ((adj[u] >> v) & 1):
                    continue
                    
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Refill
                all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                random.shuffle(all_pairs)
                
                # Count before
                edges_before = count_edges(adj)
                
                greedy_fill(adj, all_pairs)
                
                edges_after = count_edges(adj)
                
                if edges_after > edges_before:
                    improved = True
                    break
                else:
                    # Restore
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
        return adj

    # Restart loop
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize with random sparse graph or empty
        adj = [0] * n
        
        # Random greedy start
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        
        # Greedy fill
        greedy_fill(adj, all_pairs)
        
        # Local search
        adj = local_search(adj)
        
        current_edges = count_edges(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
    # Convert best_adj to list of edges
    if best_adj is None:
        # Fallback to seed
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
        
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
