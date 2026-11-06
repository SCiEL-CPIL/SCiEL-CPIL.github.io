import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0

    # Bitmask representation for fast K2,3 checks
    # adj_mask[i] is a bitmask of neighbors of vertex i
    # common_mask[u][v] = adj_mask[u] & adj_mask[v]
    # K2,3-free means: for all u != v, popcount(adj_mask[u] & adj_mask[v]) <= 2

    def popcount(x):
        return bin(x).count('1')

    def is_valid_add(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # After adding (u,v):
        # - u and v now share all their previous common neighbors (unchanged)
        # - For any w in N(u) (excluding v), w and v now share one more common neighbor (u)
        #   so we need: for all w in N(u)\{v}, |N(w) ∩ N(v) \ {u}| <= 1  (i.e., popcount((adj_mask[w] & adj_mask[v]) & ~bit(u)) <= 1)
        #   Equivalently: popcount(adj_mask[w] & adj_mask[v]) <= 1 + (1 if u in N(v) else 0)
        #   Since u is NOT in N(v) yet, we need popcount(adj_mask[w] & adj_mask[v]) <= 1
        # - Similarly for w in N(v)\{u}: popcount(adj_mask[u] & adj_mask[w]) <= 1
        # - Also u and v: popcount(adj_mask[u] & adj_mask[v]) <= 2 (they'll be neighbors after)
        
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # u and v common neighbors (before adding): must be <= 2
        if popcount(mask_u & mask_v) > 2:
            return False
        
        # For each neighbor w of u (w != v): check common(w, v) <= 1
        # w is in mask_u, w != v (v not in mask_u yet)
        temp = mask_u
        while temp:
            lsb = temp & (-temp)
            w = lsb.bit_length() - 1
            if w == v:
                temp ^= lsb
                continue
            if popcount(adj_mask[w] & mask_v) > 1:
                temp ^= lsb
                return False
            temp ^= lsb
        
        # For each neighbor w of v (w != u): check common(u, w) <= 1
        temp = mask_v
        while temp:
            lsb = temp & (-temp)
            w = lsb.bit_length() - 1
            if w == u:
                temp ^= lsb
                continue
            if popcount(mask_u & adj_mask[w]) > 1:
                temp ^= lsb
                return False
            temp ^= lsb
        
        return True

    def add_edge_fast(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge_fast(adj_mask, u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    def edges_from_mask(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mask[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def edge_count(adj_mask):
        cnt = 0
        for u in range(n):
            cnt += popcount(adj_mask[u])
        return cnt // 2

    def validate_adj_mask(adj_mask):
        """Verify K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj_mask[u] & adj_mask[v]) > 2:
                    return False
        return True

    # Structured initial construction: 5 groups of 4 vertices
    # Within each group: complete graph K4
    # Between groups i and j: connect based on a pattern that limits common neighbors
    # 
    # Better: use the known construction. Let's try a greedy from a good seed.
    
    def greedy_construct(seed_offset=0):
        rng = random.Random(seed_offset)
        adj_mask = [0] * n
        
        # Start with a structured base: 5 cliques of size 4
        # Group g: vertices 4g .. 4g+3
        for g in range(5):
            base = 4 * g
            for i in range(4):
                for j in range(i + 1, 4):
                    adj_mask[base + i] |= (1 << (base + j))
                    adj_mask[base + j] |= (1 << (base + i))
        
        # Try to add inter-group edges greedily
        all_inter = []
        for g1 in range(5):
            for g2 in range(g1 + 1, 5):
                for i in range(4):
                    for j in range(4):
                        all_inter.append((4 * g1 + i, 4 * g2 + j))
        
        rng.shuffle(all_inter)
        
        for u, v in all_inter:
            if time.time() > deadline:
                break
            if is_valid_add(adj_mask, u, v):
                add_edge_fast(adj_mask, u, v)
        
        return adj_mask

    def local_search(adj_mask, max_time=30.0):
        """Aggressive local search: add edges, 1-edge swaps, 2-edge swaps."""
        current = list(adj_mask)
        best = list(current)
        best_cnt = edge_count(current)
        start_ls = time.time()
        ls_deadline = start_ls + max_time
        
        # Phase 1: Greedy add
        all_edges_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
        for _ in range(20):
            if time.time() > ls_deadline:
                break
            random.shuffle(all_edges_list)
            added = False
            for u, v in all_edges_list:
                if not (current[u] & (1 << v)):
                    if is_valid_add(current, u, v):
                        add_edge_fast(current, u, v)
                        added = True
            if not added:
                break
        
        cnt = edge_count(current)
        if cnt > best_cnt:
            best_cnt = cnt
            best = list(current)
        
        # Phase 2: 1-edge swap (remove one, try to add others)
        for _ in range(2000):
            if time.time() > ls_deadline:
                break
            if not any(current[u] & (1 << v) for u in range(n) for v in range(u+1, n)):
                break
            
            # Pick random existing edge
            existing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current[u] & (1 << v):
                        existing.append((u, v))
            if not existing:
                break
            u, v = random.choice(existing)
            
            saved = list(current)
            remove_edge_fast(current, u, v)
            
            # Try to add as many as possible
            random.shuffle(all_edges_list)
            for a, b in all_edges_list:
                if not (current[a] & (1 << b)):
                    if is_valid_add(current, a, b):
                        add_edge_fast(current, a, b)
            
            cnt = edge_count(current)
            if cnt > best_cnt:
                best_cnt = cnt
                best = list(current)
            else:
                current = saved
        
        # Phase 3: 2-edge swap
        for _ in range(500):
            if time.time() > ls_deadline:
                break
            
            existing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current[u] & (1 << v):
                        existing.append((u, v))
            if len(existing) < 2:
                break
            
            (u1, v1), (u2, v2) = random.sample(existing, 2)
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue
            
            saved = list(current)
            remove_edge_fast(current, u1, v1)
            remove_edge_fast(current, u2, v2)
            
            random.shuffle(all_edges_list)
            for a, b in all_edges_list:
                if not (current[a] & (1 << b)):
                    if is_valid_add(current, a, b):
                        add_edge_fast(current, a, b)
            
            cnt = edge_count(current)
            if cnt > best_cnt:
                best_cnt = cnt
                best = list(current)
            else:
                current = saved
        
        return best

    # Main loop
    best_adj = None
    best_count = 0
    
    # Run multiple greedy constructions
    for seed in range(20):
        if time.time() > deadline:
            break
        adj = greedy_construct(seed)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj)
    
    # Local search on best
    if best_adj and time.time() < deadline - 5:
        improved = local_search(best_adj, max_time=min(30.0, deadline - time.time() - 3))
        cnt = edge_count(improved)
        if cnt > best_count:
            best_count = cnt
            best_adj = improved
    
    # More local search iterations if time permits
    while time.time() < deadline - 3:
        improved = local_search(best_adj, max_time=min(15.0, deadline - time.time() - 2))
        cnt = edge_count(improved)
        if cnt > best_count:
            best_count = cnt
            best_adj = improved
        else:
            # Try a fresh greedy + local search
            adj = greedy_construct(random.randint(0, 10000))
            cnt2 = edge_count(adj)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = list(adj)
            if time.time() > deadline - 5:
                break
            improved2 = local_search(best_adj, max_time=min(10.0, deadline - time.time() - 2))
            cnt3 = edge_count(improved2)
            if cnt3 > best_count:
                best_count = cnt3
                best_adj = improved2
    
    # Validate and return
    if best_adj is None:
        # Fallback: just return a valid small graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if is_valid_add(adj, u, v):
                    add_edge_fast(adj, u, v)
        best_adj = adj
    
    # Final validation
    if not validate_adj_mask(best_adj):
        # Something went wrong, rebuild conservatively
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if is_valid_add(adj, u, v):
                    add_edge_fast(adj, u, v)
        best_adj = adj
    
    return edges_from_mask(best_adj)
