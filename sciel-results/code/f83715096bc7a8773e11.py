"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute edge indices
    edge_list = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u+1, n):
            edge_to_idx[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    E = len(edge_list)  # 105
    
    # Precompute for each edge (u,v): the set of "conflict vertex pairs"
    # For edge i with endpoints (u,v), a conflict is a pair (s,t) where
    # s is adjacent to u (not v), t is adjacent to v (not u), s != t
    # The constraint: deg[s] != deg[t] for all such pairs
    
    # For faster checking, precompute for each edge the list of other edges
    # that share endpoint u or v
    edges_sharing_u = [[] for _ in range(n)]
    edges_sharing_v = [[] for _ in range(n)]
    for i in range(E):
        u, v = edge_list[i]
        edges_sharing_u[u].append(i)
        edges_sharing_v[v].append(i)
    
    # Build adjacency and degree from mask - optimized
    def get_adj_deg(mask):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for i in range(E):
            if mask & (1 << i):
                u, v = edge_list[i]
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        return adj, deg
    
    def is_valid_mask(mask):
        """Fast validity check using adjacency sets."""
        adj, deg = get_adj_deg(mask)
        present_edges = [i for i in range(E) if mask & (1 << i)]
        for i in present_edges:
            u, v = edge_list[i]
            ns = adj[u] - {v}
            nt = adj[v] - {u}
            # Collect degrees of ns and nt
            deg_ns = {}
            for s in ns:
                d = deg[s]
                if d in deg_ns:
                    return False
                deg_ns[d] = s
            for t in nt:
                d = deg[t]
                if d in deg_ns:
                    return False
                deg_ns[d] = t
        return True
    
    # Try multiple initial constructions
    def try_bipartite(a_size, b_size):
        """Try K_{a_size, b_size} with specific partition."""
        A = set(range(a_size))
        B = set(range(a_size, a_size + b_size))
        mask = 0
        for a in A:
            for b in B:
                mask |= (1 << edge_to_idx[(a, b)])
        return mask
    
    def greedy_add(mask, count):
        """Greedily add edges that maintain validity."""
        for i in range(E):
            if mask & (1 << i):
                continue
            new_mask = mask | (1 << i)
            if is_valid_mask(new_mask):
                mask = new_mask
                count += 1
        return mask, count
    
    # Try several initial structures
    candidates = []
    
    # K_{7,8}
    m = try_bipartite(7, 8)
    if is_valid_mask(m):
        candidates.append((m, bin(m).count('1')))
    
    # K_{6,9}
    m = try_bipartite(6, 9)
    if is_valid_mask(m):
        candidates.append((m, bin(m).count('1')))
    
    # K_{5,10}
    m = try_bipartite(5, 10)
    if is_valid_mask(m):
        candidates.append((m, bin(m).count('1')))
    
    # K_{8,7} (different partition)
    m = try_bipartite(8, 7)
    if is_valid_mask(m):
        candidates.append((m, bin(m).count('1')))
    
    # Greedy from empty
    m = 0
    m, c = greedy_add(m, 0)
    candidates.append((m, c))
    
    # Pick best candidate
    if candidates:
        candidates.sort(key=lambda x: -x[1])
        best_mask, best_count = candidates[0]
    else:
        best_mask, best_count = 0, 0
    
    # Phase 1: Greedy add
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        changed = False
        for i in range(E):
            if best_mask & (1 << i):
                continue
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                changed = True
        if not changed:
            break
    
    # Phase 2: Local search with 1-for-1 and 1-for-2 swaps
    present = [i for i in range(E) if best_mask & (1 << i)]
    absent = [i for i in range(E) if not (best_mask & (1 << i))]
    
    for iteration in range(2000):
        if time.time() > deadline - 5:
            break
        rng = random.Random(iteration)
        
        present = [i for i in range(E) if best_mask & (1 << i)]
        absent = [i for i in range(E) if not (best_mask & (1 << i))]
        
        if not present:
            break
        
        improved = False
        
        # Try single add
        for i in absent:
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-2 swap (improving)
        if absent and present and len(absent) >= 2:
            for _ in range(300):
                if time.time() > deadline - 5:
                    break
                ri = rng.choice(present)
                a1 = rng.choice(absent)
                a2 = rng.choice(absent)
                if a1 == a2:
                    continue
                new_mask = (best_mask & ~(1 << ri)) | (1 << a1) | (1 << a2)
                if is_valid_mask(new_mask):
                    best_mask = new_mask
                    best_count += 1
                    improved = True
                    break
        
        # Try 2-for-2 swap (neutral, may enable future improvements)
        if not improved and absent and present and len(absent) >= 2 and len(present) >= 2:
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                r1 = rng.choice(present)
                r2 = rng.choice(present)
                if r1 == r2:
                    continue
                a1 = rng.choice(absent)
                a2 = rng.choice(absent)
                if a1 == a2:
                    continue
                new_mask = best_mask
                new_mask &= ~(1 << r1)
                new_mask &= ~(1 << r2)
                new_mask |= (1 << a1)
                new_mask |= (1 << a2)
                if is_valid_mask(new_mask):
                    best_mask = new_mask
                    improved = True
                    break
        
        # Random restart every 200 iterations if stuck
        if not improved and iteration > 0 and iteration % 200 == 0:
            new_mask = best_mask
            num_remove = min(4, best_count)
            for _ in range(num_remove):
                ri = rng.choice(present)
                new_mask &= ~(1 << ri)
            # Re-add greedily
            new_mask, new_count = greedy_add(new_mask, bin(new_mask).count('1'))
            if new_count > best_count:
                best_mask = new_mask
                best_count = new_count
    
    # Phase 3: Final polish - try all single adds one more time
    for i in range(E):
        if best_mask & (1 << i):
            continue
        new_mask = best_mask | (1 << i)
        if is_valid_mask(new_mask):
            best_mask = new_mask
            best_count += 1
    
    # Convert result
    result = []
    for i in range(E):
        if best_mask & (1 << i):
            u, v = edge_list[i]
            result.append([int(u), int(v)])
    result.sort()
    return result
# EVOLVE-BLOCK-END
