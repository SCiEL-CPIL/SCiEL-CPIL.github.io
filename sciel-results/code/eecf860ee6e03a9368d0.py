"""Dense Q3-free graph on 16 vertices via vertex-pair partition + local search."""
import time
import random

def construct():
    deadline = time.time() + 110
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3 subgraph.
        adj[i] is a bitmask of neighbors of i.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def count_edges(adj):
        count = 0
        for u in range(n):
            count += adj[u].bit_count()
        return count // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def greedy_add(adj, edge_order):
        added = 0
        for u, v in edge_order:
            if time.time() > deadline - 3:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def local_search(adj, max_iters=200):
        """
        Local search with edge swaps and perturbations.
        """
        best_adj = adj[:]
        best_count = count_edges(adj)
        
        # Precompute missing edges
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
        
        # Phase 1: Multiple greedy passes
        for pass_i in range(5):
            if time.time() > deadline - 5:
                break
            if not missing:
                break
            shuffled = missing[:]
            random.shuffle(shuffled)
            greedy_add(adj, shuffled)
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
        
        # Phase 2: Targeted swaps
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            missing_now = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
            if not missing_now:
                break
            
            # Try random swap: remove an edge, try to add missing edges
            found_improvement = False
            for _attempt in range(5):
                if time.time() > deadline - 5:
                    break
                if not missing_now:
                    break
                
                u_miss, v_miss = random.choice(missing_now)
                
                # Find existing edges to potentially remove
                existing = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        bit = mask & (-mask)
                        v = bit.bit_length() - 1
                        if v > u:
                            existing.append((u, v))
                        mask -= bit
                
                if not existing:
                    break
                
                u_rem, v_rem = random.choice(existing)
                
                # Try removing this edge
                trial = adj[:]
                trial[u_rem] &= ~(1 << v_rem)
                trial[v_rem] &= ~(1 << u_rem)
                
                # Try adding the missing edge
                if not creates_q3_fast(u_miss, v_miss, trial):
                    trial[u_miss] |= (1 << v_miss)
                    trial[v_miss] |= (1 << u_miss)
                
                # Greedy add more edges
                missing_trial = [(u, v) for u in range(n) for v in range(u+1, n) if not (trial[u] & (1 << v))]
                random.shuffle(missing_trial)
                greedy_add(trial, missing_trial)
                
                new_count = count_edges(trial)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = trial[:]
                    adj = trial[:]
                    missing_now = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
                    found_improvement = True
                    break
            
            if not found_improvement:
                # Perturb: remove a random edge and greedy add
                existing = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        bit = mask & (-mask)
                        v = bit.bit_length() - 1
                        if v > u:
                            existing.append((u, v))
                        mask -= bit
                
                if not existing:
                    break
                
                u_rem, v_rem = random.choice(existing)
                trial = adj[:]
                trial[u_rem] &= ~(1 << v_rem)
                trial[v_rem] &= ~(1 << u_rem)
                
                missing_trial = [(u, v) for u in range(n) for v in range(u+1, n) if not (trial[u] & (1 << v))]
                random.shuffle(missing_trial)
                greedy_add(trial, missing_trial)
                
                new_count = count_edges(trial)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = trial[:]
                    adj = trial[:]
        
        return best_adj, best_count

    # Vertex-pair partition strategy
    def partition_init(partition_seed=0):
        """
        Initialize graph based on vertex partition.
        Vertices are split into two groups. Edges within each group and between groups
        are added greedily, respecting Q3 constraint.
        """
        adj = [0] * n
        random.seed(partition_seed)
        
        # Try different partition sizes
        for split in [4, 5, 6, 7, 8]:
            if time.time() > deadline - 5:
                break
            
            group_a = list(range(split))
            group_b = list(range(split, n))
            
            # Shuffle groups for diversity
            random.shuffle(group_a)
            random.shuffle(group_b)
            
            # Build edge order: inter-group edges first, then intra-group
            edge_order = []
            
            # Inter-group edges
            for u in group_a:
                for v in group_b:
                    if u < v:
                        edge_order.append((u, v))
                    else:
                        edge_order.append((v, u))
            
            # Intra-group A edges
            for i in range(len(group_a)):
                for j in range(i + 1, len(group_a)):
                    u, v = group_a[i], group_a[j]
                    if u < v:
                        edge_order.append((u, v))
                    else:
                        edge_order.append((v, u))
            
            # Intra-group B edges
            for i in range(len(group_b)):
                for j in range(i + 1, len(group_b)):
                    u, v = group_b[i], group_b[j]
                    if u < v:
                        edge_order.append((u, v))
                    else:
                        edge_order.append((v, u))
            
            random.shuffle(edge_order)
            
            trial_adj = [0] * n
            greedy_add(trial_adj, edge_order)
            trial_count = count_edges(trial_adj)
            
            if trial_count > count_edges(adj):
                adj = trial_adj[:]
        
        return adj

    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    # Strategy 1: Partition-based initialization
    for seed in range(10):
        if time.time() > deadline - 15:
            break
        
        adj = partition_init(seed)
        adj, cnt = local_search(adj)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
    
    # Strategy 2: Random greedy + local search
    for seed in range(10):
        if time.time() > deadline - 15:
            break
        
        random.seed(seed + 100)
        adj = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        greedy_add(adj, order)
        
        adj, cnt = local_search(adj)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
    
    # Strategy 3: Spectral-inspired initialization
    if time.time() < deadline - 10:
        # Use degree-based ordering
        adj = [0] * n
        order = all_edges[:]
        
        # Sort by vertex index difference (favors "close" vertices)
        order.sort(key=lambda e: abs(e[0] - e[1]))
        random.shuffle(order)
        greedy_add(adj, order)
        
        adj, cnt = local_search(adj)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
    
    # Final polish
    if time.time() < deadline - 2:
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (best_adj[u] & (1 << v))]
        
        random.shuffle(remaining)
        greedy_add(best_adj, remaining)
        
        remaining2 = [(u, v) for u in range(n) for v in range(u+1, n) 
                      if not (best_adj[u] & (1 << v))]
        if remaining2:
            remaining2.sort(key=lambda e: (best_adj[e[0]].bit_count() + best_adj[e[1]].bit_count()))
            greedy_add(best_adj, remaining2)
        
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)
