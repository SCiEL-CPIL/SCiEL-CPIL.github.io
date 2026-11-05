"""Combined approach: C9 union baseline + optimized bitmask B&B + enhanced local search."""
import time
import random
import math
import sys

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the max triangle-free induced subgraph."""
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 150s limit
    
    N = 18
    full_mask = (1 << N) - 1
    popcount = [0] * (1 << N)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Helper: Check if a bitmask represents an independent set
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True
    
    # Helper: Check if adding edge (u,v) keeps graph K4-free
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)
    
    # Exact B&B for largest triangle-free induced subgraph - optimized
    def largest_triangle_free(adj, time_limit=None):
        best = [0]
        
        # Greedy heuristic: try multiple random orders
        for trial in range(8):
            order = list(range(N))
            random.shuffle(order)
            S = 0
            for v in order:
                if is_independent(adj[v] & S, adj):
                    S |= (1 << v)
            sz = popcount[S]
            if sz > best[0]:
                best[0] = sz
        
        if best[0] == N:
            return N
        
        # Precompute for each vertex: which other vertices form a triangle with it
        # tri[v] = bitmask of vertices w such that {v,w} has a common neighbor connected to both
        # Actually, for triangle-free: S is triangle-free iff for all v in S, N(v)∩S is independent
        # Key insight: v can join S iff N(v)∩S is independent in the induced subgraph on S
        
        # Branch and bound with better branching
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = popcount[S_mask]
            
            # Prune: upper bound
            if sz + popcount[C_mask] <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return
            
            # Filter candidates: remove those that can't be added
            new_C = 0
            c = C_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                if is_independent(adj[v] & S_mask, adj):
                    new_C |= lsb
                c ^= lsb
            
            if not new_C:
                if sz > best[0]:
                    best[0] = sz
                return
            
            C_mask = new_C
            if sz + popcount[C_mask] <= best[0]:
                return
            
            # Branch: pick vertex with max degree in candidate-induced subgraph for better pruning
            # Compute degree of each candidate within C_mask
            best_v = -1
            best_deg = -1
            c = C_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                deg = popcount[adj[v] & C_mask]
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                c ^= lsb
            
            lsb = 1 << best_v
            
            # Branch 1: Include v
            bnb(S_mask | lsb, C_mask ^ lsb)
            
            # Branch 2: Exclude v
            bnb(S_mask, C_mask ^ lsb)
        
        bnb(0, full_mask)
        return best[0]
    
    # Get edge list from adjacency bitmasks
    def get_edge_list(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Create C9 ∪ C9 baseline
    def make_c9_union_adj():
        adj = [0] * N
        for i in range(9):
            a = i
            b = (i + 1) % 9
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        for i in range(9):
            a = 9 + i
            b = 9 + (i + 1) % 9
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        return adj
    
    # Verify K4-free
    def is_k4_free(adj):
        for i in range(N):
            for j in range(i + 1, N):
                if not ((adj[i] >> j) & 1):
                    continue
                for k in range(j + 1, N):
                    if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1):
                        continue
                    for l in range(k + 1, N):
                        if ((adj[i] >> l) & 1) and ((adj[j] >> l) & 1) and ((adj[k] >> l) & 1):
                            return False
        return True
    
    random.seed(42)
    
    # Start with C9 ∪ C9
    current_adj = make_c9_union_adj()
    current_score = largest_triangle_free(current_adj, deadline - 5.0)
    best_adj = current_adj[:]
    best_score = current_score
    
    # Enhanced local search: single edge flips + 2-edge swaps
    def local_search(adj, best_score, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Single edge flips
            for i in range(N):
                if time.time() > deadline:
                    return best_score
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        return best_score
                    old_val = (adj[i] >> j) & 1
                    
                    if old_val == 0:
                        if not can_add_edge(adj, i, j):
                            continue
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    
                    score = largest_triangle_free(adj, deadline - 5.0)
                    
                    if score < best_score:
                        best_score = score
                        improved = True
                    else:
                        if old_val == 0:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                        else:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # 2-edge swaps: add one edge, remove another
            if time.time() < deadline - 5.0:
                all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
                random.shuffle(all_pairs)
                for _ in range(min(20, len(all_pairs))):
                    if time.time() > deadline:
                        break
                    e1, e2 = all_pairs[:2]
                    # Try: add e1, remove e2 (or vice versa)
                    for (u1, v1), (u2, v2) in [((e1[0], e1[1]), (e2[0], e2[1])), ((e2[0], e2[1]), (e1[0], e1[1]))]:
                        if time.time() > deadline:
                            break
                        # Check if we can do this swap
                        val1 = (adj[u1] >> v1) & 1
                        val2 = (adj[u2] >> v2) & 1
                        
                        # Only consider add-remove or remove-add
                        if val1 == val2:
                            continue
                        
                        adj_copy = adj[:]
                        if val1 == 0:
                            # Add e1
                            if not can_add_edge(adj_copy, u1, v1):
                                continue
                            adj_copy[u1] |= (1 << v1)
                            adj_copy[v1] |= (1 << u1)
                        else:
                            adj_copy[u1] &= ~(1 << v1)
                            adj_copy[v1] &= ~(1 << u1)
                        
                        if val2 == 0:
                            # Add e2
                            if not can_add_edge(adj_copy, u2, v2):
                                continue
                            adj_copy[u2] |= (1 << v2)
                            adj_copy[v2] |= (1 << u2)
                        else:
                            adj_copy[u2] &= ~(1 << v2)
                            adj_copy[v2] &= ~(1 << u2)
                        
                        score = largest_triangle_free(adj_copy, deadline - 5.0)
                        if score < best_score:
                            best_score = score
                            improved = True
                            adj = adj_copy
        
        return best_score
    
    # Run local search from C9 ∪ C9
    best_score = local_search(current_adj, best_score, deadline)
    if best_score < current_score:
        best_adj = current_adj[:]
    
    # Random restarts with perturbation
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        
        # Generate a random K4-free graph by greedy edge addition
        adj = [0] * N
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        if not is_k4_free(adj):
            continue
        
        score = largest_triangle_free(adj, deadline - 5.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            best_score = local_search(adj, best_score, deadline)
            if best_score < score:
                best_adj = adj[:]
        
        # Perturb C9 union
        if attempts % 3 == 0:
            adj2 = make_c9_union_adj()
            num_adds = random.randint(3, 15)
            for _ in range(num_adds):
                u, v = random.sample(range(N), 2)
                if can_add_edge(adj2, u, v):
                    adj2[u] |= (1 << v)
                    adj2[v] |= (1 << u)
            
            if is_k4_free(adj2):
                score2 = largest_triangle_free(adj2, deadline - 5.0)
                if score2 < best_score:
                    best_score = score2
                    best_adj = adj2[:]
                    best_score = local_search(adj2, best_score, deadline)
                    if best_score < score2:
                        best_adj = adj2[:]
        
        # Remove edges from current best
        if attempts % 5 == 0 and time.time() < deadline - 15.0:
            adj3 = best_adj[:]
            num_removals = random.randint(1, 5)
            current_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if (adj3[i] >> j) & 1]
            if current_edges:
                for _ in range(min(num_removals, len(current_edges))):
                    u, v = random.choice(current_edges)
                    adj3[u] &= ~(1 << v)
                    adj3[v] &= ~(1 << u)
                    current_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if (adj3[i] >> j) & 1]
                    if not current_edges:
                        break
                
                score3 = largest_triangle_free(adj3, deadline - 5.0)
                if score3 < best_score:
                    best_score = score3
                    best_adj = adj3[:]
    
    return get_edge_list(best_adj)

# EVOLVE-BLOCK-END
