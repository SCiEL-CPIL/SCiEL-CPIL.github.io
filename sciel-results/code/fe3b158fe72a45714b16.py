import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n = 16
    all_vertices = set(range(n))
    
    # Q3 edge list (8 vertices, 12 edges)
    q3_edges = [
        (0,1),(0,2),(0,4),
        (1,3),(1,5),
        (2,3),(2,6),
        (3,7),
        (4,5),(4,6),
        (5,7),
        (6,7)
    ]
    
    def has_q3_with_edge(adj_mask, u, v):
        """Check if adding edge u-v creates a Q3. adj_mask is list of int bitmasks."""
        # Get neighbors of u and v (excluding each other)
        neu_mask = adj_mask[u] & ~(1 << v) & ~(1 << u)
        nev_mask = adj_mask[v] & ~(1 << u) & ~(1 << v)
        
        neu = [x for x in range(n) if (neu_mask >> x) & 1]
        nev = [x for x in range(n) if (nev_mask >> x) & 1]
        
        if len(neu) < 2 or len(nev) < 2:
            return False
        
        # In Q3 with u=0, v=1:
        # pos2, pos4: neighbors of u
        # pos3, pos5: neighbors of v
        # pos2-pos3, pos4-pos5 must be edges
        # pos6: neighbor of pos2 and pos4
        # pos7: neighbor of pos3 and pos5
        # pos6-pos7, pos3-pos7, pos5-pos7 must be edges
        
        for pos2 in neu:
            for pos4 in neu:
                if pos2 == pos4:
                    continue
                # Precompute: neighbors of pos2 and pos4
                n_pos2_mask = adj_mask[pos2]
                n_pos4_mask = adj_mask[pos4]
                
                for pos3 in nev:
                    if pos3 == pos2 or pos3 == pos4:
                        continue
                    if not ((n_pos2_mask >> pos3) & 1):
                        continue  # pos2-pos3 required
                    n_pos3_mask = adj_mask[pos3]
                    
                    for pos5 in nev:
                        if pos5 == pos2 or pos5 == pos3 or pos5 == pos4:
                            continue
                        if not ((n_pos4_mask >> pos5) & 1):
                            continue  # pos4-pos5 required
                        n_pos5_mask = adj_mask[pos5]
                        
                        # Find candidates for pos6: neighbor of both pos2 and pos4
                        common_24 = n_pos2_mask & n_pos4_mask
                        excluded_mask = (1 << u) | (1 << v) | (1 << pos2) | (1 << pos3) | (1 << pos4) | (1 << pos5)
                        cand_pos6_mask = common_24 & ~excluded_mask
                        
                        # Find candidates for pos7: neighbor of both pos3 and pos5
                        common_35 = n_pos3_mask & n_pos5_mask
                        cand_pos7_mask = common_35 & ~excluded_mask
                        
                        # Check all pairs
                        pos6 = cand_pos6_mask
                        while pos6:
                            p6 = (pos6 & -pos6).bit_length() - 1
                            pos6 &= pos6 - 1
                            
                            pos7 = cand_pos7_mask
                            while pos7:
                                p7 = (pos7 & -pos7).bit_length() - 1
                                pos7 &= pos7 - 1
                                
                                if p6 == p7:
                                    continue
                                # Check remaining edges: pos6-pos7, pos3-pos7, pos5-pos7
                                if ((n_pos2_mask >> p7) & 1) or ((n_pos4_mask >> p7) & 1):
                                    # pos6 must be neighbor of pos7
                                    if ((adj_mask[p6] >> p7) & 1) and \
                                       ((n_pos3_mask >> p7) & 1) and \
                                       ((n_pos5_mask >> p7) & 1):
                                        return True
        return False
    
    def count_edges(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    def verify_no_q3(adj_mask):
        """Verify the graph has no Q3 subgraph."""
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    if has_q3_with_edge(adj_mask, u, v):
                        return False
        return True
    
    best_edge_count = 0
    best_adj_mask = None
    
    random.seed(42)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def try_greedy_add(adj_mask, time_limit):
        """Greedily add missing edges, return (adj_mask, edge_count)."""
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_mask[i] >> j) & 1)]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > time_limit:
                break
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            if has_q3_with_edge(adj_mask, u, v):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        return adj_mask
    
    def local_search(adj_mask, time_limit, max_swaps=500):
        """Try edge swaps: remove one edge, add another. Return improved adj_mask."""
        current_count = count_edges(adj_mask)
        improved = True
        swaps_done = 0
        
        while improved and swaps_done < max_swaps and time.time() < time_limit:
            improved = False
            swaps_done += 1
            
            present = [(i, j) for i in range(n) for j in range(i+1, n) if (adj_mask[i] >> j) & 1]
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_mask[i] >> j) & 1)]
            
            if not present or not missing:
                break
            
            # Try random swap
            for attempt in range(20):
                if time.time() > time_limit:
                    return adj_mask
                if not present or not missing:
                    break
                rem = random.choice(present)
                add = random.choice(missing)
                # Remove rem, add add
                adj_mask[rem[0]] &= ~(1 << rem[1])
                adj_mask[rem[1]] &= ~(1 << rem[0])
                adj_mask[add[0]] |= (1 << add[1])
                adj_mask[add[1]] |= (1 << add[0])
                
                # Check if still Q3-free (only need to check edges involved in the change)
                # Actually we need to verify no Q3 was created by the new edge
                ok = not has_q3_with_edge(adj_mask, add[0], add[1])
                if ok:
                    # Also check that removing rem didn't leave a Q3 (it can't, since we only removed)
                    new_count = count_edges(adj_mask)
                    if new_count > current_count:
                        current_count = new_count
                        improved = True
                        # Update lists
                        present = [(i, j) for i in range(n) for j in range(i+1, n) if (adj_mask[i] >> j) & 1]
                        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_mask[i] >> j) & 1)]
                    else:
                        # Revert
                        adj_mask[add[0]] &= ~(1 << add[1])
                        adj_mask[add[1]] &= ~(1 << add[0])
                        adj_mask[rem[0]] |= (1 << rem[1])
                        adj_mask[rem[1]] |= (1 << rem[0])
                else:
                    # Revert
                    adj_mask[add[0]] &= ~(1 << add[1])
                    adj_mask[add[1]] &= ~(1 << add[0])
                    adj_mask[rem[0]] |= (1 << rem[1])
                    adj_mask[rem[1]] |= (1 << rem[0])
        
        return adj_mask
    
    def clean_q3(adj_mask, time_limit):
        """Remove edges that create Q3 until graph is Q3-free."""
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_mask[u] >> v) & 1:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        if has_q3_with_edge(adj_mask, u, v):
                            changed = True
                        else:
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
        return adj_mask
    
    # Starting structures
    def make_k3_13():
        adj = [0] * n
        for u in range(3):
            for v in range(3, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def make_k8_8():
        adj = [0] * n
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def make_k4_12():
        adj = [0] * n
        for u in range(4):
            for v in range(4, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def make_random(density):
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < density:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        return adj
    
    def make_turan(t, parts):
        """Turan graph T(n, t): complete t-partite with balanced parts."""
        adj = [0] * n
        part_size = n // t
        remainder = n % t
        # Assign vertices to parts
        part_of = []
        idx = 0
        for p in range(t):
            size = part_size + (1 if p < remainder else 0)
            for _ in range(size):
                part_of.append(p)
                idx += 1
        for i in range(n):
            for j in range(i+1, n):
                if part_of[i] != part_of[j]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    # Run multiple restarts with different strategies
    num_restarts = 300
    
    for restart in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        adj_mask = [0] * n
        
        if restart == 0:
            adj_mask = make_k3_13()
        elif restart == 1:
            adj_mask = make_k8_8()
        elif restart == 2:
            adj_mask = make_k4_12()
        elif restart == 3:
            adj_mask = make_turan(3, 16)  # T(16,3)
        elif restart == 4:
            adj_mask = make_turan(4, 16)  # T(16,4)
        elif restart == 5:
            adj_mask = make_turan(5, 16)  # T(16,5)
        elif restart < 15:
            # Various random densities
            density = 0.3 + 0.05 * (restart - 5)
            adj_mask = make_random(density)
        else:
            # Mix of structures
            strategy = random.random()
            if strategy < 0.2:
                adj_mask = make_k3_13()
            elif strategy < 0.4:
                adj_mask = make_k8_8()
            elif strategy < 0.6:
                adj_mask = make_random(random.uniform(0.3, 0.7))
            else:
                adj_mask = make_turan(random.randint(3, 5), 16)
        
        # Clean up Q3s
        adj_mask = clean_q3(adj_mask, deadline - 5)
        
        # Greedy add
        adj_mask = try_greedy_add(adj_mask, deadline - 5)
        
        # Local search with edge swaps
        adj_mask = local_search(adj_mask, deadline - 3, max_swaps=200)
        
        ec = count_edges(adj_mask)
        
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj_mask = adj_mask[:]
    
    # Final verification
    if best_adj_mask is None:
        best_adj_mask = make_k3_13()
    
    if not verify_no_q3(best_adj_mask):
        best_adj_mask = make_k3_13()
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_mask[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
