import time
import random
import math

DEADLINE = 135  # Safety margin for 150s kill

def _compute_score_fast(adj_mask, n):
    """Fast score computation using bitmasks."""
    deg = [0] * n
    for i in range(n):
        deg[i] = bin(adj_mask[i]).count('1')
    
    max_sum = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                common = adj_mask[u] & adj_mask[v]
                common &= ~((1 << (v + 1)) - 1)
                if common:
                    # Just check if any triangle exists with this pair
                    # For max sum, we need to find the max w
                    while common:
                        w_bit = common & (-common)
                        w = w_bit.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= w_bit
    return max_sum

def _compute_score_and_degrees(adj_mask, n):
    """Compute score and degrees."""
    deg = [0] * n
    for i in range(n):
        deg[i] = bin(adj_mask[i]).count('1')
    
    max_sum = 0
    worst_triangles = []
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                common = adj_mask[u] & adj_mask[v]
                common &= ~((1 << (v + 1)) - 1)
                while common:
                    w_bit = common & (-common)
                    w = w_bit.bit_length() - 1
                    s = deg[u] + deg[v] + deg[w]
                    worst_triangles.append(((u, v, w), s))
                    if s > max_sum:
                        max_sum = s
                    common ^= w_bit
    
    worst_triangles.sort(key=lambda x: x[1], reverse=True)
    return max_sum, deg, worst_triangles

def _get_all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _swap_edge(adj_mask, n, remove_edge, add_edge, all_pairs_set):
    """Perform an edge swap and return new adj_mask."""
    u1, v1 = remove_edge
    u2, v2 = add_edge
    
    # Remove edge
    adj_mask[u1] ^= (1 << v1)
    adj_mask[v1] ^= (1 << u1)
    
    # Add edge
    adj_mask[u2] |= (1 << v2)
    adj_mask[v2] |= (1 << u2)
    
    return adj_mask

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    all_pairs = _get_all_pairs(n)
    all_pairs_set = set(all_pairs)
    
    best_score = float('inf')
    best_adj = None
    
    # Initial graph: K_{8,8} + 1 internal edge
    # This gives score 26 typically
    adj_init = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj_init[u] |= (1 << v)
            adj_init[v] |= (1 << u)
    # Add internal edge (0,1)
    adj_init[0] |= (1 << 1)
    adj_init[1] |= (1 << 0)
    
    # Verify edge count
    edge_count = sum(bin(adj_init[i]).count('1') for i in range(n)) // 2
    # Should be 65
    
    # Main SA loop with multiple restarts
    max_restarts = 8
    iters_per_restart = 40000
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Initialize
        if restart == 0:
            adj = adj_init[:]
        elif restart == 1:
            # Different internal edge
            adj = [0] * n
            for u in range(8):
                for v in range(8, 16):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            adj[3] |= (1 << 5)
            adj[5] |= (1 << 3)
        elif restart == 2:
            # K_{8,8} + 2 internal edges, remove 1 cross edge
            adj = [0] * n
            for u in range(8):
                for v in range(8, 16):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            adj[0] |= (1 << 1)
            adj[1] |= (1 << 0)
            adj[2] |= (1 << 3)
            adj[3] |= (1 << 2)
            # Remove cross edge (0,8)
            adj[0] &= ~(1 << 8)
            adj[8] &= ~(1 << 0)
        elif restart == 3:
            # K_{8,8} + edge in right side
            adj = [0] * n
            for u in range(8):
                for v in range(8, 16):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            adj[8] |= (1 << 9)
            adj[9] |= (1 << 8)
        else:
            # Random 65-edge graph
            random.shuffle(all_pairs)
            adj = [0] * n
            for u, v in all_pairs[:65]:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        current_score = _compute_score_fast(adj, n)
        current_deg = [bin(adj[i]).count('1') for i in range(n)]
        
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
        
        # SA parameters
        T_start = 3.0
        T_end = 0.01
        total_iters = iters_per_restart
        
        for i in range(total_iters):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            progress = i / total_iters
            T = T_start * math.pow(T_end / T_start, progress)
            
            # Get current worst triangles for targeted moves
            if i % 100 == 0:  # Recompute every 100 steps
                current_score, current_deg, worst_tris = _compute_score_and_degrees(adj, n)
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj[:]
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.3:
                # Targeted move: remove edge from worst triangle
                if worst_tris:
                    worst_tri = worst_tris[0][0]
                    u, v, w = worst_tri
                    # Try removing each edge of the worst triangle
                    tri_edges = [(min(u,v), max(u,v)), (min(u,w), max(u,w)), (min(v,w), max(v,w))]
                    # Sort by degree sum (higher first)
                    tri_edges.sort(key=lambda e: current_deg[e[0]] + current_deg[e[1]], reverse=True)
                    
                    for rem_edge in tri_edges:
                        # Find a non-edge to add
                        non_edges = [pe for pe in all_pairs if pe not in {(u,v) for u in range(n) for v in range(u+1,n) if adj[u] & (1 << v)}]
                        if not non_edges:
                            continue
                        
                        # Sort non-edges by degree sum (lower is better)
                        non_edges.sort(key=lambda pe: current_deg[pe[0]] + current_deg[pe[1]])
                        
                        # Try top 10 non-edges
                        for add_edge in non_edges[:10]:
                            # Make temporary copy
                            new_adj = adj[:]
                            # Remove edge
                            ra, rb = rem_edge
                            new_adj[ra] ^= (1 << rb)
                            new_adj[rb] ^= (1 << ra)
                            # Add edge
                            aa, ab = add_edge
                            new_adj[aa] |= (1 << ab)
                            new_adj[ab] |= (1 << aa)
                            
                            new_score = _compute_score_fast(new_adj, n)
                            if new_score < current_score:
                                current_score = new_score
                                adj = new_adj
                                current_deg = [bin(adj[k]).count('1') for k in range(n)]
                                if current_score < best_score:
                                    best_score = current_score
                                    best_adj = adj[:]
                                break
                            elif random.random() < math.exp((current_score - new_score) / T):
                                current_score = new_score
                                adj = new_adj
                                current_deg = [bin(adj[k]).count('1') for k in range(n)]
                                break
            else:
                # Random swap
                # Pick random edge to remove
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u] & (1 << v)]
                if not edge_list:
                    continue
                rem_edge = random.choice(edge_list)
                
                # Pick random non-edge to add
                non_edges = [pe for pe in all_pairs if pe not in set(edge_list)]
                if not non_edges:
                    continue
                add_edge = random.choice(non_edges)
                
                # Make temporary copy
                new_adj = adj[:]
                ra, rb = rem_edge
                new_adj[ra] ^= (1 << rb)
                new_adj[rb] ^= (1 << ra)
                aa, ab = add_edge
                new_adj[aa] |= (1 << ab)
                new_adj[ab] |= (1 << aa)
                
                new_score = _compute_score_fast(new_adj, n)
                if new_score <= current_score:
                    current_score = new_score
                    adj = new_adj
                    current_deg = [bin(adj[k]).count('1') for k in range(n)]
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj[:]
                else:
                    delta = new_score - current_score
                    if delta > 0:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            current_score = new_score
                            adj = new_adj
                            current_deg = [bin(adj[k]).count('1') for k in range(n)]
        
        # After each restart, recompute
        current_score, current_deg, _ = _compute_score_and_degrees(adj, n)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
    
    # Final polish on best found
    if best_adj is not None and time.time() - start_time < DEADLINE - 5:
        adj = best_adj[:]
        current_score = best_score
        current_deg = [bin(adj[i]).count('1') for i in range(n)]
        _, _, worst_tris = _compute_score_and_degrees(adj, n)
        
        polish_iters = 50000
        T_start = 0.5
        T_end = 0.005
        
        for i in range(polish_iters):
            if time.time() - start_time > DEADLINE - 3:
                break
            
            progress = i / polish_iters
            T = T_start * math.pow(T_end / T_start, progress)
            
            # Recompute periodically
            if i % 50 == 0:
                current_score, current_deg, worst_tris = _compute_score_and_degrees(adj, n)
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj[:]
            
            move_type = random.random()
            
            if move_type < 0.5 and worst_tris:
                # Targeted move on worst triangle
                worst_tri = worst_tris[0][0]
                u, v, w = worst_tri
                tri_edges = [(min(u,v), max(u,v)), (min(u,w), max(u,w)), (min(v,w), max(v,w))]
                tri_edges.sort(key=lambda e: current_deg[e[0]] + current_deg[e[1]], reverse=True)
                
                for rem_edge in tri_edges:
                    edge_list = [(a, b) for a in range(n) for b in range(a + 1, n) if adj[a] & (1 << b)]
                    non_edges = [pe for pe in all_pairs if pe not in set(edge_list)]
                    if not non_edges:
                        continue
                    
                    non_edges.sort(key=lambda pe: current_deg[pe[0]] + current_deg[pe[1]])
                    
                    for add_edge in non_edges[:15]:
                        new_adj = adj[:]
                        ra, rb = rem_edge
                        new_adj[ra] ^= (1 << rb)
                        new_adj[rb] ^= (1 << ra)
                        aa, ab = add_edge
                        new_adj[aa] |= (1 << ab)
                        new_adj[ab] |= (1 << aa)
                        
                        new_score = _compute_score_fast(new_adj, n)
                        if new_score < current_score:
                            current_score = new_score
                            adj = new_adj
                            current_deg = [bin(adj[k]).count('1') for k in range(n)]
                            if current_score < best_score:
                                best_score = current_score
                                best_adj = adj[:]
                            break
                        elif random.random() < math.exp((current_score - new_score) / T):
                            current_score = new_score
                            adj = new_adj
                            current_deg = [bin(adj[k]).count('1') for k in range(n)]
                            break
            else:
                # Random swap
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u] & (1 << v)]
                if not edge_list:
                    continue
                rem_edge = random.choice(edge_list)
                
                non_edges = [pe for pe in all_pairs if pe not in set(edge_list)]
                if not non_edges:
                    continue
                add_edge = random.choice(non_edges)
                
                new_adj = adj[:]
                ra, rb = rem_edge
                new_adj[ra] ^= (1 << rb)
                new_adj[rb] ^= (1 << ra)
                aa, ab = add_edge
                new_adj[aa] |= (1 << ab)
                new_adj[ab] |= (1 << aa)
                
                new_score = _compute_score_fast(new_adj, n)
                if new_score <= current_score:
                    current_score = new_score
                    adj = new_adj
                    current_deg = [bin(adj[k]).count('1') for k in range(n)]
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj[:]
                else:
                    delta = new_score - current_score
                    if delta > 0:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            current_score = new_score
                            adj = new_adj
                            current_deg = [bin(adj[k]).count('1') for k in range(n)]
        
        # Final update
        final_score = _compute_score_fast(adj, n)
        if final_score < best_score:
            best_score = final_score
            best_adj = adj[:]
    
    # Fallback
    if best_adj is None:
        adj = adj_init[:]
        best_adj = adj[:]
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
