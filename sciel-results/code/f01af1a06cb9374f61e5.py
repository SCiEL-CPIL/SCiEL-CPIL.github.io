"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _score_bitmask(adj_bits, n, edge_count):
    """
    Compute max triangle degree sum using bitmasks.
    adj_bits: list of n integers, each a bitmask of neighbors.
    Returns max degree sum over all triangles.
    """
    deg = [bin(adj_bits[i]).count('1') for i in range(n)]
    max_sum = 0
    for i in range(n):
        ai = adj_bits[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                common = ai & adj_bits[j] & (~((1 << (j + 1)) - 1))
                while common:
                    lsb = common & (-common)
                    k = lsb.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > max_sum:
                        max_sum = s
                    common ^= lsb
    return max_sum

def _build_adj_bits(edges, n):
    """Build bitmask adjacency representation from edge list."""
    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    return adj_bits

def _sa_search(n, edges, deadline, max_iter=200000, seed=42):
    """
    Simulated annealing with bitmask scoring.
    Edge swaps: remove one edge, add another (maintaining edge count).
    Uses adaptive temperature and kick mechanism to escape local minima.
    """
    start_time = time.time()
    random.seed(seed)
    
    current_edges = list(edges)
    adj_bits = _build_adj_bits(current_edges, n)
    current_score = _score_bitmask(adj_bits, n, len(current_edges))
    best_edges = list(current_edges)
    best_score = current_score
    
    all_edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set((min(u, v), max(u, v)) for u, v in current_edges)
    
    # Temperature schedule with restarts
    T0 = 4.0
    T_min = 0.05
    n_iter = min(max_iter, 300000)
    
    # Track iterations without improvement for kick mechanism
    iter_since_improve = 0
    kick_threshold = 5000
    kick_count = 0
    max_kicks = 10
    
    for it in range(n_iter):
        if time.time() - start_time > deadline - 5:
            break
        
        # Adaptive temperature: cool faster when improving, slower when stuck
        if it < n_iter // 2:
            T = T0 * (T_min / T0) ** (it / (n_iter // 2))
        else:
            # Second half: very slow cooling
            T = T_min * 0.5 + 0.5 * T_min * math.exp(-(it - n_iter // 2) / 5000)
        
        # Pick edge to remove - bias toward high-degree vertices
        # This targets edges that are part of more triangles
        deg = [bin(adj_bits[i]).count('1') for i in range(n)]
        
        # Weighted selection: higher degree vertices more likely to be selected
        weights = [deg[i] + 1 for i in range(n)]
        total_w = sum(weights)
        
        # Pick two vertices for removal edge
        r1 = random.random() * total_w
        cumulative = 0
        u1 = 0
        for i in range(n):
            cumulative += weights[i]
            if r1 <= cumulative:
                u1 = i
                break
        
        r2 = random.random() * total_w
        cumulative = 0
        v1 = 0
        for i in range(n):
            if i == u1:
                continue
            cumulative += weights[i]
            if r2 <= cumulative:
                v1 = i
                break
        
        # Find the edge index
        e1 = (min(u1, v1), max(u1, v1))
        if e1 not in edge_set:
            continue
        
        # Find edge index in current_edges
        idx = -1
        for ei in range(len(current_edges)):
            if (min(current_edges[ei][0], current_edges[ei][1]), 
                max(current_edges[ei][0], current_edges[ei][1])) == e1:
                idx = ei
                break
        if idx == -1:
            continue
        
        # Pick edge to add - prefer non-edges that would create fewer triangles
        attempts = 0
        while attempts < 300:
            u2, v2 = all_edge_pairs[random.randrange(len(all_edge_pairs))]
            e2 = (min(u2, v2), max(u2, v2))
            if e2 not in edge_set and e2 != e1:
                break
            attempts += 1
        else:
            continue
        
        # Apply swap to bitmask
        adj_bits[u1] ^= (1 << v1)
        adj_bits[v1] ^= (1 << u1)
        adj_bits[u2] |= (1 << v2)
        adj_bits[v2] |= (1 << u2)
        
        new_score = _score_bitmask(adj_bits, n, len(current_edges))
        
        delta = new_score - current_score
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            # Accept
            current_edges[idx] = [u2, v2]
            edge_set.remove(e1)
            edge_set.add(e2)
            current_score = new_score
            iter_since_improve = 0
            
            if new_score < best_score:
                best_score = new_score
                best_edges = list(current_edges)
        else:
            # Revert
            adj_bits[u2] ^= (1 << v2)
            adj_bits[v2] ^= (1 << u2)
            adj_bits[u1] |= (1 << v1)
            adj_bits[v1] |= (1 << u1)
            iter_since_improve += 1
        
        # Kick mechanism: if stuck for too long, do a random perturbation
        if iter_since_improve > kick_threshold and kick_count < max_kicks:
            kick_count += 1
            # Do 3 random swaps to perturb
            for _ in range(3):
                idx_k = random.randrange(len(current_edges))
                uk1, vk1 = current_edges[idx_k]
                ek1 = (min(uk1, vk1), max(uk1, vk1))
                
                attempts_k = 0
                while attempts_k < 200:
                    uk2, vk2 = all_edge_pairs[random.randrange(len(all_edge_pairs))]
                    ek2 = (min(uk2, vk2), max(uk2, vk2))
                    if ek2 not in edge_set and ek2 != ek1:
                        break
                    attempts_k += 1
                else:
                    continue
                
                adj_bits[uk1] ^= (1 << vk1)
                adj_bits[vk1] ^= (1 << uk1)
                adj_bits[uk2] |= (1 << vk2)
                adj_bits[vk2] |= (1 << uk2)
                
                current_edges[idx_k] = [uk2, vk2]
                edge_set.remove(ek1)
                edge_set.add(ek2)
            
            current_score = _score_bitmask(adj_bits, n, len(current_edges))
            iter_since_improve = 0
    
    return best_edges, best_score

def _local_polish(n, edges, deadline):
    """
    Exhaustive single-swap local search with bitmask scoring.
    Repeats until no improvement found or deadline.
    """
    start_time = time.time()
    current_edges = list(edges)
    adj_bits = _build_adj_bits(current_edges, n)
    current_score = _score_bitmask(adj_bits, n, len(current_edges))
    best_edges = list(current_edges)
    best_score = current_score
    
    all_edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set((min(u, v), max(u, v)) for u, v in current_edges)
    
    improved = True
    while improved and time.time() - start_time < deadline - 3:
        improved = False
        for i in range(len(current_edges)):
            if time.time() - start_time > deadline - 3:
                break
            u1, v1 = current_edges[i]
            e1 = (min(u1, v1), max(u1, v1))
            
            for u2, v2 in all_edge_pairs:
                e2 = (min(u2, v2), max(u2, v2))
                if e2 in edge_set or e2 == e1:
                    continue
                
                # Try swap
                adj_bits[u1] ^= (1 << v1)
                adj_bits[v1] ^= (1 << u1)
                adj_bits[u2] |= (1 << v2)
                adj_bits[v2] |= (1 << u2)
                
                new_score = _score_bitmask(adj_bits, n, len(current_edges))
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = list(current_edges)
                    current_edges[i] = [u2, v2]
                    edge_set.remove(e1)
                    edge_set.add(e2)
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    adj_bits[u2] ^= (1 << v2)
                    adj_bits[v2] ^= (1 << u2)
                    adj_bits[u1] |= (1 << v1)
                    adj_bits[v1] |= (1 << u1)
    
    return best_edges, best_score

def _multi_start_sa(n, deadline):
    """
    Run SA from multiple different initial configurations.
    """
    start_time = time.time()
    best_overall = None
    best_score_overall = float('inf')
    
    # Multiple initial configurations
    configs = []
    
    # Config 1: K_{6,10} + matching in part of size 10 (best baseline)
    e1 = [[u, v] for u in range(6) for v in range(6, 16)]
    e1 += [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    configs.append(e1)
    
    # Config 2: K_{6,10} + different internal edges
    e2 = [[u, v] for u in range(6) for v in range(6, 16)]
    e2 += [[6, 8], [7, 9], [10, 12], [11, 13], [14, 15]]
    configs.append(e2)
    
    # Config 3: K_{8,8} + one edge
    e3 = [[u, v] for u in range(8) for v in range(8, 16)]
    e3.append([0, 1])
    configs.append(e3)
    
    # Config 4: K_{7,9} + 2 edges in part of size 9
    e4 = [[u, v] for u in range(7) for v in range(7, 16)]
    e4 += [[7, 8], [9, 10]]
    configs.append(e4)
    
    # Config 5: K_{5,11} + 10 edges in part of size 11
    e5 = [[u, v] for u in range(5) for v in range(5, 16)]
    inner = list(range(5, 16))
    random.seed(123)
    all_inner = [(inner[a], inner[b]) for a in range(len(inner)) for b in range(a+1, len(inner))]
    random.shuffle(all_inner)
    e5 += [list(x) for x in all_inner[:10]]
    configs.append(e5)
    
    # Config 6: Random graph with 65 edges
    random.seed(777)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    e6 = [list(x) for x in all_pairs[:65]]
    configs.append(e6)
    
    # Time budget per config
    time_per_config = (deadline - 15 - start_time) / len(configs)
    
    for ci, cfg in enumerate(configs):
        cfg_deadline = start_time + time_per_config * (ci + 1)
        # Ensure at least some time
        if cfg_deadline < start_time + 3:
            cfg_deadline = start_time + 3
        
        # Run SA with different seeds
        edges, score = _sa_search(n, cfg, cfg_deadline, max_iter=150000, seed=42 + ci * 100)
        
        if score < best_score_overall:
            best_score_overall = score
            best_overall = list(edges)
    
    return best_overall, best_score_overall

def construct():
    n = 16
    deadline = time.time() + 145  # 145s budget, 5s safety margin
    
    # Phase 1: Multi-start simulated annealing
    edges, score = _multi_start_sa(n, deadline)
    
    # Phase 2: Local polish
    edges, score = _local_polish(n, edges, deadline)
    
    # Phase 3: Another SA round starting from polished result (if time)
    remaining = deadline - 5 - time.time()
    if remaining > 10:
        edges2, score2 = _sa_search(n, edges, deadline - 5, max_iter=80000, seed=999)
        if score2 < score:
            edges, score = edges2, score2
        edges, score = _local_polish(n, edges, deadline)
    
    # Phase 4: Final polish with more iterations
    if time.time() < deadline - 5:
        edges, score = _local_polish(n, edges, deadline)
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
