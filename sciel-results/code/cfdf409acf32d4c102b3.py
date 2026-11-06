"""Program for the graph construction task."""
import time
import random

DEADLINE = 140  # safety margin before 150s hard kill

def _compute_score_fast(n, adj_masks, degs):
    """
    Compute the largest degree sum over triangles using bitmask intersection.
    """
    max_sum = 0
    for u in range(n):
        adj_u = adj_masks[u]
        for v in range(u + 1, n):
            if (adj_u >> v) & 1:
                common = adj_u & adj_masks[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    m ^= lsb
    return max_sum

def _graph_to_state(edges):
    """Convert edge list to adj_masks and degs."""
    n = 16
    adj_masks = [0] * n
    degs = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        degs[u] += 1
        degs[v] += 1
    return adj_masks, degs

def _state_to_edges(adj_masks):
    """Convert adj_masks back to edge list."""
    n = 16
    edges = []
    for i in range(n):
        mask = adj_masks[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    # Precompute all potential pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_score = float('inf')
    best_edges = None
    
    def try_local_search(initial_edges, time_budget_remaining):
        nonlocal best_score, best_edges
        
        adj_masks, degs = _graph_to_state(initial_edges)
        current_score = _compute_score_fast(n, adj_masks, degs)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = _state_to_edges(adj_masks)
            
        # Local search loop with simulated annealing / guided search
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in initial_edges)
        current_edges_list = list(edge_set)
        non_edges = [p for p in all_pairs if p not in edge_set]
        
        # Simulated annealing parameters
        temperature = 5.0
        cooling_rate = 0.999
        min_temperature = 0.1
        
        steps = 0
        max_steps = 50000
        
        while time.time() - start_time < time_budget_remaining and steps < max_steps:
            steps += 1
            
            # Pick random edge to remove
            rem_u, rem_v = random.choice(current_edges_list)
            
            # Pick random non-edge to add
            if not non_edges:
                break
            add_u, add_v = random.choice(non_edges)
            
            # Perform swap
            adj_masks[rem_u] &= ~(1 << rem_v)
            adj_masks[rem_v] &= ~(1 << rem_u)
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            edge_set.remove((rem_u, rem_v))
            
            adj_masks[add_u] |= (1 << add_v)
            adj_masks[add_v] |= (1 << add_u)
            degs[add_u] += 1
            degs[add_v] += 1
            edge_set.add((add_u, add_v))
            
            new_score = _compute_score_fast(n, adj_masks, degs)
            
            # Accept if better, or with probability exp(-delta/T)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = _state_to_edges(adj_masks)
                
                # Update lists
                idx_rem = current_edges_list.index((rem_u, rem_v))
                current_edges_list[idx_rem] = (add_u, add_v)
                idx_add = non_edges.index((add_u, add_v))
                non_edges[idx_add] = (rem_u, rem_v)
            else:
                # Decide whether to accept worse move (simulated annealing)
                delta = new_score - current_score
                if temperature > min_temperature:
                    import math
                    if random.random() < math.exp(-delta / temperature):
                        # Accept worse move
                        current_score = new_score
                        idx_rem = current_edges_list.index((rem_u, rem_v))
                        current_edges_list[idx_rem] = (add_u, add_v)
                        idx_add = non_edges.index((add_u, add_v))
                        non_edges[idx_add] = (rem_u, rem_v)
                    else:
                        # Revert
                        adj_masks[add_u] &= ~(1 << add_v)
                        adj_masks[add_v] &= ~(1 << add_u)
                        degs[add_u] -= 1
                        degs[add_v] -= 1
                        edge_set.remove((add_u, add_v))
                        
                        adj_masks[rem_u] |= (1 << rem_v)
                        adj_masks[rem_v] |= (1 << rem_u)
                        degs[rem_u] += 1
                        degs[rem_v] += 1
                        edge_set.add((rem_u, rem_v))
                else:
                    # Revert if not accepting
                    adj_masks[add_u] &= ~(1 << add_v)
                    adj_masks[add_v] &= ~(1 << add_u)
                    degs[add_u] -= 1
                    degs[add_v] -= 1
                    edge_set.remove((add_u, add_v))
                    
                    adj_masks[rem_u] |= (1 << rem_v)
                    adj_masks[rem_v] |= (1 << rem_u)
                    degs[rem_u] += 1
                    degs[rem_v] += 1
                    edge_set.add((rem_u, rem_v))
            
            # Cool down
            temperature *= cooling_rate
            
            # If we found a very good score, we can stop early
            if best_score <= 23:
                break
    
    # Multiple restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate initial graph
        if i % 3 == 0:
            # Random graph
            shuffled_pairs = all_pairs[:]
            random.shuffle(shuffled_pairs)
            init_edges = [list(p) for p in shuffled_pairs[:m_target]]
        elif i % 3 == 1:
            # K8,8 + random edge within partition
            edges = [[u, v] for u in range(8) for v in range(8, 16)]
            if random.random() < 0.5:
                u = random.randint(0, 7)
                v = random.randint(0, 7)
                while u == v:
                    v = random.randint(0, 7)
            else:
                u = random.randint(8, 15)
                v = random.randint(8, 15)
                while u == v:
                    v = random.randint(8, 15)
            edges.append([u, v])
            init_edges = edges
        else:
            # Near-complete bipartite with some internal edges
            edges = []
            # Start with K8,8
            for u in range(8):
                for v in range(8, 16):
                    if random.random() < 0.95:  # Keep most bipartite edges
                        edges.append([u, v])
            
            # Add internal edges to reach 65
            internal_pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)] + \
                             [(i, j) for i in range(8, 16) for j in range(i + 1, 16)]
            random.shuffle(internal_pairs)
            
            for pair in internal_pairs:
                if len(edges) >= m_target:
                    break
                edges.append(list(pair))
            
            # If we have too many, remove some
            if len(edges) > m_target:
                random.shuffle(edges)
                edges = edges[:m_target]
            
            init_edges = edges
        
        try_local_search(init_edges, DEADLINE)
        
        if best_score <= 23:
            break
            
    return {'n': n, 'edges': best_edges}
