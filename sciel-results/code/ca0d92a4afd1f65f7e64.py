import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges_total = n_vertices * d // 2
    
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Precompute edges
    # Edge index: u * d + i where u < u^(1<<i) is not unique index.
    # Let's just list all (u, i) where u < u^(1<<i).
    edges = []
    edge_map = {} # (u, i) -> idx
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, i))
                edge_map[(u, i)] = idx
                idx += 1
    
    # 2. Precompute conflicts
    # For each edge e_idx, find the 6 C4s it belongs to.
    # A C4 is defined by (u, i, j) where i != j.
    # The 4 edges are:
    # e1: (u, i)
    # e2: (u^(1<<i), j)  -> Note: u^(1<<i) might be > v, so we need to normalize to (min, dim)
    # e3: (u, j)
    # e4: (u^(1<<j), i)
    
    # Helper to get edge index from (u, dim)
    def get_edge_idx(u, dim):
        v = u ^ (1 << dim)
        if u < v:
            return edge_map[(u, dim)]
        else:
            return edge_map[(v, dim)]

    conflicts = [[] for _ in range(n_edges_total)]
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                # Face spanned by i and j at corner u
                # Edges:
                e1_u, e1_d = u, i
                e2_u, e2_d = u ^ (1 << i), j
                e3_u, e3_d = u, j
                e4_u, e4_d = u ^ (1 << j), i
                
                idx1 = get_edge_idx(e1_u, e1_d)
                idx2 = get_edge_idx(e2_u, e2_d)
                idx3 = get_edge_idx(e3_u, e3_d)
                idx4 = get_edge_idx(e4_u, e4_d)
                
                # For each of these 4 edges, the other 3 are its partners in this C4
                partners = [idx1, idx2, idx3, idx4]
                for k in range(4):
                    # The edge at k is blocked if the other 3 are present
                    other = [p for p in range(4) if p != k]
                    conflicts[partners[k]].append((partners[other[0]], partners[other[1]], partners[other[2]]))
                
    # 3. State and Operations
    # present: bytearray of length n_edges_total
    # count: int
    
    def can_add(state, idx):
        # Check 6 C4s
        for p1, p2, p3 in conflicts[idx]:
            if state[p1] and state[p2] and state[p3]:
                return False
        return True
        
    def greedy_fill(state, order):
        count = 0
        # We need to track count efficiently. 
        # Since we modify state, we should pass count or recalculate?
        # Recalculating sum(state) is O(N). N=448. 
        # But we add one by one.
        for idx in order:
            if not state[idx]:
                if can_add(state, idx):
                    state[idx] = 1
                    count += 1
        return count

    # 4. ILS Loop
    best_state = bytearray(n_edges_total)
    best_count = 0
    
    # Initial Greedy
    order = list(range(n_edges_total))
    random.shuffle(order)
    best_count = greedy_fill(best_state, order)
    
    current_kick_size = 2
    stagnation_counter = 0
    
    while time.time() - start_time < time_limit:
        # Perturb Best
        current_state = best_state[:]
        
        # Remove random edges
        for _ in range(current_kick_size):
            # Pick random present edge
            # To speed up, we could keep a list of present edges, but 448 is small.
            # Just pick random idx and check if present.
            attempts = 0
            while attempts < 10:
                idx = random.randint(0, n_edges_total - 1)
                if current_state[idx]:
                    current_state[idx] = 0
                    break
                attempts += 1
        
        # Local Search: Greedy Fill with shuffled order
        order = list(range(n_edges_total))
        random.shuffle(order)
        new_count = greedy_fill(current_state, order)
        
        if new_count > best_count:
            best_count = new_count
            best_state = current_state
            stagnation_counter = 0
            # Reset kick size to small
            current_kick_size = 2
        else:
            stagnation_counter += 1
            if stagnation_counter % 10 == 0:
                # Increase kick size to escape local minima
                current_kick_size = min(current_kick_size + 1, 20)
            else:
                # Keep kick size or slightly vary
                current_kick_size = max(2, min(20, current_kick_size + random.choice([-1, 0, 1])))

    # 5. Output
    out = []
    for idx, (u, i) in enumerate(edges):
        if best_state[idx]:
            v = u ^ (1 << i)
            out.append([u, v])
            
    return out
