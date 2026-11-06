import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin
    
    n = 16
    m_target = 65
    
    adj_mask = [0] * n
    
    def get_degrees():
        return [bin(adj_mask[i]).count('1') for i in range(n)]

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    def has_edge(u, v):
        return (adj_mask[u] & (1 << v)) != 0

    def calculate_score():
        degs = get_degrees()
        max_sum = 0
        for u in range(n):
            m = adj_mask[u]
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                if u < v:
                    common = adj_mask[u] & adj_mask[v]
                    c = common
                    while c:
                        w_bit = c & -c
                        w = w_bit.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                        c &= c - 1
                m &= m - 1
        return max_sum

    def init_k610():
        for i in range(n): adj_mask[i] = 0
        for u in range(6):
            for v in range(6, 16):
                add_edge(u, v)
        matching_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
        for u, v in matching_edges:
            add_edge(u, v)

    def init_random():
        for i in range(n): adj_mask[i] = 0
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for i in range(m_target):
            u, v = pairs[i]
            add_edge(u, v)

    def init_k79():
        for i in range(n): adj_mask[i] = 0
        for u in range(7):
            for v in range(7, 16):
                add_edge(u, v)
        # Add 2 edges in B (size 9)
        # Pick 4 distinct vertices from 7-15
        b_verts = list(range(7, 16))
        random.shuffle(b_verts)
        add_edge(b_verts[0], b_verts[1])
        add_edge(b_verts[2], b_verts[3])

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_score = 100
    best_mask = [0] * n
    
    # Try multiple initializations
    inits = [init_k610, init_k79, init_random, init_random, init_k610]
    random.shuffle(inits)
    
    for init_func in inits:
        if time.time() > deadline - 10:
            break
            
        init_func()
        current_score = calculate_score()
        if current_score < best_score:
            best_score = current_score
            best_mask = adj_mask[:]
            
        T = 1.0
        cooling = 0.999
        stagnation_count = 0
        
        while time.time() < deadline - 10:
            removed = None
            added = None
            
            attempts = 0
            while removed is None and attempts < 20:
                u, v = random.choice(all_pairs)
                if has_edge(u, v):
                    removed = (u, v)
                attempts += 1
                
            attempts = 0
            while added is None and attempts < 20:
                x, y = random.choice(all_pairs)
                if not has_edge(x, y):
                    added = (x, y)
                attempts += 1
                
            if removed is None or added is None:
                continue
                
            u, v = removed
            x, y = added
            
            remove_edge(u, v)
            add_edge(x, y)
            
            new_score = calculate_score()
            
            if new_score < current_score:
                current_score = new_score
                stagnation_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_mask = adj_mask[:]
            elif new_score == current_score:
                stagnation_count += 1
            else:
                delta = new_score - current_score
                if random.random() < np.exp(-delta / T):
                    stagnation_count += 1
                else:
                    add_edge(u, v)
                    remove_edge(x, y)
                    stagnation_count += 1
                    
            T *= cooling
            if T < 0.01:
                T = 0.01
                
            if stagnation_count > 500:
                break
                
    # Reconstruct edge list from best_mask
    final_edges = []
    for i in range(n):
        m = best_mask[i]
        while m:
            j = (m & -m).bit_length() - 1
            if i < j:
                final_edges.append([i, j])
            m &= m - 1
            
    return {'n': n, 'edges': final_edges}
