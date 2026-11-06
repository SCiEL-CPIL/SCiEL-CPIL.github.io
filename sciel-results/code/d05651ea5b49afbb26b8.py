import numpy as np
import time
import random
import math

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # 1. Construct the base graph: Polarity Graph of PG(2,5)
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    adj_mask = [0] * n
    edge_set = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edge_set.add((i, j))
    
    current_edges = len(edge_set)
    
    def count_c4_add(u, v):
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        count = 0
        temp = nu
        while temp:
            w_bit = temp & (-temp)
            w = w_bit.bit_length() - 1
            temp ^= w_bit
            count += (adj_mask[w] & nv).bit_count()
        return count
    
    def count_c4_remove(u, v):
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        count = 0
        temp = nu
        while temp:
            w_bit = temp & (-temp)
            w = w_bit.bit_length() - 1
            temp ^= w_bit
            count += (adj_mask[w] & nv).bit_count()
        return count
    
    def get_delta(u, v, x, y):
        lost = count_c4_remove(u, v)
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        gained = count_c4_add(x, y)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        return gained - lost

    # 2. Greedy addition of remaining edges
    edges_to_add = m - current_edges
    all_non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
    
    for _ in range(edges_to_add):
        if time.time() > deadline:
            break
        best_edge = None
        min_c4s = float('inf')
        for u, v in all_non_edges:
            c4s = count_c4_add(u, v)
            if c4s < min_c4s:
                min_c4s = c4s
                best_edge = (u, v)
        if best_edge:
            u, v = best_edge
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edge_set.add(best_edge)
            all_non_edges.remove(best_edge)

    def count_total_c4s():
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (1 << j) & adj_mask[i]:
                    common = adj_mask[i] & adj_mask[j]
                    common &= ~(1 << i) & ~(1 << j)
                    c = common.bit_count()
                    total += c * (c - 1) // 2
        return total

    current_c4 = count_total_c4s()
    best_c4 = current_c4
    best_adj_mask = adj_mask[:]
    best_edge_set = set(edge_set)
    
    current_edges_list = list(edge_set)
    
    # 3. Restarting Simulated Annealing
    # Run multiple SA cycles to explore the space more thoroughly
    while time.time() < deadline - 10:
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.99999  # Slower cooling to allow more steps
        
        steps = 0
        while T > T_min and time.time() < deadline - 10:
            steps += 1
            
            idx_rm = random.randint(0, len(current_edges_list) - 1)
            e_rm = current_edges_list[idx_rm]
            u, v = e_rm
            
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if (x, y) not in edge_set:
                    break
            
            delta = get_delta(u, v, x, y)
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                edge_set.remove(e_rm)
                current_edges_list[idx_rm] = (x, y)
                
                adj_mask[x] |= (1 << y)
                adj_mask[y] |= (1 << x)
                edge_set.add((x, y))
                
                current_c4 += delta
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj_mask = adj_mask[:]
                    best_edge_set = set(edge_set)
                    
            T *= cooling_rate

    # 4. Final polish: deterministic local search on best solution
    if time.time() < deadline - 3:
        adj_mask = best_adj_mask[:]
        edge_set = set(best_edge_set)
        current_c4 = best_c4
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            edges_list = list(edge_set)
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
            
            for e_rm in edges_list:
                if time.time() > deadline - 3:
                    break
                u, v = e_rm
                for x, y in non_edges:
                    if (x, y) == (u, v):
                        continue
                    delta = get_delta(u, v, x, y)
                    if delta < 0:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        edge_set.remove(e_rm)
                        adj_mask[x] |= (1 << y)
                        adj_mask[y] |= (1 << x)
                        edge_set.add((x, y))
                        current_c4 += delta
                        improved = True
                        if current_c4 < best_c4:
                            best_c4 = current_c4
                            best_adj_mask = adj_mask[:]
                            best_edge_set = set(edge_set)
                        break
    
    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
