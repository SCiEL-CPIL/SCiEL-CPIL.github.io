import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 55.0
    
    N = 256
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Find bad nodes where 2*a is a square
    bad_nodes = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            bad_nodes.add(a)
            
    neighbors = [[] for _ in range(N + 1)]
    for v in range(1, N + 1):
        for s in squares:
            u = s - v
            if 1 <= u <= N:
                neighbors[v].append(u)
                
    in_set = [False] * (N + 1)
    conf_counts = [0] * (N + 1)
    
    def add_node(v):
        in_set[v] = True
        for u in neighbors[v]:
            conf_counts[u] += 1
            
    def remove_node(v):
        in_set[v] = False
        for u in neighbors[v]:
            conf_counts[u] -= 1
            
    def get_size():
        return sum(in_set[1:])
        
    def greedy_fill():
        added = True
        while added:
            added = False
            valid_nodes = [v for v in range(1, N + 1) 
                           if not in_set[v] and v not in bad_nodes and conf_counts[v] == 0]
            if valid_nodes:
                v = random.choice(valid_nodes)
                add_node(v)
                added = True

    best_size = 0
    best_set = [False] * (N + 1)
    
    for _ in range(100):
        in_set = [False] * (N + 1)
        conf_counts = [0] * (N + 1)
        greedy_fill()
        size = get_size()
        if size > best_size:
            best_size = size
            best_set = in_set.copy()
            
        if time.time() - start_time > deadline:
            break

    cur_set = best_set.copy()
    cur_conf = [0] * (N + 1)
    for v in range(1, N + 1):
        if cur_set[v]:
            for u in neighbors[v]:
                cur_conf[u] += 1
                
    cur_size = best_size
    in_set = cur_set
    conf_counts = cur_conf
    
    t = 1.0
    t_min = 0.01
    cooling = 0.995
    
    while time.time() - start_time < deadline:
        if cur_size > 0:
            candidates = [v for v in range(1, N + 1) if in_set[v]]
            if candidates:
                v_to_remove = random.choice(candidates)
                
                old_in_set = in_set.copy()
                old_conf = conf_counts.copy()
                
                remove_node(v_to_remove)
                
                greedy_fill()
                
                new_size = get_size()
                
                if new_size > cur_size:
                    cur_size = new_size
                    if cur_size > best_size:
                        best_size = cur_size
                        best_set = in_set.copy()
                else:
                    in_set = old_in_set
                    conf_counts = old_conf
                    
        t *= cooling
        if t < t_min:
            t = t_min

    result = [v for v in range(1, N + 1) if best_set[v]]
    if not result:
        return [1]
    return result
