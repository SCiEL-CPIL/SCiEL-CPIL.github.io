import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
    
    k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in k4s:
        a, b, c, d = quad
        k4_edges.append([
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ])
    
    def count_violations(colors):
        viol = 0
        for e in k4_edges:
            if len({colors[e[0]], colors[e[1]], colors[e[2]],
                    colors[e[3]], colors[e[4]], colors[e[5]]}) < 5:
                viol += 1
        return viol
    
    def is_valid(colors):
        return count_violations(colors) == 0
    
    def get_17_sol():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    def local_search(C, time_limit):
        best_colors = None
        best_viol = float('inf')
        
        for init_type in ['z', 'rand', 'struct']:
            if time.time() > time_limit:
                break
            
            if init_type == 'z':
                cur = [(i + j) % C for i in range(16) for j in range(i + 1, 16)]
            elif init_type == 'struct':
                cur = [(i * 5 + j) % C for i in range(16) for j in range(i + 1, 16)]
            else:
                cur = [random.randint(0, C - 1) for _ in range(120)]
            
            cur_viol = count_violations(cur)
            if cur_viol == 0:
                return cur[:]
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_colors = cur[:]
            
            temp = 1.0
            min_temp = 0.01
            cooling = 0.9997
            max_iters = 5000
            no_improve = 0
            
            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                
                # Find violating K4
                viol_idx = -1
                for _ in range(20):
                    idx = random.randint(0, len(k4s) - 1)
                    e = k4_edges[idx]
                    if len({cur[e[0]], cur[e[1]], cur[e[2]],
                            cur[e[3]], cur[e[4]], cur[e[5]]}) < 5:
                        viol_idx = idx
                        break
                
                if viol_idx == -1:
                    cur_viol = count_violations(cur)
                    if cur_viol == 0:
                        return cur[:]
                    if cur_viol < best_viol:
                        best_viol = cur_viol
                        best_colors = cur[:]
                    no_improve += 5
                    continue
                
                e = k4_edges[viol_idx]
                
                # Try moves
                best_move = None
                best_new_viol = cur_viol + 1  # start higher
                
                for pos in range(6):
                    edge_idx = e[pos]
                    old_c = cur[edge_idx]
                    
                    for _ in range(4):
                        new_c = random.randint(0, C - 1)
                        if new_c == old_c:
                            new_c = (new_c + 1) % C
                        
                        cur[edge_idx] = new_c
                        nv = count_violations(cur)
                        
                        if nv < best_new_viol:
                            best_new_viol = nv
                            best_move = (edge_idx, new_c)
                        
                        cur[edge_idx] = old_c
                
                if best_move is None:
                    # Random move
                    pos = random.randint(0, 5)
                    edge_idx = e[pos]
                    new_c = random.randint(0, C - 1)
                    if new_c == cur[edge_idx]:
                        new_c = (new_c + 1) % C
                    best_move = (edge_idx, new_c)
                    best_new_viol = cur_viol
                
                edge_idx, new_c = best_move
                old_c = cur[edge_idx]
                cur[edge_idx] = new_c
                new_viol = count_violations(cur)
                
                if new_viol <= cur_viol:
                    cur_viol = new_viol
                    no_improve = 0
                    if cur_viol == 0:
                        return cur[:]
                    if cur_viol < best_viol:
                        best_viol = cur_viol
                        best_colors = cur[:]
                else:
                    delta = new_viol - cur_viol
                    prob = np.exp(-delta / temp) if temp > 0.001 else 0
                    if random.random() < prob:
                        cur_viol = new_viol
                        no_improve = 0
                    else:
                        cur[edge_idx] = old_c
                        no_improve += 1
                
                temp = max(min_temp, temp * cooling)
                
                if no_improve > 400:
                    break
        
        return best_colors
    
    for C in [14, 15, 16]:
        if time.time() > deadline - 3:
            break
        
        if C == 14:
            budget = min(40, deadline - time.time() - 3)
        elif C == 15:
            budget = min(40, deadline - time.time() - 3)
        else:
            budget = min(25, deadline - time.time() - 3)
        
        if budget < 5:
            continue
        
        end_time = time.time() + budget
        
        while time.time() < end_time:
            sol = local_search(C, min(end_time, time.time() + 12))
            if sol is not None and is_valid(sol):
                return sol
    
    return get_17_sol()
