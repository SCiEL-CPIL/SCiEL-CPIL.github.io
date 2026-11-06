import time
import random
from itertools import combinations

def _count_k4(adj, n):
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                c_list = []
                c = common
                while c:
                    lsb = c & -c
                    idx = lsb.bit_length() - 1
                    c_list.append(idx)
                    c ^= lsb
                for i in range(len(c_list)):
                    w = c_list[i]
                    for j in range(i + 1, len(c_list)):
                        x = c_list[j]
                        if (adj[w] >> x) & 1:
                            count += 1
    return count

def _count_indep6(adj, n):
    count = 0
    # Use 5-subsets
    for S5 in combinations(range(n), 5):
        mask = 0
        for v in S5:
            mask |= (1 << v)
        # Check independence
        is_indep = True
        for v in S5:
            if adj[v] & mask:
                is_indep = False
                break
        if not is_indep:
            continue
        
        # Check for 6th vertex
        for w in range(n):
            if w in S5: continue
            if (adj[w] & mask) == 0:
                count += 1
    return count // 6

def _violations(adj, n):
    return _count_k4(adj, n) + _count_indep6(adj, n)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    best_adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
    
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
        found_valid = False
        for r in range(20): # 20 restarts
            if time.time() > deadline:
                break
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            current_v = _violations(adj, n)
            if current_v == 0:
                best_n = n
                best_adj = adj
                found_valid = True
                break
            
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Try 5 random moves
                best_delta = 0
                best_edge = None
                for _ in range(5):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v: continue
                    old_state = (adj[u] >> v) & 1
                    if old_state:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    
                    new_v = _violations(adj, n)
                    delta = new_v - current_v
                    if delta < best_delta:
                        best_delta = delta
                        best_edge = (u, v, old_state)
                    
                    if old_state:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                
                if best_edge:
                    u, v, old_state = best_edge
                    if old_state:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    current_v += best_delta
                else:
                    # Perturb
                    for _ in range(3):
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u != v:
                            if (adj[u] >> v) & 1:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                            else:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                    current_v = _violations(adj, n)
                
                if current_v == 0:
                    best_n = n
                    best_adj = adj
                    found_valid = True
                    break
            
            if found_valid:
                break
        if found_valid:
            break

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': best_n, 'edges': edges}
