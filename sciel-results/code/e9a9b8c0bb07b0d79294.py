import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def time_left():
        return deadline - time.time()

    def is_triangle_free(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True

    def max_independent_set_size(n, adj):
        masks = [0] * n
        for i in range(n):
            m = 0
            for j in adj[i]:
                m |= (1 << j)
            masks[i] = m
        
        best = 0
        def dfs(candidates, current_size):
            nonlocal best
            if time_left() < 0.05:
                return
            if current_size + candidates.bit_count() <= best:
                return
            if candidates == 0:
                best = max(best, current_size)
                return
            v = (candidates & -candidates).bit_length() - 1
            candidates_without_v = candidates & ~(1 << v)
            dfs(candidates_without_v & ~masks[v], current_size + 1)
            dfs(candidates_without_v, current_size)
        dfs((1 << n) - 1, 0)
        return best

    for target_n in range(39, 29, -1):
        if time_left() < 10:
            break
            
        n = target_n
        edges = []
        adj = [set() for _ in range(n)]
        
        # Initialize
        for _ in range(500):
            if time_left() < 10:
                break
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or (u, v) in edges or (v, u) in edges:
                continue
            creates_triangle = False
            for w in adj[u]:
                if w in adj[v]:
                    creates_triangle = True
                    break
            if not creates_triangle:
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                
        T = 1.0
        cooling = 0.999
        current_alpha = max_independent_set_size(n, adj)
        
        for step in range(20000):
            if time_left() < 5:
                break
            if current_alpha < 10:
                if is_triangle_free(n, edges):
                    return {'n': n, 'edges': edges}
            
            if random.random() < 0.5:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or (u, v) in edges or (v, u) in edges:
                    continue
                creates_triangle = False
                for w in adj[u]:
                    if w in adj[v]:
                        creates_triangle = True
                        break
                if creates_triangle:
                    continue
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                new_alpha = max_independent_set_size(n, adj)
                delta = new_alpha - current_alpha
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_alpha = new_alpha
                else:
                    edges.pop()
                    adj[u].remove(v)
                    adj[v].remove(u)
            else:
                if not edges:
                    continue
                idx = random.randint(0, len(edges)-1)
                u, v = edges[idx]
                edges.pop(idx)
                adj[u].remove(v)
                adj[v].remove(u)
                new_alpha = max_independent_set_size(n, adj)
                delta = new_alpha - current_alpha
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_alpha = new_alpha
                else:
                    edges.insert(idx, [u, v])
                    adj[u].add(v)
                    adj[v].add(u)
            
            T *= cooling
            
    # Fallback
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}

