import time
import random
from collections import deque
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    best_n = None
    best_edges = None

    def verify_girth(n, edges, min_girth):
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def generate_random_regular(n, k, seed=None):
        if seed is not None:
            random.seed(seed)
        if n * k % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        edge_set = set()
        valid = True
        for i in range(n * k // 2):
            u, v = stubs[2 * i], stubs[2 * i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        if not valid:
            return None
        edges = list(edge_set)
        return n, edges

    def local_search_girth(n, edges, target_girth, max_time):
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u, v), max(u, v)))
        
        def get_girth():
            min_len = target_girth
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                q = deque([start])
                dist[start] = 0
                while q:
                    u = q.popleft()
                    if dist[u] >= target_girth // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_len:
                                min_len = cycle_len
                                if min_len == 3:
                                    return min_len
            return min_len

        current_girth = get_girth()
        
        if current_girth >= target_girth:
            return n, edges

        temperature = 10.0
        cooling_rate = 0.995
        max_iter = 2000
        
        for i in range(max_iter):
            if time.time() > max_time:
                break
            if current_girth >= target_girth:
                return n, edges
            
            u1 = random.randint(0, n - 1)
            if not adj[u1]: continue
            v1 = random.choice(adj[u1])
            
            u2 = random.randint(0, n - 1)
            if not adj[u2]: continue
            v2 = random.choice(adj[u2])
            
            if len({u1, v1, u2, v2}) < 4:
                continue
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
                
            key1 = (min(u1, u2), max(u1, u2))
            key2 = (min(v1, v2), max(v1, v2))
            
            if key1 in edge_set or key2 in edge_set:
                continue
                
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            edge_set.remove((min(u1, v1), max(u1, v1)))
            edge_set.remove((min(u2, v2), max(u2, v2)))
            
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            
            edge_set.add(key1)
            edge_set.add(key2)
            
            new_girth = get_girth()
            
            if new_girth <= current_girth:
                current_girth = new_girth
            else:
                adj[u1].remove(u2)
                adj[u2].remove(u1)
                adj[v1].remove(v2)
                adj[v2].remove(v1)
                
                edge_set.remove(key1)
                edge_set.remove(key2)
                
                adj[u1].append(v1)
                adj[v1].append(u1)
                adj[u2].append(v2)
                adj[v2].append(u2)
                
                edge_set.add((min(u1, v1), max(u1, v1)))
                edge_set.add((min(u2, v2), max(u2, v2)))
                
                delta = new_girth - current_girth
                if random.random() < math.exp(-delta / temperature):
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                    edge_set.remove((min(u1, v1), max(u1, v1)))
                    edge_set.remove((min(u2, v2), max(u2, v2)))
                    
                    adj[u1].append(u2)
                    adj[u2].append(u1)
                    adj[v1].append(v2)
                    adj[v2].append(v1)
                    
                    edge_set.add(key1)
                    edge_set.add(key2)
                    current_girth = new_girth
            
            temperature *= cooling_rate
        
        if current_girth >= target_girth:
            final_edges = list(edge_set)
            return n, final_edges
        return None, None

    start_n = max(moore, 500)
    upper_n = 1500
    
    for n_try in range(start_n, upper_n + 1, 50):
        if time_left() < 15:
            break
            
        for seed in range(5):
            if time_left() < 10:
                break
            
            result = generate_random_regular(n_try, k, seed=seed)
            if result is None:
                continue
            n_r, edges_r = result
            
            ls_result = local_search_girth(n_r, edges_r, g, time.time() + 10)
            if ls_result[0] is not None:
                n_ls, edges_ls = ls_result
                return n_ls, edges_ls
                
    if best_n is None:
        n_fallback = 2000
        result = generate_random_regular(n_fallback, k, seed=12345)
        if result:
            n_r, edges_r = result
            ls_result = local_search_girth(n_r, edges_r, g, time.time() + 10)
            if ls_result[0]:
                return ls_result
            return n_r, edges_r

    return best_n, best_edges
