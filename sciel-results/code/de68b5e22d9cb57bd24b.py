import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140
    n = 20
    
    best_edges = None
    best_count = 0
    
    def greedy_construct(seed):
        random.seed(seed)
        adj_color = [[-1]*n for _ in range(n)]  # adj_color[u][v] = colour or -1
        adj_list = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > DEADLINE:
                break
            added = False
            for c in range(n):
                # Check if adding (u,v,c) violates any 4-cycle
                valid = True
                for w in adj_list[v]:
                    if w == u:
                        continue
                    for x in adj_list[u]:
                        if x == w or x == v:
                            continue
                        if x in adj_list[w]:
                            # 4-cycle u-v-w-x-u
                            colors = {c, adj_color[v][w], adj_color[w][x], adj_color[x][u]}
                            if len(colors) < 4:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    adj_color[u][v] = c
                    adj_color[v][u] = c
                    adj_list[u].add(v)
                    adj_list[v].add(u)
                    edges.append([u, v, c])
                    added = True
                    break
        return edges
    
    def local_improve(edges):
        adj_color = [[-1]*n for _ in range(n)]
        adj_list = [set() for _ in range(n)]
        for u, v, c in edges:
            adj_color[u][v] = c
            adj_color[v][u] = c
            adj_list[u].add(v)
            adj_list[v].add(u)
        
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj_color[u][v] != -1:
                        continue
                    for c in range(n):
                        valid = True
                        for w in adj_list[v]:
                            if w == u:
                                continue
                            for x in adj_list[u]:
                                if x == w or x == v:
                                    continue
                                if x in adj_list[w]:
                                    colors = {c, adj_color[v][w], adj_color[w][x], adj_color[x][u]}
                                    if len(colors) < 4:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if valid:
                            adj_color[u][v] = c
                            adj_color[v][u] = c
                            adj_list[u].add(v)
                            adj_list[v].add(u)
                            edges.append([u, v, c])
                            improved = True
                            break
        return edges
    
    # Multiple restarts
    for i in range(50):
        if time.time() > DEADLINE - 5:
            break
        edges = greedy_construct(i)
        edges = local_improve(edges)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    if best_edges is None:
        return [[0, v, 0] for v in range(1, n)]
    
    return best_edges
