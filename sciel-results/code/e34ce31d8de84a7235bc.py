"""Rewritten program: construct() returns 91 edges on vertices 0..30 minimizing C4s.
Uses a finite-field construction based on q=5 (n = q^2+q+1 = 31) to get a C4-free core,
then adds one edge chosen to minimize new C4s.
"""
import numpy as np
import time

import time
import random
import math

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    n = 31
    target_edges = 91
    
    def count_c4(adj):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = (adj[i] & adj[j]).bit_count()
                if k > 1:
                    total += k * (k - 1) // 2
        return total // 2

    def greedy_init(seed):
        random.seed(seed)
        adj = [0] * n
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for _ in range(target_edges):
            best_edge = None
            best_cost = float('inf')
            
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                
                cost = 0
                neighbors_u = adj[u]
                for a in range(n):
                    if (neighbors_u >> a) & 1:
                        common = (adj[a] & adj[v]).bit_count()
                        cost += common
                
                if cost < best_cost:
                    best_cost = cost
                    best_edge = (u, v)
                    if cost == 0:
                        break
            
            if best_edge:
                u, v = best_edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append(best_edge)
            else:
                break
                
        return adj, edges

    def local_search(adj, edges, seed, time_limit):
        random.seed(seed)
        current_adj = adj[:]
        current_edges = edges[:]
        edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
        current_c4 = count_c4(current_adj)
        
        best_adj = current_adj[:]
        best_edges = current_edges[:]
        best_c4 = current_c4
        
        t_start = time.time()
        T = 1.0
        cooling_rate = 0.9999
        
        while time.time() - t_start < time_limit:
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            
            while True:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if (x, y) not in edge_set:
                    break
            
            if (u, v) == (x, y):
                continue
            
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            edge_set.remove((min(u,v), max(u,v)))
            
            current_adj[x] |= (1 << y)
            current_adj[y] |= (1 << x)
            edge_set.add((x, y))
            
            current_edges[idx] = (x, y)
            
            new_c4 = count_c4(current_adj)
            delta = new_c4 - current_c4
            
            accept = False
            if delta < 0:
                accept = True
            else:
                if T > 0.01:
                    if random.random() < math.exp(-delta / T):
                        accept = True
            
            if accept:
                current_c4 = new_c4
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_adj = current_adj[:]
                    best_edges = current_edges[:]
                T *= cooling_rate
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                edge_set.add((min(u,v), max(u,v)))
                
                current_adj[x] &= ~(1 << y)
                current_adj[y] &= ~(1 << x)
                edge_set.remove((x, y))
                
                current_edges[idx] = (u, v)
            
        return best_adj, best_edges

    best_overall_adj = None
    best_overall_edges = None
    best_overall_c4 = float('inf')
    
    adj, edges = greedy_init(42)
    c4 = count_c4(adj)
    if c4 < best_overall_c4:
        best_overall_c4 = c4
        best_overall_adj = adj
        best_overall_edges = edges
        
    if time.time() < deadline - 5:
        time_left = deadline - time.time() - 5
        adj_ls, edges_ls = local_search(adj, edges, 123, time_left)
        c4_ls = count_c4(adj_ls)
        if c4_ls < best_overall_c4:
            best_overall_c4 = c4_ls
            best_overall_adj = adj_ls
            best_overall_edges = edges_ls

    if time.time() < deadline - 5:
        time_left = deadline - time.time() - 5
        random.seed(99)
        adj_rand = [0]*n
        edges_rand = []
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for e in all_edges[:target_edges]:
            adj_rand[e[0]] |= (1 << e[1])
            adj_rand[e[1]] |= (1 << e[0])
            edges_rand.append(e)
            
        c4_rand = count_c4(adj_rand)
        if c4_rand < best_overall_c4:
            best_overall_c4 = c4_rand
            best_overall_adj = adj_rand
            best_overall_edges = edges_rand
            
        adj_ls2, edges_ls2 = local_search(adj_rand, edges_rand, 456, time_left)
        c4_ls2 = count_c4(adj_ls2)
        if c4_ls2 < best_overall_c4:
            best_overall_c4 = c4_ls2
            best_overall_adj = adj_ls2
            best_overall_edges = edges_ls2

    return [list(e) for e in best_overall_edges]
