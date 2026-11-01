import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 320
    
    def generate_random_k_regular(n, k):
        half_edges = []
        for i in range(n):
            half_edges.extend([i] * k)
        
        edges = []
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts:
            random.shuffle(half_edges)
            edges = []
            valid = True
            edge_set = set()
            
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in edge_set:
                    valid = False
                    break
                edge_set.add(edge)
                edges.append(edge)
            
            if valid:
                return edges, edge_set
            attempts += 1
            
        return None, None

    def build_adjacency(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def get_girth(adj, n, max_girth):
        min_girth = max_girth
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_girth // 2:
                    continue
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        if v != parent[curr]:
                            cycle_len = dist[curr] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                                if min_girth == 3:
                                    return 3
        return min_girth

    def find_short_cycle(adj, n, max_girth):
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_girth // 2:
                    continue
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        if v != parent[curr]:
                            cycle_len = dist[curr] + dist[v] + 1
                            if cycle_len < max_girth:
                                path1 = [curr]
                                while path1[-1] != u:
                                    path1.append(parent[path1[-1]])
                                path1.reverse()
                                
                                path2 = [v]
                                while path2[-1] != u:
                                    path2.append(parent[path2[-1]])
                                path2.reverse()
                                
                                cycle = path1[::-1] + path2[1:]
                                return cycle
        return None

    def perform_2_switch(edges, edge_set, adj, n, u, v, w, x, orientation):
        if orientation == 0:
            new_e1 = (u, w)
            new_e2 = (v, x)
        else:
            new_e1 = (u, x)
            new_e2 = (v, w)
            
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            return False, None, None
            
        e1_key = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
        e2_key = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
        
        rem_e1 = (min(u, v), max(u, v))
        rem_e2 = (min(w, x), max(w, x))
        
        if e1_key in edge_set and e1_key != rem_e1 and e1_key != rem_e2:
            return False, None, None
        if e2_key in edge_set and e2_key != rem_e1 and e2_key != rem_e2:
            return False, None, None
        if e1_key == e2_key:
            return False, None, None
            
        idx1 = -1
        idx2 = -1
        for i, e in enumerate(edges):
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key == rem_e1:
                idx1 = i
            elif key == rem_e2:
                idx2 = i
                
        if idx1 == -1 or idx2 == -1:
            return False, None, None
            
        edges[idx1] = (new_e1[0], new_e1[1])
        edges[idx2] = (new_e2[0], new_e2[1])
        
        edge_set.discard(rem_e1)
        edge_set.discard(rem_e2)
        edge_set.add(e1_key)
        edge_set.add(e2_key)
        
        adj[u].remove(v)
        adj[v].remove(u)
        adj[w].remove(x)
        adj[x].remove(w)
        
        adj[u].append(new_e1[1])
        adj[new_e1[1]].append(u)
        adj[v].append(new_e2[1])
        adj[new_e2[1]].append(v)
        
        return True, new_e1, new_e2

    best_girth = 0
    best_edges = None
    
    for trial in range(10):
        if time.time() > deadline:
            break
            
        edges, edge_set = generate_random_k_regular(n, k)
        if edges is None:
            continue
            
        adj = build_adjacency(edges, n)
        
        current_girth = get_girth(adj, n, g)
        
        if current_girth >= g:
            return n, edges
            
        steps = 0
        max_steps = 20000
        
        while steps < max_steps and time.time() < deadline:
            cycle = find_short_cycle(adj, n, g)
            if cycle is None:
                return n, edges
                
            L = len(cycle)
            if L >= g:
                return n, edges
                
            idx = random.randint(0, L-1)
            u = cycle[idx]
            v = cycle[(idx+1) % L]
            
            cycle_edges_set = set()
            for i in range(L):
                a = cycle[i]
                b = cycle[(i+1) % L]
                cycle_edges_set.add((min(a,b), max(a,b)))
                
            w, x = None, None
            for _ in range(50):
                if not edges:
                    break
                e_idx = random.randint(0, len(edges)-1)
                cand_u, cand_v = edges[e_idx]
                cand_key = (min(cand_u, cand_v), max(cand_u, cand_v))
                if cand_key not in cycle_edges_set:
                    w, x = cand_u, cand_v
                    break
                    
            if w is None:
                e_idx = random.randint(0, len(edges)-1)
                w, x = edges[e_idx]
                if (min(u,v), max(u,v)) == (min(w,x), max(w,x)):
                    continue

            orientation = random.randint(0, 1)
            
            success, new_e1, new_e2 = perform_2_switch(edges, edge_set, adj, n, u, v, w, x, orientation)
            
            if success:
                new_girth = get_girth(adj, n, g)
                
                if new_girth >= g:
                    return n, edges
                    
                if new_girth > current_girth:
                    current_girth = new_girth
                elif new_girth == current_girth:
                    if random.random() < 0.5:
                        current_girth = new_girth
                    else:
                        # Revert
                        perform_2_switch(edges, edge_set, adj, n, u, new_e1[1], v, new_e2[1], orientation)
                else:
                    # Revert
                    perform_2_switch(edges, edge_set, adj, n, u, new_e1[1], v, new_e2[1], orientation)
                
            steps += 1
            
    return n, edges
