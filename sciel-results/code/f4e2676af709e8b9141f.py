import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Generate Paley 37 graph
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    def get_edges():
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return edges

    def find_k4s_and_i4s():
        k4s = []
        i4s = []
        edge_mat = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in adj[i]:
                edge_mat[i][j] = True
                edge_mat[j][i] = True
        
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            if edge_mat[a][b] and edge_mat[a][c] and edge_mat[a][d] and \
               edge_mat[b][c] and edge_mat[b][d] and edge_mat[c][d]:
                k4s.append(combo)
            elif not edge_mat[a][b] and not edge_mat[a][c] and not edge_mat[a][d] and \
                 not edge_mat[b][c] and not edge_mat[b][d] and not edge_mat[c][d]:
                i4s.append(combo)
        
        return k4s, i4s

    current_k4s, current_i4s = find_k4s_and_i4s()
    
    best_n = n
    best_edges = get_edges()
    
    while time.time() < deadline:
        target_n = n + 1
        found_S = False
        
        # Pre-compute K4s by vertex for fast checking
        k4_by_vertex = [set() for _ in range(n)]
        for k in current_k4s:
            for v in k:
                k4_by_vertex[v].add(k)
        
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            S = set()
            possible = True
            
            # Greedy loop
            while True:
                # Find unhit I4s
                unhit_i4s = []
                for i4 in current_i4s:
                    if not (set(i4) & S):
                        unhit_i4s.append(i4)
                
                if not unhit_i4s:
                    break
                
                i4 = random.choice(unhit_i4s)
                
                candidates = []
                for v in i4:
                    if v in S:
                        continue
                    bad = False
                    for k in k4_by_vertex[v]:
                        if k.issubset(S | {v}):
                            bad = True
                            break
                    if not bad:
                        candidates.append(v)
                
                if not candidates:
                    possible = False
                    break
                
                v = random.choice(candidates)
                S.add(v)
            
            if possible:
                valid_S = True
                for k in current_k4s:
                    if k.issubset(S):
                        valid_S = False
                        break
                
                if valid_S:
                    adj.append(set(S))
                    for v in S:
                        adj[v].add(n)
                    n += 1
                    found_S = True
                    break
        
        if not found_S:
            break
        
        current_k4s, current_i4s = find_k4s_and_i4s()
        best_n = n
        best_edges = get_edges()

    return best_n, best_edges
