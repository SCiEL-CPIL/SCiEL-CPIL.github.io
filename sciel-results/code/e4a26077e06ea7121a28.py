import random
import time
import math

def construct():
    start_time = time.time()
    max_time = 55.0
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    n = len(moduli)
    
    # Build non-coprime graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(moduli[i], moduli[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        if time.time() > start_time + max_time:
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        U = P.union(X)
        if not U:
            pivot = -1
        else:
            max_count = -1
            pivot = -1
            for u in U:
                count = 0
                for v in P:
                    if v in adj[u]:
                        count += 1
                if count > max_count:
                    max_count = count
                    pivot = u
            
        if pivot != -1:
            candidates = [v for v in P if v not in adj[pivot]]
        else:
            candidates = list(P)
            
        for v in candidates:
            new_R = R + [v]
            new_P = P.intersection(adj[v])
            new_X = X.intersection(adj[v])
            
            bron_kerbosch(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)
            
            if time.time() > start_time + max_time:
                return

    initial_P = set(range(n))
    bron_kerbosch([], initial_P, set())
    
    clique_moduli = [moduli[i] for i in best_clique]
    
    residues_for = {m: list(range(m)) for m in clique_moduli}
    
    def is_compatible(r, m, selected):
        for r_j, m_j in selected:
            g = math.gcd(m, m_j)
            if r % g == r_j % g:
                return False
        return True
        
    def find_assignment(moduli_list):
        selected = []
        order = moduli_list[:]
        order.sort()
        
        for m in order:
            res_list = residues_for[m][:]
            random.shuffle(res_list)
            found = False
            for r in res_list:
                if is_compatible(r, m, selected):
                    selected.append((r, m))
                    found = True
                    break
        return selected

    result = find_assignment(clique_moduli)
    
    if not result:
        result = [(0, 2)]
        
    return [[r, m] for r, m in result]
