"""Construct a large (5,5)-free graph by growing from Paley graph with direct constraint checking."""

import time
import random

def _paley_graph(q):
    """Return adjacency as list of sets for Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [set() for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _check_k5_with_new_vertex(n, adj, S):
    """
    Check if adding new vertex with neighborhood S creates a K5.
    New vertex x connects to S. K5 exists if there's a K4 in S.
    """
    S_list = list(S)
    m = len(S_list)
    if m < 4:
        return False
    
    # Check all 4-subsets of S for being a K4
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            if b not in adj[a]:
                continue
            for k in range(j + 1, m):
                c = S_list[k]
                if c not in adj[a] or c not in adj[b]:
                    continue
                for l in range(k + 1, m):
                    d = S_list[l]
                    if d not in adj[a] or d not in adj[b] or d not in adj[c]:
                        continue
                    if d in adj[a] and d in adj[b] and d in adj[c]:
                        return True
    return False

def _check_indep5_with_new_vertex(n, adj, S):
    """
    Check if adding new vertex with neighborhood S creates an independent 5-set.
    New vertex x is non-adjacent to T = V - S - {x}. Independent 5 exists if T contains independent 4-set.
    """
    T = set(range(n)) - S
    T_list = list(T)
    m = len(T_list)
    if m < 4:
        return False
    
    # Check all 4-subsets of T for being independent
    for i in range(m):
        a = T_list[i]
        for j in range(i + 1, m):
            b = T_list[j]
            if b in adj[a]:
                continue
            for k in range(j + 1, m):
                c = T_list[k]
                if c in adj[a] or c in adj[b]:
                    continue
                for l in range(k + 1, m):
                    d = T_list[l]
                    if d in adj[a] or d in adj[b] or d in adj[c]:
                        continue
                    if d not in adj[a] and d not in adj[b] and d not in adj[c]:
                        return True
    return False

def _is_valid_assignment(n, adj, S):
    """Check if new vertex with neighborhood S keeps graph (5,5)-free."""
    if _check_k5_with_new_vertex(n, adj, S):
        return False
    if _check_indep5_with_new_vertex(n, adj, S):
        return False
    return True

def _try_add_vertex(n, adj, max_attempts=500):
    """Try to add a new vertex with a valid neighborhood."""
    for attempt in range(max_attempts):
        S = set()
        for v in range(n):
            if random.random() < 0.5:
                S.add(v)
        
        if _is_valid_assignment(n, adj, S):
            return S
    return None

def _flip_edge(n, adj, i, j):
    """Flip edge between i and j."""
    if j in adj[i]:
        adj[i].discard(j)
        adj[j].discard(i)
    else:
        adj[i].add(j)
        adj[j].add(i)

def construct():
    deadline = time.time() + 290  # 40s safety margin
    start_time = time.time()
    
    # Start with Paley graph on 37 vertices
    q = 37
    adj = _paley_graph(q)
    n = q
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to grow the graph by adding vertices
    while time.time() < deadline:
        if n >= 42:
            break
        
        # Try to add new vertex directly
        S = _try_add_vertex(n, adj, max_attempts=200)
        
        if S is not None:
            adj.append(set(S))
            for v in S:
                adj[v].add(n)
            n += 1
            best_n = n
            best_adj = [set(s) for s in adj]
            continue
        
        # Direct addition failed. Try flipping edges to create space
        found_growth = False
        for flip_attempt in range(50):
            if time.time() > deadline:
                break
            
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            _flip_edge(n, adj, i, j)
            
            S_new = _try_add_vertex(n, adj, max_attempts=50)
            
            if S_new is not None:
                adj.append(set(S_new))
                for v in S_new:
                    adj[v].add(n)
                n += 1
                best_n = n
                best_adj = [set(s) for s in adj]
                found_growth = True
                break
            else:
                _flip_edge(n, adj, i, j)
        
        if not found_growth:
            # Try a few more random restarts
            for seed in range(5):
                if time.time() > deadline:
                    break
                random.seed(seed)
                S = _try_add_vertex(n, adj, max_attempts=100)
                if S is not None:
                    adj.append(set(S))
                    for v in S:
                        adj[v].add(n)
                    n += 1
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    found_growth = True
                    break
            
            if not found_growth:
                break
    
    edges = [(i, j) for i in range(best_n) for j in range(i + 1, best_n) if j in best_adj[i]]
    return best_n, edges
