import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 20
    best_edges = []
    best_score = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Phase 1: Greedy
    for seed in range(10000):
        if time.time() > deadline - 20.0:
            break
        
        if seed % 2 == 0:
            edges = _greedy_random_numba(n, all_edges, seed) # Just standard
        else:
            edges = _greedy_random(n, all_edges, seed)
            
        score = len(edges)
        if score > best_score:
            best_score = score
            best_edges = edges
            
    # Phase 2: Local Search
    if time.time() < deadline - 5.0 and best_edges:
        improved = _local_search_numpy(n, best_edges, all_edges, deadline - 3.0)
        if len(improved) > best_score:
            best_edges = improved
            best_score = len(improved)
            
    return best_edges

def _greedy_random(n, all_edges, seed):
    random.seed(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    shuffled = all_edges[:]
    random.shuffle(shuffled)
    for u, v in shuffled:
        # Check K2,3
        if adj[u, v] == 1: continue
        # Common neighbors of u, v
        common = (adj[u] & adj[v]).sum()
        if common >= 3: continue
        
        # Check pairs (u, w) and (v, w)
        valid = True
        for w in range(n):
            if w == u or w == v: continue
            # Common neighbors of u, w after adding (u,v)
            # adj[u] has v? No, we are checking before adding.
            # If we add (u,v), v becomes neighbor of u.
            # So common neighbors of u, w = (adj[u] & adj[w]) + (1 if adj[w,v]==1 else 0)
            cuw = (adj[u] & adj[w]).sum()
            if adj[w, v] == 1:
                cuw += 1
            if cuw >= 3:
                valid = False
                break
            cvw = (adj[v] & adj[w]).sum()
            if adj[w, u] == 1:
                cvw += 1
            if cvw >= 3:
                valid = False
                break
        if valid:
            adj[u, v] = 1
            adj[v, u] = 1
            edges.append([u, v])
    return edges

def _local_search_numpy(n, edges, all_edges, deadline):
    adj = np.zeros((n, n), dtype=np.int8)
    current_edges = set()
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
        current_edges.add((min(u, v), max(u, v)))
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any edge
        for u, v in all_edges:
            if time.time() > deadline: break
            if (min(u, v), max(u, v)) in current_edges: continue
            if _can_add_numpy(adj, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
                current_edges.add((min(u, v), max(u, v)))
                improved = True
                break # Restart loop
                
        if improved: continue
        
        # Try 2-swap
        edge_list = list(current_edges)
        random.shuffle(edge_list)
        for e in edge_list:
            if time.time() > deadline: break
            u1, v1 = e
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            current_edges.remove(e)
            
            candidates = []
            for u2, v2 in all_edges:
                if (min(u2, v2), max(u2, v2)) in current_edges: continue
                if _can_add_numpy(adj, u2, v2):
                    candidates.append((u2, v2))
            
            if len(candidates) >= 2:
                random.shuffle(candidates)
                found = False
                for i in range(min(len(candidates), 50)):
                    u2, v2 = candidates[i]
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    current_edges.add((min(u2, v2), max(u2, v2)))
                    
                    for j in range(i+1, min(len(candidates), 100)):
                        u3, v3 = candidates[j]
                        if (min(u3, v3), max(u3, v3)) in current_edges: continue
                        if _can_add_numpy(adj, u3, v3):
                            adj[u3, v3] = 1
                            adj[v3, u3] = 1
                            current_edges.add((min(u3, v3), max(u3, v3)))
                            improved = True
                            found = True
                            break
                    if found: break
                    if not found:
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0
                        current_edges.remove((min(u2, v2), max(u2, v2)))
                
                if not found:
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    current_edges.add(e)
            else:
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                current_edges.add(e)
                
    return [list(e) for e in current_edges]

def _can_add_numpy(adj, u, v):
    if adj[u, v] == 1: return False
    if (adj[u] & adj[v]).sum() >= 3: return False
    for w in range(len(adj)):
        if w == u or w == v: continue
        cuw = (adj[u] & adj[w]).sum()
        if adj[w, v] == 1: cuw += 1
        if cuw >= 3: return False
        cvw = (adj[v] & adj[w]).sum()
        if adj[w, u] == 1: cvw += 1
        if cvw >= 3: return False
    return True
