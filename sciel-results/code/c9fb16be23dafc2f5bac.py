"""Construct a K4-free graph with no independent 6-set, maximizing n."""
import random
import time

def _find_k4(n, adj):
    """Find a K4 in the graph. Returns (i,j,k,l) or None."""
    for i in range(n):
        ni = adj[i]
        for j in range(i+1, n):
            if j not in ni:
                continue
            for k in range(j+1, n):
                if k not in ni or k not in adj[j]:
                    continue
                common = ni & adj[j] & adj[k]
                common.discard(i)
                common.discard(j)
                common.discard(k)
                if common:
                    l = min(common)
                    return (i, j, k, l)
    return None

def _find_indep6(n, adj):
    """Find an independent set of size 6. Returns a set of 6 vertices or None.
    Uses backtracking with pruning. May be slow for large n."""
    # Order vertices by degree (ascending)
    order = sorted(range(n), key=lambda v: len(adj[v]))
    
    def search(chosen, candidates):
        if len(chosen) == 6:
            return chosen
        if not candidates:
            return None
        if len(chosen) + len(candidates) < 6:
            return None
        
        # Pick the vertex with minimum degree in the induced subgraph
        # Use the order to pick a good vertex
        v = min(candidates, key=lambda x: order.index(x) if x in order else 0)
        
        # Option 1: include v
        new_candidates = candidates - {v}
        new_candidates -= adj[v]
        result = search(chosen | {v}, new_candidates)
        if result is not None:
            return result
        
        # Option 2: exclude v
        result = search(chosen, candidates - {v})
        if result is not None:
            return result
        
        return None
    
    return search(set(), set(range(n)))

def _is_valid(n, adj):
    """Check if graph is valid (no K4, no independent 6-set)."""
    if _find_k4(n, adj) is not None:
        return False
    if _find_indep6(n, adj) is not None:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 125
    
    best_n = 0
    best_adj = None
    
    for target_n in range(35, 1, -1):
        if time.time() > deadline - 20:
            break
        
        found_valid = False
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            p = 0.55 + random.random() * 0.15
            adj = [set() for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Repair phase with time limit
            for repair_iter in range(1000):
                if time.time() > deadline - 5:
                    break
                
                k4 = _find_k4(target_n, adj)
                if k4 is not None:
                    i, j, k, l = k4
                    edges_to_remove = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                    edge = random.choice(edges_to_remove)
                    adj[edge[0]].discard(edge[1])
                    adj[edge[1]].discard(edge[0])
                    continue
                
                # Use a faster check for independent 6-set during repair
                # Just use randomized greedy
                indep6 = None
                for trial in range(50):
                    order = list(range(target_n))
                    random.shuffle(order)
                    indep = set()
                    for v in order:
                        if not (adj[v] & indep):
                            indep.add(v)
                            if len(indep) >= 6:
                                indep6 = set(list(indep)[:6])
                                break
                    if indep6:
                        break
                
                if indep6 is not None:
                    verts = list(indep6)
                    added = False
                    for a in range(len(verts)):
                        for b in range(a+1, len(verts)):
                            u, v = verts[a], verts[b]
                            if v not in adj[u]:
                                adj[u].add(v)
                                adj[v].add(u)
                                added = True
                                break
                        if added:
                            break
                    if not added:
                        break
                    continue
                
                # Final validation with thorough check
                if _is_valid(target_n, adj):
                    if target_n > best_n:
                        best_n = target_n
                        best_adj = [set(s) for s in adj]
                    found_valid = True
                    break
            
            if found_valid:
                break
        
        if found_valid:
            break
    
    if best_adj is None or best_n < 2:
        n = 11
        adj = [set() for _ in range(n)]
        for i in range(n):
            adj[i].add((i+1) % n)
            adj[i].add((i-1) % n)
        best_n = n
        best_adj = adj
    
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
