# EVOLVE-BLOCK-START
import random
from itertools import combinations

P = {'n': 15}

def construct():
    n = P['n']
    
    def check_valid(adj, deg):
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for s in adj[u]:
                        if s == v:
                            continue
                        for t in adj[v]:
                            if t == u or t == s:
                                continue
                            if deg[s] == deg[t]:
                                return False
        return True

    def greedy_build(seed_val):
        rnd = random.Random(seed_val)
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        all_e = list(combinations(range(n), 2))
        rnd.shuffle(all_e)
        
        for u, v in all_e:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            
            if check_valid(adj, deg):
                edges.append((u, v))
            else:
                adj[u].remove(v)
                adj[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
        
        # Local improvement with swaps
        edge_set = set(edges)
        improved = True
        max_iterations = 50
        iter_count = 0
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            # Try adding
            missing = [e for e in combinations(range(n), 2) if e not in edge_set]
            rnd.shuffle(missing)
            for u, v in missing:
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                if check_valid(adj, deg):
                    edges.append((u, v))
                    edge_set.add((u, v))
                    improved = True
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    deg[u] -= 1
                    deg[v] -= 1
            
            # If no improvement, try swaps
            if not improved:
                swaps_tried = 0
                for u, v in list(edges):
                    if swaps_tried > 20: break
                    # Remove (u, v)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    deg[u] -= 1
                    deg[v] -= 1
                    edge_set.remove((u, v))
                    edges.remove((u, v))
                    
                    # Try adding a random missing edge
                    missing = [e for e in combinations(range(n), 2) if e not in edge_set]
                    rnd.shuffle(missing)
                    found = False
                    for c, d in missing:
                        adj[c].add(d)
                        adj[d].add(c)
                        deg[c] += 1
                        deg[d] += 1
                        if check_valid(adj, deg):
                            edges.append((c, d))
                            edge_set.add((c, d))
                            found = True
                            improved = True
                            break
                        else:
                            adj[c].remove(d)
                            adj[d].remove(c)
                            deg[c] -= 1
                            deg[d] -= 1
                    if not found:
                        # Restore (u, v)
                        adj[u].add(v)
                        adj[v].add(u)
                        deg[u] += 1
                        deg[v] += 1
                        edge_set.add((u, v))
                        edges.append((u, v))
                    swaps_tried += 1

        return edges

    best_edges = []
    best_count = 0
    
    for seed_val in range(2000):
        edges = greedy_build(seed_val)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            if best_count >= 75:
                break
    
    result = [[int(u), int(v)] for u, v in best_edges]
    return result

# EVOLVE-BLOCK-END
