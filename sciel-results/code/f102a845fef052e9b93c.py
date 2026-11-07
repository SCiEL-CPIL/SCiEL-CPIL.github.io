import time
import random

def construct():
    n = 15
    # Start with K_{7,8}
    # Partitions: 0..6 (size 7), 7..14 (size 8)
    edges = set()
    for u in range(7):
        for v in range(7, 15):
            edges.add((u, v))
    
    best_edges = list(edges)
    best_count = len(best_edges)
    
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    def get_degrees(edge_set):
        deg = [0] * n
        adj = [[] for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        return deg, adj

    def is_valid(edge_set):
        deg, adj = get_degrees(edge_set)
        for u, v in edge_set:
            # Check constraint for edge (u, v)
            # Neighbors of u excluding v
            # Neighbors of v excluding u
            # We need to check if there exist s in N(u)\{v}, t in N(v)\{u} with s != t and deg[s] == deg[t]
            
            # Optimization: Group neighbors by degree
            deg_counts_u = {}
            for s in adj[u]:
                if s == v: continue
                d_s = deg[s]
                if d_s not in deg_counts_u:
                    deg_counts_u[d_s] = []
                deg_counts_u[d_s].append(s)
                
            deg_counts_v = {}
            for t in adj[v]:
                if t == u: continue
                d_t = deg[t]
                if d_t not in deg_counts_v:
                    deg_counts_v[d_t] = []
                deg_counts_v[d_t].append(t)
                
            # Check common degrees
            for d_val in deg_counts_u:
                if d_val in deg_counts_v:
                    set_u = deg_counts_u[d_val]
                    set_v = deg_counts_v[d_val]
                    # If the union of vertices with this degree in both neighborhoods has size > 1, 
                    # we can pick distinct s, t.
                    # Exception: if set_u == set_v == [x], then only (x,x) is possible, which is s==t, so it's safe.
                    # So violation if len(set_u + set_v) > 1 AND NOT (len(set_u)==1 and len(set_v)==1 and set_u[0]==set_v[0])
                    
                    # Actually, simpler:
                    # If there is any s in set_u and t in set_v with s != t, it's a violation.
                    # This is true if:
                    # 1. len(set_u) >= 2
                    # 2. len(set_v) >= 2
                    # 3. len(set_u) >= 1 and len(set_v) >= 1 and there exists s in set_u, t in set_v with s != t.
                    
                    # Let's just check explicitly for small sets.
                    violated = False
                    for s in set_u:
                        for t in set_v:
                            if s != t:
                                violated = True
                                break
                        if violated: break
                    if violated:
                        return False
        return True

    current_edges = set(best_edges)
    
    while time.time() < deadline:
        # Try adding a random missing edge
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        missing = [e for e in all_possible if e not in current_edges]
        
        if missing:
            e = random.choice(missing)
            current_edges.add(e)
            if is_valid(current_edges):
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
            else:
                current_edges.remove(e)
                
        # Occasionally try swapping to escape local optima
        if current_edges and random.random() < 0.1:
            e_remove = random.choice(list(current_edges))
            current_edges.remove(e_remove)
            missing_now = [e for e in all_possible if e not in current_edges]
            if missing_now:
                e_add = random.choice(missing_now)
                current_edges.add(e_add)
                if is_valid(current_edges):
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = list(current_edges)
                else:
                    current_edges.remove(e_add)
                    current_edges.add(e_remove)
                    
    return [[u, v] for u, v in best_edges]
