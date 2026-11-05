import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    n = 16
    vertices = list(range(n))
    
    # Initialize with complete graph
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            edges.add((i, j))
            
    # Precompute vertex degrees for fast lookup? 
    # Or just calculate on the fly.
    
    def get_degree(v, edge_set):
        count = 0
        for u, w in edge_set:
            if u == v or w == v:
                count += 1
        return count

    def contains_q3(edge_set, timeout=0.5):
        t_start = time.time()
        verts = list(vertices)
        random.shuffle(verts)
        
        for S in itertools.combinations(verts, 8):
            if time.time() - t_start > timeout:
                return None
                
            S_set = set(S)
            edge_count = 0
            for u in S:
                for v in S:
                    if u < v and (u, v) in edge_set:
                        edge_count += 1
            if edge_count < 12:
                continue
                
            S_list = list(S)
            for U_idx in itertools.combinations(range(8), 4):
                U = [S_list[i] for i in U_idx]
                V = [S_list[i] for i in range(8) if i not in U_idx]
                
                for pi in itertools.permutations(V):
                    valid = True
                    q3_edges = []
                    for i in range(4):
                        u = U[i]
                        for j in range(4):
                            if j == i:
                                continue
                            v = V[j]
                            if (u, v) not in edge_set and (v, u) not in edge_set:
                                valid = False
                                break
                            q3_edges.append((min(u,v), max(u,v)))
                        if not valid:
                            break
                    if valid:
                        return q3_edges
        return None

    removed_edges = []
    
    # Greedy removal
    while time.time() < deadline - 20:
        q3_edges = contains_q3(edges, timeout=1.0)
        if q3_edges is None:
            break
            
        # Pick edge with highest sum of degrees
        max_deg_sum = -1
        best_e = None
        for e in q3_edges:
            u, v = e
            deg_u = get_degree(u, edges)
            deg_v = get_degree(v, edges)
            if deg_u + deg_v > max_deg_sum:
                max_deg_sum = deg_u + deg_v
                best_e = e
        e_to_remove = best_e if best_e else random.choice(q3_edges)
        
        edges.remove(e_to_remove)
        removed_edges.append(e_to_remove)
        
    # Local Search: Add back edges
    random.shuffle(removed_edges)
    
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        random.shuffle(removed_edges)
        for e in removed_edges:
            if time.time() > deadline - 5:
                break
            if e in edges:
                continue
            
            edges.add(e)
            q3 = contains_q3(edges, timeout=0.2)
            if q3 is not None:
                edges.remove(e)
            else:
                try:
                    removed_edges.remove(e)
                except ValueError:
                    pass
                    
    return [[u, v] for u, v in edges]
