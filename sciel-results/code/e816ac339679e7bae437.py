import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 148
    
    n = 16
    adj = [[False]*n for _ in range(n)]
    edge_set = set()
    deg = [0]*n
    
    def add_edge(u, v):
        u, v = min(u,v), max(u,v)
        if u != v and (u,v) not in edge_set:
            edge_set.add((u,v))
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
    
    def remove_edge(u, v):
        u, v = min(u,v), max(u,v)
        if (u,v) in edge_set:
            edge_set.remove((u,v))
            adj[u][v] = adj[v][u] = False
            deg[u] -= 1
            deg[v] -= 1
    
    def compute_score():
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def get_triangles_involving(u, v):
        """Get all triangles that include edge (u,v)"""
        triangles = []
        for k in range(n):
            if k != u and k != v and adj[u][k] and adj[v][k]:
                triangles.append((u, v, k))
        return triangles
    
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    
    best_score = float('inf')
    best_edge_set = None
    best_adj = None
    best_deg = None
    
    def try_local_search(init_edge_set, max_iterations=100000, use_sa=False):
        nonlocal best_score, best_edge_set, best_adj, best_deg
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
            deg[i] = 0
        for e in init_edge_set:
            add_edge(*e)
        
        current_score = compute_score()
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
        
        no_improve = 0
        sa_temp = 2.0 if use_sa else 0.0
        sa_cool = 0.98 if use_sa else 1.0
        
        for it in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try single swaps with move ordering: prefer removing high-degree edges
            edges_list = sorted(list(edge_set), key=lambda e: -(deg[e[0]] + deg[e[1]]))
            
            for e_remove in edges_list:
                if time.time() > deadline - 3:
                    break
                u_rem, v_rem = e_remove
                
                # Try adding edges, prioritize those that don't create high-degree triangles
                for e_add in all_edges:
                    if e_add in edge_set or e_add == e_remove:
                        continue
                    
                    u_add, v_add = e_add
                    
                    # Quick heuristic: skip if this would likely increase max triangle sum
                    # Check if adding this edge creates triangles with high degree sum
                    max_potential = 0
                    for k in range(n):
                        if k != u_add and k != v_add and adj[u_add][k] and adj[v_add][k]:
                            s = deg[u_add] + 1 + deg[v_add] + 1 + deg[k]
                            if s > max_potential:
                                max_potential = s
                    # Also check triangles involving removed edge vertices
                    for k in range(n):
                        if k != u_rem and k != v_rem and adj[u_rem][k] and adj[v_rem][k]:
                            s = deg[u_rem] - 1 + deg[v_rem] - 1 + deg[k]
                            if s > max_potential:
                                max_potential = s
                    
                    if max_potential > current_score + 2:
                        continue
                    
                    remove_edge(*e_remove)
                    add_edge(*e_add)
                    
                    new_score = compute_score()
                    
                    accept = False
                    if new_score < current_score:
                        accept = True
                    elif use_sa and sa_temp > 0.01:
                        import math as _math
                        if random.random() < _math.exp((current_score - new_score) / sa_temp):
                            accept = True
                    
                    if accept:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_edge_set = set(edge_set)
                            best_adj = [row[:] for row in adj]
                            best_deg = deg[:]
                        break
                    else:
                        remove_edge(*e_add)
                        add_edge(*e_remove)
                
                if improved:
                    break
            
            if not improved:
                # Try 2-swap moves with sampling
                improved2 = False
                edges_list2 = list(edge_set)
                num_pairs = min(len(edges_list2), 20)
                for i in range(num_pairs):
                    if time.time() > deadline - 3:
                        break
                    for j in range(i+1, min(i+10, len(edges_list2))):
                        if time.time() > deadline - 3:
                            break
                        e_rem1, e_rem2 = edges_list2[i], edges_list2[j]
                        
                        # Sample some add edges
                        available = [e for e in all_edges if e not in edge_set and e != e_rem1 and e != e_rem2]
                        if len(available) < 2:
                            continue
                        
                        sample_size = min(15, len(available))
                        sampled = random.sample(available, sample_size)
                        
                        for e_add1 in sampled:
                            if time.time() > deadline - 3:
                                break
                            for e_add2 in sampled:
                                if e_add2 == e_add1:
                                    continue
                                
                                remove_edge(*e_rem1)
                                remove_edge(*e_rem2)
                                add_edge(*e_add1)
                                add_edge(*e_add2)
                                
                                new_score = compute_score()
                                
                                if new_score < current_score:
                                    current_score = new_score
                                    improved2 = True
                                    if current_score < best_score:
                                        best_score = current_score
                                        best_edge_set = set(edge_set)
                                        best_adj = [row[:] for row in adj]
                                        best_deg = deg[:]
                                    break
                                else:
                                    remove_edge(*e_add1)
                                    remove_edge(*e_add2)
                                    add_edge(*e_rem1)
                                    add_edge(*e_rem2)
                                
                                if improved2:
                                    break
                            
                            if improved2:
                                break
                        
                        if improved2:
                            break
                    
                    if improved2:
                        break
                
                if improved2:
                    no_improve = 0
                    continue
                else:
                    no_improve += 1
                    if use_sa:
                        sa_temp *= sa_cool
                        if no_improve > 100:
                            break
                    else:
                        if no_improve > 30:
                            break
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Config 1: K_{8,8} + one edge in part 0 (various positions)
    for u in range(8):
        for v in range(u+1, 8):
            base = set((a,b) for a in range(8) for b in range(8,16))
            initial_configs.append(base | {(u,v)})
    
    # Config 2: K_{8,8} + one edge in part 1
    for u in range(8):
        for v in range(u+1, 8):
            base = set((a,b) for a in range(8) for b in range(8,16))
            initial_configs.append(base | {(u+8, v+8)})
    
    # Config 3: K_{7,9} + 2 edges within part 0
    for u in range(7):
        for v in range(u+1, 7):
            base = set((a,b) for a in range(7) for b in range(7,16))
            for w in range(v+1, 7):
                initial_configs.append(base | {(u,v), (v,w)})
    
    # Config 4: K_{6,10} + 5 edges within part 0
    for combo in [(0,1,2,3,4), (0,1,2,3,5), (0,1,2,4,5), (0,1,3,4,5), (0,2,3,4,5), (1,2,3,4,5)]:
        base = set((a,b) for a in range(6) for b in range(6,16))
        internal_edges = [(combo[i], combo[j]) for i in range(len(combo)) for j in range(i+1, len(combo))]
        if len(internal_edges) >= 5:
            initial_configs.append(base | set(internal_edges[:5]))
    
    # Config 5: More balanced - start from random graph with 65 edges
    random.seed(42)
    for trial in range(5):
        random_edges = random.sample(all_edges, 65)
        initial_configs.append(set(random_edges))
    
    # Shuffle and try configurations
    random.seed(42)
    random.shuffle(initial_configs)
    
    # Try configurations with and without simulated annealing
    tried = 0
    for idx, config in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        use_sa = (idx % 3 == 0)  # Use SA for every 3rd config
        try_local_search(config, max_iterations=30000, use_sa=use_sa)
        tried += 1
        if tried >= 25:
            break
    
    if best_edge_set is None:
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
            deg[i] = 0
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        add_edge(0, 1)
        best_edge_set = set(edge_set)
    
    edge_list = [[u, v] for (u, v) in best_edge_set]
    return {'n': n, 'edges': edge_list}
