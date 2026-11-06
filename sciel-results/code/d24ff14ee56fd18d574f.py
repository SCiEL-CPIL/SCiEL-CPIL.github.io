import random
import math

def construct():
    n = 16
    target_edges = 65
    
    def get_score_and_worst(edge_set):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        max_sum = 0
        worst_triangles = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u].intersection(adj[v])
                    for w in common:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                            worst_triangles = [(u, v, w)]
                        elif s == max_sum:
                            worst_triangles.append((u, v, w))
        return max_sum, worst_triangles

    def local_search(initial_edges, max_steps=5000):
        edges = set(initial_edges)
        current_score, worst_tris = get_score_and_worst(edges)
        
        T = 2.0
        T_min = 0.05
        cooling = 0.999
        
        all_possible_edges = set((i, j) for i in range(n) for j in range(i+1, n))
        
        for step in range(max_steps):
            if T < T_min:
                break
                
            if worst_tris:
                tri = random.choice(worst_tris)
                e1 = (min(tri[0], tri[1]), max(tri[0], tri[1]))
                e2 = (min(tri[0], tri[2]), max(tri[0], tri[2]))
                e3 = (min(tri[1], tri[2]), max(tri[1], tri[2]))
                e_remove = random.choice([e1, e2, e3])
            else:
                e_remove = random.choice(list(edges))
                
            available_adds = [e for e in all_possible_edges if e not in edges]
            if not available_adds:
                break
            e_add = random.choice(available_adds)
            
            new_edges = set(edges)
            new_edges.remove(e_remove)
            new_edges.add(e_add)
            
            new_score, new_worst_tris = get_score_and_worst(new_edges)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                edges = new_edges
                current_score = new_score
                worst_tris = new_worst_tris
                
            T *= cooling
            
            if current_score <= 23:
                break
                
        return edges, current_score

    best_edges = None
    best_score = float('inf')
    
    # Multi-start: try 10 random initial graphs
    for _ in range(10):
        # Generate random graph with 65 edges
        all_possible_edges = list((i, j) for i in range(n) for j in range(i+1, n))
        random.shuffle(all_possible_edges)
        initial_edges = set(all_possible_edges[:target_edges])
        
        res_edges, res_score = local_search(initial_edges)
        
        if res_score < best_score:
            best_score = res_score
            best_edges = res_edges
            
        if best_score <= 23:
            break
            
    edge_list = [[u, v] for u, v in best_edges]
    return {'n': n, 'edges': edge_list}
