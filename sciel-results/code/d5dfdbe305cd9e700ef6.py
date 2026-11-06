import time
import random

def construct():
    start_time = time.time()
    # Leave 20s margin for overhead/return
    time_limit = 100 

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
        return adj, edges

    def is_clique(vertices, adj):
        if len(vertices) <= 2:
            return True
        v_list = list(vertices)
        for i in range(len(v_list)):
            u = v_list[i]
            for j in range(i + 1, len(v_list)):
                v = v_list[j]
                if v not in adj[u]:
                    return False
        return True

    def solve_instance(n, p, seed):
        adj, edges = get_graph(n, p, seed)
        
        # 1. Greedy Initialization
        remaining_edges = set(edges)
        cliques = []
        
        edge_list = list(edges)
        random.shuffle(edge_list)
        
        for e in edge_list:
            if e not in remaining_edges:
                continue
            u, v = e
            clique = {u, v}
            remaining_edges.remove(e)
            
            # Expand greedily
            # Get common neighbors of u and v
            candidates = list(adj[u] & adj[v])
            random.shuffle(candidates)
            
            for w in candidates:
                # Check if w is connected to all in clique
                can_add = True
                for x in clique:
                    if w not in adj[x]:
                        can_add = False
                        break
                if can_add:
                    clique.add(w)
                    # Remove edges between w and existing clique members
                    for x in list(clique):
                        if x == w:
                            continue
                        rem_e = tuple(sorted((w, x)))
                        if rem_e in remaining_edges:
                            remaining_edges.remove(rem_e)
                            
            cliques.append(clique)
            
        # 2. Local Search Refinement
        # Goal: Minimize len(cliques)
        # Moves:
        # A. Merge two cliques if union is a clique.
        # B. Move an edge from C1 to C2 if C2 U {e} is a clique.
        
        # Shuffle cliques to randomize
        random.shuffle(cliques)
        
        while time.time() - start_time < time_limit:
            improved = False
            # Try merges first
            if len(cliques) > 1:
                i, j = random.sample(range(len(cliques)), 2)
                C1 = cliques[i]
                C2 = cliques[j]
                union = C1 | C2
                if is_clique(union, adj):
                    cliques[i] = union
                    cliques.pop(j)
                    improved = True
            
            if not improved:
                # Try edge moves
                # Only try if we have time
                if len(cliques) > 1:
                    i = random.randint(0, len(cliques) - 1)
                    C1 = cliques[i]
                    if len(C1) > 2:
                        # Pick random edge from C1
                        v1, v2 = random.sample(list(C1), 2)
                        
                        # Temporarily remove
                        C1_copy = C1.copy()
                        C1_copy.remove(v1)
                        C1_copy.remove(v2)
                        
                        # Try to move to random other clique
                        found = False
                        # Try a few random targets
                        for _ in range(3):
                            j = random.randint(0, len(cliques) - 1)
                            if j == i:
                                continue
                            C2 = cliques[j]
                            new_C2 = C2 | {v1, v2}
                            if is_clique(new_C2, adj):
                                # Move
                                cliques[j] = new_C2
                                if len(C1_copy) >= 2:
                                    cliques[i] = C1_copy
                                else:
                                    cliques.pop(i)
                                found = True
                                break
                        # If not found, do nothing (state unchanged)
                        
        return [list(c) for c in cliques]

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    return [res1, res2]
