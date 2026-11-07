import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Precompute adjacency from K to I
        # adj_k_to_i[u] = list of v in I such that (u,v) is an edge
        adj_k_to_i = [[] for _ in range(k)]
        for v in range(k, n):
            for u in neighbors[v]:
                adj_k_to_i[u].append(v)
        
        # Total edges
        total_edges = k * (k - 1) // 2
        for v in range(k, n):
            total_edges += len(neighbors[v])
            
        # Objective function:
        # Minimize Total_Cliques = Num_Large + (Total_Edges - Edges_Covered)
        # Maximize Score = Edges_Covered - Num_Large
        # Edges_Covered = sum over large cliques C of |C|*(|C|-1)/2
        # Num_Large = count of large cliques
        # For a clique S U {v} with |S|=m:
        # Edges covered = m*(m+1)/2
        # Num Large = 1
        # Contribution to Score = m*(m+1)/2 - 1
        
        def calc_score(assign):
            # assign: list of length k, assign[u] is v (index in I, so 0 to n-k-1) or -1
            # Group K vertices by assigned I vertex
            groups = {}
            for u in range(k):
                v_idx = assign[u]
                if v_idx != -1:
                    if v_idx not in groups:
                        groups[v_idx] = []
                    groups[v_idx].append(u)
            
            score = 0
            for v_idx, verts in groups.items():
                m = len(verts)
                if m >= 2:
                    # Check validity: all u in verts must be adjacent to v_idx
                    # This is guaranteed by the move generation if we are careful, 
                    # but let's assume valid state.
                    edges_covered = m * (m + 1) // 2
                    score += edges_covered - 1
            
            # Return negative score to minimize? Or just maximize score.
            # We want to maximize this score.
            return score

        # Initial state: random assignment
        # Each u in K is assigned to a random v in N(u) or -1
        # With some probability, assign to -1 to allow flexibility
        
        best_assign = None
        best_score = -float('inf')
        
        rng = random.Random(seed + 777)
        
        # Multiple restarts
        num_restarts = 10
        iterations_per_restart = 5000
        
        for r in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Initialize
            assign = [0] * k
            for u in range(k):
                if adj_k_to_i[u]:
                    # 50% chance to assign, 50% to leave unassigned
                    if rng.random() < 0.5:
                        assign[u] = rng.choice(adj_k_to_i[u])
                    else:
                        assign[u] = -1
                else:
                    assign[u] = -1
            
            current_score = calc_score(assign)
            current_assign = assign[:]
            
            if current_score > best_score:
                best_score = current_score
                best_assign = current_assign[:]
                
            # Simulated Annealing
            T = 10.0
            cooling = 0.99
            
            for i in range(iterations_per_restart):
                if time.time() > deadline - 1.0:
                    break
                    
                # Pick a random K vertex
                u = rng.randint(0, k - 1)
                old_v = current_assign[u]
                
                # Pick a new assignment for u
                # Options: -1 or any v in adj_k_to_i[u]
                options = [-1] + adj_k_to_i[u]
                new_v = rng.choice(options)
                
                if new_v == old_v:
                    continue
                
                current_assign[u] = new_v
                new_score = calc_score(current_assign)
                
                delta = new_score - current_score
                
                if delta >= 0 or rng.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_assign = current_assign[:]
                else:
                    current_assign[u] = old_v
                    
                T *= cooling
                
        # Convert best_assign to cliques
        # 1. Form large cliques from groups
        groups = {}
        for u in range(k):
            v_idx = best_assign[u]
            if v_idx != -1:
                if v_idx not in groups:
                    groups[v_idx] = []
                groups[v_idx].append(u)
                
        cliques = []
        covered_k_edges = set()
        covered_ik_edges = set()
        
        for v_idx, verts in groups.items():
            if len(verts) >= 2:
                v_real = v_idx + k
                clique = sorted(verts + [v_real])
                cliques.append(clique)
                
                # Mark edges
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        u1, u2 = verts[i], verts[j]
                        covered_k_edges.add((min(u1, u2), max(u1, u2)))
                        covered_ik_edges.add((min(u1, v_real), max(u1, v_real)))
                        covered_ik_edges.add((min(u2, v_real), max(u2, v_real)))
                        
        # 2. Add residual K edges
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_k_edges:
                    cliques.append([u, w])
                    
        # 3. Add residual I-K edges
        for v in range(k, n):
            for u in neighbors[v]:
                e = (min(u, v), max(u, v))
                if e not in covered_ik_edges:
                    cliques.append([u, v])
                    
        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline - 1.0:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    while len(results) < 3:
        results.append([])
        
    return results
