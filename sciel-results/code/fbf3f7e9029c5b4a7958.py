import time
import random
import sys
from math import log, exp

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    target_edges = 65
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)
    
    # Initial graph: K_8,8 + edge (0,1)
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((u, v))
    edges.add((0, 1))
    
    def edge_to_idx(e):
        u, v = e
        return u * n + v
    
    def compute_score_and_triangles(edge_set):
        """Compute max triangle degree sum and return triangle list."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        max_sum = 0
        triangles = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            s = deg[i] + deg[j] + deg[k]
                            triangles.append((i, j, k, s))
                            if s > max_sum:
                                max_sum = s
        return max_sum, triangles, deg, adj
    
    def compute_soft_score(edge_set, beta=0.5):
        """Soft-max surrogate: encourages reducing spread of triangle sums."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        sums = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            sums.append(deg[i] + deg[j] + deg[k])
        
        if not sums:
            return 0.0, 0
        
        # Soft max: log(sum(exp(beta * s))) / beta approximates max when beta is large
        # But we want to minimize, so use: max + (1/beta)*log(sum(exp(-beta*(s - max))))
        # This penalizes multiple high-sum triangles
        mx = max(sums)
        # Count triangles with sum >= mx-1 to encourage spreading
        near_max = sum(1 for s in sums if s >= mx - 1)
        return mx + 0.1 * near_max, mx
    
    current_score, _, _, _ = compute_score_and_triangles(edges)
    best_score = current_score
    best_edges = set(edges)
    
    # Phase 1: Fast local search with targeted swaps
    # Focus on edges in triangles with high degree sums
    phase1_deadline = start_time + 30
    
    def try_improve_local(edge_set, max_iter=2000):
        """Local search: only accept moves that reduce score."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        cur_score = 0
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > cur_score:
                                cur_score = s
        
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline:
            improved = False
            iters += 1
            # Find edges in worst triangles
            worst_triangles = []
            for i in range(n):
                for j in adj[i]:
                    if j > i:
                        for k in adj[j]:
                            if k > j and k in adj[i]:
                                s = deg[i] + deg[j] + deg[k]
                                if s >= cur_score - 1:
                                    worst_triangles.append((i, j, k, s))
            
            # Try removing edges from worst triangles and adding elsewhere
            candidate_removals = set()
            for (i, j, k, s) in worst_triangles:
                candidate_removals.add((i, j))
                candidate_removals.add((i, k))
                candidate_removals.add((j, k))
            
            # Filter to actual edges
            candidate_removals = [e for e in candidate_removals if e in edge_set]
            random.shuffle(candidate_removals)
            
            # Candidate additions: edges not in graph, prefer low-degree endpoints
            non_edges = [e for e in all_edges if e not in edge_set]
            # Sort by sum of degrees of endpoints (lower is better for adding)
            non_edges.sort(key=lambda e: deg[e[0]] + deg[e[1]])
            
            for e_rem in candidate_removals[:5]:
                for e_add in non_edges[:10]:
                    if e_add == e_rem:
                        continue
                    new_edges = set(edge_set)
                    new_edges.remove(e_rem)
                    new_edges.add(e_add)
                    
                    # Quick score check
                    new_deg = deg[:]
                    new_adj = [set(s) for s in adj]
                    u_r, v_r = e_rem
                    u_a, v_a = e_add
                    new_deg[u_r] -= 1
                    new_deg[v_r] -= 1
                    new_deg[u_a] += 1
                    new_deg[v_a] += 1
                    new_adj[u_r].discard(v_r)
                    new_adj[v_r].discard(u_r)
                    new_adj[u_a].add(v_a)
                    new_adj[v_a].add(u_a)
                    
                    new_score = 0
                    for i in range(n):
                        for j in new_adj[i]:
                            if j > i:
                                for k in new_adj[j]:
                                    if k > j and k in new_adj[i]:
                                        s = new_deg[i] + new_deg[j] + new_deg[k]
                                        if s > new_score:
                                            new_score = s
                    
                    if new_score < cur_score:
                        edge_set = new_edges
                        deg = new_deg
                        adj = new_adj
                        cur_score = new_score
                        improved = True
                        break
                if improved:
                    break
        return edge_set, cur_score
    
    # Phase 1: Local search
    edges, current_score = try_improve_local(edges, max_iter=500)
    if current_score < best_score:
        best_score = current_score
        best_edges = set(edges)
    
    # Phase 2: Simulated annealing with soft objective
    phase2_deadline = start_time + 95
    T = 2.0
    T_min = 0.01
    cooling = 0.997
    
    # Rebuild state
    deg = [0] * n
    adj = [set() for _ in range(n)]
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    
    def full_score(edge_set):
        d = [0] * n
        a = [set() for _ in range(n)]
        for u, v in edge_set:
            d[u] += 1
            d[v] += 1
            a[u].add(v)
            a[v].add(u)
        mx = 0
        for i in range(n):
            for j in a[i]:
                if j > i:
                    for k in a[j]:
                        if k > j and k in a[i]:
                            s = d[i] + d[j] + d[k]
                            if s > mx:
                                mx = s
        return mx
    
    # SA loop
    sa_steps = 0
    while time.time() < phase2_deadline:
        sa_steps += 1
        # Pick a random swap
        non_edges = [e for e in all_edges if e not in edges]
        if not non_edges:
            break
        e_add = random.choice(non_edges)
        e_rem = random.choice(list(edges))
        if e_add == e_rem:
            continue
        
        new_edges = set(edges)
        new_edges.remove(e_rem)
        new_edges.add(e_add)
        
        new_score = full_score(new_edges)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < exp(-delta / max(T, 1e-9)):
            edges = new_edges
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(edges)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Phase 3: Final polish with exact local search
    phase3_deadline = deadline - 5
    edges, current_score = try_improve_local(edges, max_iter=3000)
    if current_score < best_score:
        best_score = current_score
        best_edges = set(edges)
    
    # Final verification
    final_score, _, _, _ = compute_score_and_triangles(best_edges)
    if final_score < best_score:
        best_score = final_score
    
    return {'n': n, 'edges': [[u, v] for u, v in best_edges if u < v]}

if __name__ == "__main__":
    res = construct()
    print(len(res['edges']))
