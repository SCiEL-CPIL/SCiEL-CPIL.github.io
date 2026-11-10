"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 110  # 110s safety margin before 120s limit
    n = 16
    
    # Precompute Q3 structure
    # Q3 vertices: 0..7, edges connect i,j if i^j is a power of 2
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if (i ^ j) & (i ^ j - 1) == 0:
                q3_edges.append((i, j))
    
    q3_adj = [[] for _ in range(8)]
    for a, b in q3_edges:
        q3_adj[a].append(b)
        q3_adj[b].append(a)
    
    def creates_q3(u, v, adj):
        """Check if adding edge (u,v) creates a Q3 in the graph."""
        adj_temp = adj | {(min(u,v), max(u,v))}
        
        def in_adj(a, b):
            return (min(a,b), max(a,b)) in adj_temp
        
        neigh_u = [w for w in range(n) if w != v and in_adj(u, w)]
        neigh_v = [w for w in range(n) if w != u and in_adj(v, w)]
        
        for a2 in neigh_u:
            for a4 in neigh_u:
                if a4 == a2:
                    continue
                for a3 in neigh_v:
                    if a3 == a2 or a3 == a4:
                        continue
                    if not in_adj(a2, a3):
                        continue
                    for a5 in neigh_v:
                        if a5 == a2 or a5 == a3 or a5 == a4:
                            continue
                        if not in_adj(a4, a5):
                            continue
                        used = {u, v, a2, a3, a4, a5}
                        for a6 in range(n):
                            if a6 in used:
                                continue
                            if not in_adj(a2, a6) or not in_adj(a4, a6):
                                continue
                            for a7 in range(n):
                                if a7 in used or a7 == a6:
                                    continue
                                if not in_adj(a3, a7) or not in_adj(a5, a7):
                                    continue
                                if not in_adj(a6, a7):
                                    continue
                                return True
        return False
    
    def try_add_edges(edge_set, order):
        """Try adding edges in the given order, return new edge_set and count added."""
        added = 0
        for u, v in order:
            if (min(u,v), max(u,v)) in edge_set:
                continue
            if not creates_q3(u, v, edge_set):
                edge_set.add((min(u,v), max(u,v)))
                added += 1
        return edge_set, added
    
    # Start with K3,13 (39 edges)
    best_edge_set = set()
    for a in range(3):
        for b in range(3, 16):
            best_edge_set.add((a, b))
    
    best_count = len(best_edge_set)
    
    # Generate all possible edges not in initial set
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in best_edge_set:
                all_edges.append((u, v))
    
    # Try multiple random orderings to find a good solution
    random.seed(42)
    best_result = best_edge_set.copy()
    best_result_count = best_count
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Create a random permutation of all edges
        random_order = all_edges[:]
        random.shuffle(random_order)
        
        # Try adding edges in this order
        trial_set = best_edge_set.copy()
        _, added = try_add_edges(trial_set, random_order)
        
        if added + best_count > best_result_count:
            best_result_count = added + best_count
            best_result = trial_set.copy()
        
        # Also try a few more passes with different orders if time permits
        if time.time() < deadline - 30:
            # Try another pass with shuffled remaining edges
            remaining = [e for e in all_edges if e not in trial_set]
            random.shuffle(remaining)
            trial_set2 = trial_set.copy()
            _, added2 = try_add_edges(trial_set2, remaining)
            
            if added2 + best_result_count > best_result_count:
                best_result_count = added2 + best_result_count
                best_result = trial_set2.copy()
    
    # Convert back to list of edges
    result_edges = [[u, v] for u, v in best_result]
    return result_edges
# EVOLVE-BLOCK-END
