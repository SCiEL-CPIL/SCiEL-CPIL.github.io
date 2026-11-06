import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if there exists a 2-coloring with no monochromatic edge.
    """
    if not edges:
        return True
    
    m = len(edges)
    # Precompute: for each vertex, which edges contain it
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    # State:
    # assign: -1 unassigned, 0 or 1 assigned
    # c0, c1: count of 0s and 1s in each edge
    # cu: count of unassigned in each edge
    
    def propagate(assign, c0, c1, cu):
        # Returns False if contradiction, True otherwise.
        # Modifies 'assign', c0, c1, cu in place.
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                if cu[i] == 0:
                    continue
                if c0[i] == 4:
                    # Must have at least one 1. If cu[i]==1, the last one is 1.
                    if cu[i] == 1:
                        for v in edges[i]:
                            if assign[v] == -1:
                                assign[v] = 1
                                cu[i] -= 1
                                c1[i] += 1
                                for j in vert_edges[v]:
                                    if j != i:
                                        cu[j] -= 1
                                        c1[j] += 1
                                changed = True
                                break
                elif c1[i] == 4:
                    # Must have at least one 0. If cu[i]==1, the last one is 0.
                    if cu[i] == 1:
                        for v in edges[i]:
                            if assign[v] == -1:
                                assign[v] = 0
                                cu[i] -= 1
                                c0[i] += 1
                                for j in vert_edges[v]:
                                    if j != i:
                                        cu[j] -= 1
                                        c0[j] += 1
                                changed = True
                                break
        return True

    def solve(assign, c0, c1, cu):
        # Find next unassigned vertex
        v = -1
        best_crit = -1
        # Heuristic: pick vertex in an edge with most assigned vertices
        for i in range(m):
            if cu[i] == 0:
                continue
            # Criticality: prefer edges close to contradiction
            # If c0[i]+c1[i] is high, it's critical
            crit = c0[i] + c1[i]
            if crit > best_crit:
                best_crit = crit
                # Pick first unassigned vertex in this edge
                for vv in edges[i]:
                    if assign[vv] == -1:
                        v = vv
                        break
        if v == -1:
            return True
        
        # Try 0
        na0 = assign[:]
        nc00 = c0[:]
        nc10 = c1[:]
        ncu0 = cu[:]
        na0[v] = 0
        for j in vert_edges[v]:
            ncu0[j] -= 1
            nc00[j] += 1
        if propagate(na0, nc00, nc10, ncu0):
            if solve(na0, nc00, nc10, ncu0):
                return True
        
        # Try 1
        na1 = assign[:]
        nc01 = c0[:]
        nc11 = c1[:]
        ncu1 = cu[:]
        na1[v] = 1
        for j in vert_edges[v]:
            ncu1[j] -= 1
            nc11[j] += 1
        if propagate(na1, nc01, nc11, ncu1):
            if solve(na1, nc01, nc11, ncu1):
                return True
        
        return False

    assign = [-1] * n
    c0 = [0] * m
    c1 = [0] * m
    cu = [5] * m
    
    if not propagate(assign, c0, c1, cu):
        return False
    
    return solve(assign, c0, c1, cu)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Seed solution: Complete 5-uniform hypergraph on n=9 vertices (126 edges)
    # This is guaranteed non-2-colorable.
    seed_edges = [list(e) for e in combinations(range(9), 5)]
    best_edges = seed_edges
    best_n = 9
    best_count = len(best_edges)
    
    def greedy_delete(n, edges, time_limit):
        """Try to remove edges while maintaining non-2-colorability."""
        current_edges = set(tuple(e) for e in edges)
        
        # Order edges by how "critical" they are? Or just random shuffle.
        # Random shuffle is simple and effective for local search.
        removal_order = list(current_edges)
        random.shuffle(removal_order)
        
        improved = True
        while improved:
            improved = False
            if time.time() > time_limit:
                break
            for edge_tuple in removal_order:
                if time.time() > time_limit:
                    break
                if edge_tuple not in current_edges:
                    continue
                test_edges = current_edges - {edge_tuple}
                test_list = [list(e) for e in test_edges]
                if not is_2_colorable_fast(n, test_list):
                    current_edges = test_edges
                    improved = True
                    break  # Restart removal order after each successful deletion
        
        return [list(e) for e in current_edges]
    
    # 1. Prune the seed
    if time.time() < deadline - 20:
        deleted = greedy_delete(best_n, best_edges, deadline - 20)
        if len(deleted) < best_count:
            best_count = len(deleted)
            best_edges = deleted
    
    # 2. Try other n values with random starts
    # Known results suggest n around 9-12 is optimal for small edge counts.
    for n in [9, 10, 11, 12]:
        if time.time() > deadline - 5:
            break
        
        restarts = 5
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
            
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Start with a subset
            # Heuristic: start with roughly half the edges?
            k = min(len(all_possible), 50 + random.randint(0, 50))
            edges = [list(e) for e in all_possible[:k]]
            
            # Ensure non-2-colorable
            # If it's colorable, add random edges until it's not
            is_non_colorable = not is_2_colorable_fast(n, edges)
            while not is_non_colorable:
                if time.time() > deadline - 5:
                    break
                if len(edges) < len(all_possible):
                    # Add a random edge
                    edge_set = set(tuple(e) for e in edges)
                    candidates = [e for e in all_possible if tuple(e) not in edge_set]
                    if not candidates:
                        break
                    edges.append(list(random.choice(candidates)))
                    is_non_colorable = not is_2_colorable_fast(n, edges)
                else:
                    break
            
            if time.time() > deadline - 5:
                break
            
            if is_non_colorable:
                # Prune
                deleted = greedy_delete(n, edges, deadline - 5)
                if len(deleted) < best_count:
                    best_count = len(deleted)
                    best_edges = deleted
                    best_n = n
    
    # Final verification
    if is_2_colorable_fast(best_n, best_edges):
        # Fallback to seed if something went wrong
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        best_count = len(best_edges)
    
    return {'n': best_n, 'edges': best_edges}
