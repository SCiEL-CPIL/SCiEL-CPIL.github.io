"""Optimized Q3-free graph construction on 16 vertices."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s

    n = 16
    # Q3 edges on vertices 0..7
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    q3_adj = {0: {1,2,4}, 1: {0,3,5}, 2: {0,3,6}, 3: {1,2,7}, 4: {0,5,6}, 5: {1,4,7}, 6: {2,4,7}, 7: {3,5,6}}

    # Precompute all 8-subsets
    all_8sets = list(combinations(range(16), 8))

    def contains_q3_fast(vs, edge_set):
        """Check if induced subgraph on vs contains Q3. Fast early-exit."""
        # Build adjacency within vs
        adj = [set() for _ in range(8)]
        vs_list = list(vs)
        for i in range(8):
            for j in range(i+1, 8):
                u, w = vs_list[i], vs_list[j]
                if (min(u,w), max(u,w)) in edge_set:
                    adj[i].add(j)
                    adj[j].add(i)
        # All vertices in Q3 have degree 3, so induced subgraph must have min degree >= 3
        for i in range(8):
            if len(adj[i]) < 3:
                return False
        # Now check for Q3 isomorphism via backtracking
        # Q3 has 8 vertices, each degree 3
        # Try all 8! = 40320 assignments, but prune heavily
        assignment = [-1] * 8
        vs_used = [False] * 8
        
        # Order Q3 vertices by their "signature" to help pruning
        # In Q3, all have degree 3, but we can use the structure
        # Let's just do simple backtracking with good pruning
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            # Get neighbors of q3_v that are already assigned
            assigned_neighbors = []
            for nb in q3_adj[q3_v]:
                if nb < q3_v and assignment[nb] != -1:
                    assigned_neighbors.append(assignment[nb])
            
            # Candidate vs indices
            for vs_i in range(8):
                if vs_used[vs_i]:
                    continue
                # Check if vs_i is adjacent to all assigned neighbors
                ok = True
                for an in assigned_neighbors:
                    if an not in adj[vs_i]:
                        ok = False
                        break
                if not ok:
                    continue
                # Also check non-neighbor constraints for assigned non-neighbors
                for nb in q3_adj[q3_v]:
                    if nb < q3_v and assignment[nb] != -1:
                        continue
                    # nb >= q3_v, not assigned yet, skip
                # Check that vs_i is NOT adjacent to assigned non-neighbors
                for q3_nb in q3_adj[q3_v]:
                    if q3_nb < q3_v and assignment[q3_nb] != -1:
                        continue
                    # This is a neighbor, already checked
                # Check non-neighbors in Q3 that are assigned
                for q3_nonnb in range(8):
                    if q3_nonnb == q3_v:
                        continue
                    if q3_nonnb in q3_adj[q3_v]:
                        continue
                    if q3_nonnb < q3_v and assignment[q3_nonnb] != -1:
                        if assignment[q3_nonnb] in adj[vs_i]:
                            ok = False
                            break
                if not ok:
                    continue
                assignment[q3_v] = vs_i
                vs_used[vs_i] = True
                if backtrack(q3_v + 1):
                    return True
                vs_used[vs_i] = False
                assignment[q3_v] = -1
            return False
        
        return backtrack(0)

    def is_q3_free(edge_set):
        for vs in all_8sets:
            if contains_q3_fast(vs, edge_set):
                return False
        return True

    # Baseline: K3,13
    k313 = set((a, b) for a in range(3) for b in range(3, 16))
    best_edges = set(k313)
    best_count = len(k313)

    # Greedy: try to add edges to K3,13
    current_edges = set(k313)
    candidate_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)
                       if (i, j) not in current_edges]

    def edge_priority(e):
        u, v = e
        if u < 3 and v < 3:
            return 0
        if u >= 3 and v >= 3:
            return 1
        return 2

    candidate_edges.sort(key=edge_priority)

    for edge in candidate_edges:
        if time.time() > deadline - 30:
            break
        u, v = edge
        creates_q3 = False
        remaining = [x for x in range(16) if x != u and x != v]
        for other6 in combinations(remaining, 6):
            vertex_set = (u, v) + other6
            # Quick degree check
            u_deg = 1
            v_deg = 1
            for x in vertex_set:
                if x == u or x == v:
                    continue
                if (min(u, x), max(u, x)) in current_edges:
                    u_deg += 1
                if (min(v, x), max(v, x)) in current_edges:
                    v_deg += 1
            if u_deg < 3 or v_deg < 3:
                continue
            test_edges = current_edges | {(u, v)}
            if contains_q3_fast(vertex_set, test_edges):
                creates_q3 = True
                break

        if not creates_q3:
            current_edges.add(edge)

    if len(current_edges) > best_count:
        best_count = len(current_edges)
        best_edges = set(current_edges)

    # Local search: try to add more edges with smarter ordering
    # Try multiple random seeds for local improvement
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        # Start from a random dense graph
        p = random.uniform(0.6, 0.8)
        edges = set()
        for i in range(16):
            for j in range(i + 1, 16):
                if random.random() < p:
                    edges.add((i, j))
        
        # If it has Q3, remove edges until Q3-free
        # This is expensive, so only do a few trials
        if len(edges) > best_count:
            # Check if Q3-free
            if is_q3_free(edges):
                best_count = len(edges)
                best_edges = set(edges)
            else:
                # Try to fix by removing edges that are part of Q3
                # Too complex, skip for now
                pass

    # Also try starting from K3,13 and doing random local search
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        random.seed(100 + trial)
        edges = set(k313)
        # Randomly add edges
        candidates = [(i, j) for i in range(16) for j in range(i + 1, 16)
                      if (i, j) not in edges]
        random.shuffle(candidates)
        for edge in candidates:
            if time.time() > deadline - 5:
                break
            u, v = edge
            # Quick check: does adding this edge create a Q3?
            # Only check 8-sets containing both u and v
            remaining = [x for x in range(16) if x != u and x != v]
            creates = False
            for other6 in combinations(remaining, 6):
                vertex_set = (u, v) + other6
                # Degree check
                u_deg = 1
                v_deg = 1
                for x in vertex_set:
                    if x == u or x == v:
                        continue
                    if (min(u, x), max(u, x)) in edges:
                        u_deg += 1
                    if (min(v, x), max(v, x)) in edges:
                        v_deg += 1
                if u_deg < 3 or v_deg < 3:
                    continue
                test_edges = edges | {(u, v)}
                if contains_q3_fast(vertex_set, test_edges):
                    creates = True
                    break
            if not creates:
                edges.add(edge)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)

    result = [[u, v] for (u, v) in sorted(best_edges)]
    return result
