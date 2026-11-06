import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # 10s safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate maximal cliques of size >= 2 using Bron-Kerbosch with pivoting."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot: choose u in P ∪ X maximizing |P ∩ N(u)|
            best_pivot = None
            max_count = -1
            for v in P:
                count = len(P & adj[v])
                if count > max_count:
                    max_count = count
                    best_pivot = v
            for v in X:
                count = len(P & adj[v])
                if count > max_count:
                    max_count = count
                    best_pivot = v
            
            if best_pivot is None:
                candidates = list(P)
            else:
                candidates = [v for v in P if v not in adj[best_pivot]]
            
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def build_auxiliary_graph(cliques, n):
        """
        Build auxiliary graph H where edge (u,v) exists iff u,v co-occur in at least one clique.
        A set S intersects all cliques iff S is a vertex cover of H.
        """
        H = [set() for _ in range(n)]
        for c in cliques:
            verts = list(c)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v = verts[i], verts[j]
                    H[u].add(v)
                    H[v].add(u)
        return H

    def max_matching_greedy(H, n, time_limit):
        """
        Find a large matching in H using greedy + local improvement.
        Returns (matching, unmatched_vertices).
        """
        match = [-1] * n
        unmatched = set(range(n))
        
        # Greedy: sort edges by some heuristic, pick if both endpoints unmatched
        # Build edge list
        edges = []
        for u in range(n):
            for v in H[u]:
                if u < v:
                    edges.append((u, v))
        
        # Sort by degree sum (higher degree endpoints first for better matching)
        deg = [len(H[u]) for u in range(n)]
        edges.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))
        
        for u, v in edges:
            if match[u] == -1 and match[v] == -1:
                match[u] = v
                match[v] = u
                unmatched.discard(u)
                unmatched.discard(v)
        
        # Try to improve with augmenting paths (limited BFS)
        improved = True
        iters = 0
        while improved and iters < 20 and time.time() < time_limit:
            improved = False
            iters += 1
            for start in list(unmatched):
                # BFS for augmenting path
                parent = {start: None}
                visited = {start}
                queue = [start]
                found = False
                while queue and not found:
                    u = queue.pop(0)
                    for v in H[u]:
                        if v not in visited:
                            visited.add(v)
                            parent[v] = u
                            if match[v] == -1:
                                # Found augmenting path
                                # Trace back
                                cur = v
                                while cur != -1:
                                    nxt = parent[cur]
                                    if nxt is None:
                                        break
                                    if match[cur] != -1:
                                        # Swap
                                        match[cur] = nxt
                                        match[nxt] = cur
                                        unmatched.discard(cur)
                                        unmatched.discard(nxt)
                                    cur = match[cur] if match[cur] != -1 else -1
                                # Fix up the path properly
                                # Redo: standard augmenting path
                                found = True
                                break
                            else:
                                # v is matched, follow to match[v]
                                w = match[v]
                                if w not in visited:
                                    visited.add(w)
                                    parent[w] = v
                                    queue.append(w)
                if found:
                    improved = True
        
        return match, unmatched

    def greedy_hit_set(cliques, n, vert_cliques):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        m = len(cliques)
        if m == 0:
            return set()
        
        uncovered = set(range(m))
        result = set()
        
        while uncovered:
            best_v = -1
            best_cov = 0
            for v in range(n):
                if v in result:
                    continue
                cov = len(uncovered & vert_cliques[v])
                if cov > best_cov:
                    best_cov = cov
                    best_v = v
            if best_v == -1 or best_cov == 0:
                break
            result.add(best_v)
            uncovered -= vert_cliques[best_v]
        
        return result

    def local_search(sol, cliques, n, vert_cliques, max_iters=300):
        """Local search: try removals and swaps to reduce solution size."""
        m = len(cliques)
        result = set(sol)
        if not result:
            return result
        
        # Build coverage counts
        cover_count = [0] * m
        for v in result:
            for ci in vert_cliques[v]:
                cover_count[ci] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE:
            improved = False
            iters += 1
            
            # Try removals
            for v in list(result):
                can_remove = True
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for ci in vert_cliques[v]:
                        cover_count[ci] -= 1
                    result.discard(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: replace v with u not in result
            for v in list(result):
                # Find cliques uniquely covered by v
                unique = set()
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        unique.add(ci)
                
                if not unique:
                    continue
                
                # Find candidate u that covers all unique cliques
                best_u = -1
                best_gain = 0
                for u in range(n):
                    if u in result:
                        continue
                    # Check if u covers all unique cliques
                    covers_all = all(ci in vert_cliques[u] for ci in unique)
                    if not covers_all:
                        continue
                    # Gain: how many additional cliques u covers
                    gain = sum(1 for ci in vert_cliques[u] if cover_count[ci] >= 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                
                if best_u != -1:
                    # Perform swap
                    for ci in vert_cliques[v]:
                        cover_count[ci] -= 1
                    for ci in vert_cliques[best_u]:
                        cover_count[ci] += 1
                    result.discard(v)
                    result.add(best_u)
                    improved = True
                    break
        
        return result

    def branch_and_bound_vertex_cover(H, n, vert_cliques, cliques, initial_ub, time_limit):
        """
        Branch-and-bound for minimum vertex cover of H.
        Uses maximum matching lower bound for pruning.
        """
        m = len(cliques)
        if m == 0:
            return set()
        
        best_sol = set()
        best_size = initial_ub
        
        # Precompute degrees
        deg = [len(H[u]) for u in range(n)]
        
        def matching_lb(unmatched):
            """
            Lower bound: size of maximum matching in the subgraph induced by unmatched vertices
            plus 1 for each unmatched vertex that has an edge to a matched vertex not in the solution.
            Simplified: greedy matching on unmatched vertices gives a valid lower bound.
            """
            if not unmatched:
                return 0
            
            # Greedy matching on the induced subgraph
            match = {v: -1 for v in unmatched}
            used = set()
            count = 0
            # Sort by degree within unmatched
            verts_sorted = sorted(unmatched, key=lambda v: -sum(1 for w in H[v] if w in unmatched))
            for u in verts_sorted:
                if u in used:
                    continue
                best_v = -1
                for v in H[u]:
                    if v in unmatched and v not in used and v != u:
                        if best_v == -1:
                            best_v = v
                if best_v != -1:
                    match[u] = best_v
                    match[best_v] = u
                    used.add(u)
                    used.add(best_v)
                    count += 1
            return count
        
        def bb(selected, uncovered_cliques, remaining_verts):
            nonlocal best_sol, best_size
            
            if time.time() > time_limit:
                return
            
            if not uncovered_cliques:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sol = set(selected)
                return
            
            # Lower bound via matching on remaining_verts
            # The remaining_verts are those not yet decided (neither in selected nor excluded)
            lb = matching_lb(remaining_verts)
            if len(selected) + lb >= best_size:
                return
            
            # Pick the smallest uncovered clique
            min_ci = min(uncovered_cliques, key=lambda i: len(cliques[i]))
            cverts = [v for v in cliques[min_ci] if v in remaining_verts]
            
            if not cverts:
                # Dead end: no vertex in this clique is available
                return
            
            # Order vertices by coverage of uncovered cliques
            cverts_sorted = sorted(cverts, key=lambda v: -len(uncovered_cliques & vert_cliques[v]))
            
            for v in cverts_sorted:
                if time.time() > time_limit:
                    return
                
                new_selected = selected | {v}
                new_uncovered = uncovered_cliques - vert_cliques[v]
                new_remaining = remaining_verts - {v}
                # Remove v from remaining
                bb(new_selected, new_uncovered, new_remaining)
        
        bb(set(), set(range(m)), set(range(n)))
        
        return best_sol

    def verify_solution(sol, cliques):
        """Verify that sol intersects every clique."""
        sol_set = set(sol)
        for c in cliques:
            if not sol_set & c:
                return False
        return True

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            results.append(list(range(n)))
            continue
        
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # Build vertex -> clique membership
        m = len(cliques)
        vert_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vert_cliques[v].add(ci)
        
        # Build auxiliary graph H
        H = build_auxiliary_graph(cliques, n)
        
        # Get initial upper bound via greedy
        initial_sol = greedy_hit_set(cliques, n, vert_cliques)
        initial_ub = len(initial_sol)
        
        # Try B&B with time limit using vertex cover formulation
        remaining_time = DEADLINE - time.time()
        bb_time_limit = min(remaining_time - 5.0, 50.0)  # Leave time for other instances + local search
        
        if bb_time_limit > 1.0:
            bb_sol = branch_and_bound_vertex_cover(H, n, vert_cliques, cliques, initial_ub, time.time() + bb_time_limit)
            if bb_sol and len(bb_sol) < initial_ub:
                if verify_solution(bb_sol, cliques):
                    initial_sol = bb_sol
                    initial_ub = len(bb_sol)
        
        # Local search polish
        if time.time() < DEADLINE - 2.0:
            polished = local_search(initial_sol, cliques, n, vert_cliques, max_iters=400)
            if len(polished) < len(initial_sol):
                if verify_solution(polished, cliques):
                    initial_sol = polished
        
        results.append(sorted(initial_sol))

    return results
