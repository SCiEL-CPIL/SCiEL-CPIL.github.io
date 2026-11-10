def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 85
    
    n = 128
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            gcd_table[i][j] = gcd(i, j)
    
    # Build the 3-uniform hypergraph of forbidden triples
    # A triple (a,b,c) is forbidden if gcd(a,b)==gcd(a,c)==gcd(b,c)
    # We want maximum independent set in this hypergraph
    forbidden_triples = set()
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, n+1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    forbidden_triples.add((a, b, c))
    
    # Build adjacency: for each vertex, which other vertices form forbidden triples with it
    # conflict[v] = set of vertices u such that there exists w with (v,u,w) forbidden
    # Actually we need: if we pick v, which pairs (u,w) become "conflicted"
    # Better: build for each vertex v, the set of hyperedges containing v
    vertex_edges = {v: [] for v in range(1, n+1)}
    for (a, b, c) in forbidden_triples:
        vertex_edges[a].append((b, c))
        vertex_edges[b].append((a, c))
        vertex_edges[c].append((a, b))
    
    # For branch and bound: state is (selected, available)
    # selected: set of chosen vertices
    # available: set of vertices that can still be added
    # When we add vertex v, we remove from available all vertices u such that
    # there exists w in selected with (v,u,w) forbidden, AND remove v itself
    # Actually: when adding v, for each (u,w) in vertex_edges[v], if w is in selected,
    # then u cannot be added. Also v itself is removed.
    
    def can_add_fast(x, selected_set):
        """Check if x can be added to selected_set"""
        for (u, w) in vertex_edges[x]:
            if w in selected_set:
                return False
        return True
    
    def hill_climb(S):
        S = list(S)
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, 129):
                if x not in S_set and can_add_fast(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    improved = True
            if improved:
                S.sort()
        return S
    
    # Greedy with different orderings to get initial solutions
    all_elements = list(range(1, 129))
    
    def greedy(order):
        S = []
        S_set = set()
        for x in order:
            if x not in S_set and can_add_fast(x, S_set):
                S.append(x)
                S_set.add(x)
        return S
    
    # Branch and bound for maximum independent set
    # We use a recursive approach with memoization-like pruning
    def branch_and_bound(available, selected_size, best_ref):
        """
        available: frozenset of vertices that can still be considered
        selected_size: number of vertices already selected
        best_ref: mutable list [best_size_found]
        Returns the best independent set found (as a set)
        """
        if time.time() >= deadline:
            return None
        if not available:
            return None
        
        # Upper bound: selected_size + len(available)
        if selected_size + len(available) <= best_ref[0]:
            return None
        
        # Try to add all remaining (optimistic)
        # Pick vertex with highest degree in the induced hypergraph on available
        # Degree of v in available = number of other vertices u in available
        # such that there's a forbidden triple involving v and u
        # Simplified: count how many vertices in available are "conflicted" with v
        # if we were to NOT pick v
        
        # For ordering: pick vertex that appears in most forbidden triples within available
        # This is expensive, so use a simpler heuristic: pick vertex with most edges
        # to other available vertices
        
        # Build a quick degree estimate
        best_v = None
        best_deg = -1
        for v in available:
            deg = 0
            for (u, w) in vertex_edges[v]:
                if u in available or w in available:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        # Branch 1: include best_v
        # When we include v, we must exclude all u such that there exists w in selected
        # with (v,u,w) forbidden. But we don't track selected set here...
        # Let's restructure: pass selected set
        
        return None  # Placeholder
    
    # Simpler branch and bound with explicit selected set
    def branch_and_bound_v2(selected, available, best_size, best_set_ref):
        """
        selected: frozenset of vertices already chosen
        available: frozenset of vertices that can still be added
        best_size: current best size found
        best_set_ref: mutable [best_set]
        """
        if time.time() >= deadline:
            return
        if not available:
            if len(selected) > best_size:
                best_set_ref[0] = set(selected)
                best_set_ref[1] = len(selected)
            return
        
        # Upper bound
        if len(selected) + len(available) <= best_size:
            return
        
        # Pick a vertex to branch on: highest "conflict degree" in available
        # A vertex v has high conflict if it's in many forbidden triples with available vertices
        # Simple heuristic: pick vertex that conflicts with most available vertices
        # v conflicts with u if there exists w such that (v,u,w) is forbidden
        # For speed, precompute conflict pairs
        
        # Build conflict set: for each v, which u in available conflict with v
        # (i.e., there's a forbidden triple containing both v and u)
        # This means: if we pick v, u might still be addable if the third vertex isn't in selected
        # Actually the constraint is: we can't have three vertices forming a forbidden triple
        # So v and u can both be in the solution as long as no w in the solution completes the triple
        
        # For branching: pick v, then:
        # Branch 1: include v. New available = available - {v} - {u : exists w in selected with (v,u,w) forbidden}
        # Branch 2: exclude v. New available = available - {v}
        
        # Choose v with max number of "potential conflicts" 
        # i.e., v is in the most forbidden triples that have at least one other vertex in available
        v = None
        max_conflicts = -1
        for candidate in available:
            conflicts = 0
            for (u, w) in vertex_edges[candidate]:
                if u in available or w in available:
                    conflicts += 1
            if conflicts > max_conflicts:
                max_conflicts = conflicts
                v = candidate
        
        # Branch 1: include v
        new_selected = selected | {v}
        # Remove v from available
        # Also remove any u where there exists w in new_selected (excluding v itself) such that (v,u,w) forbidden
        # Wait: the constraint is that no three vertices in the final set form a forbidden triple
        # If v is in the set, then for any forbidden triple (v, u, w), we can't have both u and w in the set
        # So when adding v, we don't immediately remove u or w; we just need to ensure
        # that we don't add both u and w later
        # This makes the state more complex: we need to track "conflicting pairs"
        
        # Alternative: when adding v, for each forbidden triple (v, u, w):
        # if u is already in selected, remove w from available
        # if w is already in selected, remove u from available
        # if neither is in selected, we need to remember that u and w can't both be added
        # This "remembering" makes it not a simple set-based state
        
        # Given the complexity, let's use a different approach:
        # Use the fact that the hypergraph is sparse and use greedy + local search more aggressively
        
        return
    
    # Given the complexity of exact B&B for 3-uniform hypergraphs, 
    # let's use a more sophisticated greedy + local search with better heuristics
    
    # Strategy: 
    # 1. Multiple greedy passes with different vertex orderings (by degree, random, etc.)
    # 2. Local improvement: try swapping one vertex for another
    # 3. Simulated annealing style kicks
    
    # Precompute degree of each vertex (number of forbidden triples it's in)
    vertex_degree = {v: len(vertex_edges[v]) for v in range(1, n+1)}
    
    # Orderings
    orders = [
        sorted(all_elements, key=lambda x: -vertex_degree[x]),  # high degree first
        sorted(all_elements, key=lambda x: vertex_degree[x]),   # low degree first
        list(reversed(all_elements)),
        all_elements,
        [x for x in all_elements if x % 2 == 0],
        [x for x in all_elements if x % 2 == 1],
        sorted(all_elements, key=lambda x: -x.bit_count()),
        sorted(all_elements, key=lambda x: x.bit_count()),
    ]
    
    rng = random.Random(42)
    for _ in range(10):
        orders.append(rng.sample(all_elements, 128))
    
    best_S = []
    for order in orders:
        if time.time() >= deadline:
            break
        S = greedy(order)
        S = hill_climb(S)
        if len(S) > len(best_S):
            best_S = S
    
    # Local improvement: try removing one vertex and adding two others
    def local_improve(S, max_iters=500):
        S = list(S)
        S_set = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for x in list(S):
                # Try removing x
                S_set.remove(x)
                S.remove(x)
                # Try adding any two vertices that fit
                added = []
                for a in range(1, 129):
                    if a in S_set:
                        continue
                    if can_add_fast(a, S_set):
                        S_set.add(a)
                        S.append(a)
                        added.append(a)
                        if len(added) == 2:
                            break
                if len(added) >= 2:
                    improved = True
                    S.sort()
                else:
                    # Revert: put x back, remove added
                    for a in added:
                        S_set.remove(a)
                        S.remove(a)
                    S_set.add(x)
                    S.append(x)
        return S
    
    # Apply local improvement
    if best_S:
        best_S = local_improve(best_S)
    
    # Kicks: remove some random vertices and re-greedy
    for _ in range(300):
        if time.time() >= deadline:
            break
        if best_S:
            kick_size = min(len(best_S), rng.randint(3, 10))
            to_remove = set(rng.sample(best_S, kick_size))
            S = [x for x in best_S if x not in to_remove]
            # Greedy fill
            S_set = set(S)
            for x in range(1, 129):
                if x not in S_set and can_add_fast(x, S_set):
                    S.append(x)
                    S_set.add(x)
            S = hill_climb(S)
            S = local_improve(S, max_iters=100)
            if len(S) > len(best_S):
                best_S = S
        else:
            order = rng.sample(all_elements, 128)
            S = greedy(order)
            S = hill_climb(S)
            S = local_improve(S)
            if len(S) > len(best_S):
                best_S = S
    
    return best_S
