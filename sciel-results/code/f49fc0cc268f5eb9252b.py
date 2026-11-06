import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin for 120s limit

    def make_paley(q):
        if q % 4 != 1:
            return None
        squares = {x * x % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in squares:
                    edges.append([i, j])
        return {'n': q, 'edges': edges}

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def build_adj_bit(n, edges):
        """Build adjacency as list of integers (bitmasks)."""
        adj_bits = [0] * n
        for u, v in edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        return adj_bits

    def has_indep_6_bitmask(n, adj_bits):
        """Fast check for independent set of size 6 using bitmask backtracking."""
        best = [0]  # mutable for closure
        
        def backtrack(start, current_mask, size):
            if size == 6:
                best[0] = 1
                return True
            # Pruning: not enough vertices left
            remaining = n - start
            if remaining < 6 - size:
                return False
            # Try vertices from start to n-1
            for v in range(start, n):
                if best[0]:
                    return True
                # Check if v is independent from all in current_mask
                if adj_bits[v] & current_mask == 0:
                    if backtrack(v + 1, current_mask | (1 << v), size + 1):
                        return True
            return False
        
        backtrack(0, 0, 0)
        return best[0] == 1

    def find_indep_sets_5_bitmask(n, adj_bits, max_sets=2000):
        """Find independent sets of size 5 using bitmask backtracking."""
        indep_sets = []
        
        def backtrack(start, current_mask, current_list, size):
            if len(indep_sets) >= max_sets:
                return
            if size == 5:
                indep_sets.append(tuple(current_list))
                return
            for v in range(start, n):
                if len(indep_sets) >= max_sets:
                    return
                if adj_bits[v] & current_mask == 0:
                    current_list.append(v)
                    backtrack(v + 1, current_mask | (1 << v), current_list, size + 1)
                    current_list.pop()
        
        backtrack(0, 0, [], 0)
        return indep_sets

    def find_triangles(n, adj):
        triangles = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    for k in common:
                        if k > j:
                            triangles.append((i, j, k))
        return triangles

    def is_valid(n, edges):
        """Check if graph is K4-free and has no independent set of size 6."""
        adj = build_adj(n, edges)
        adj_bits = build_adj_bit(n, edges)
        
        # K4 check
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    for k in range(j + 1, n):
                        if k in adj[i] and k in adj[j]:
                            common = adj[i] & adj[j] & adj[k]
                            if common:
                                return False
        
        # Independent set of size 6 check
        if has_indep_6_bitmask(n, adj_bits):
            return False
            
        return True

    def walk_sat_weighted(n, clauses, max_iters=5000, weights=None):
        """WalkSAT with clause weighting for better convergence."""
        x = [random.choice([True, False]) for _ in range(n)]
        if weights is None:
            weights = [1.0] * len(clauses)
        
        def is_sat(clause):
            for var, sign in clause:
                val = x[var]
                if sign == 1:
                    if val:
                        return True
                else:
                    if not val:
                        return True
            return False

        def clause_score(clause):
            """Score clause by how many literals are satisfied (0 or 1)."""
            sat_count = 0
            for var, sign in clause:
                val = x[var]
                if sign == 1:
                    if val:
                        sat_count += 1
                else:
                    if not val:
                        sat_count += 1
            return sat_count

        for it in range(max_iters):
            # Check if all clauses are satisfied
            all_sat = True
            for c in clauses:
                if not is_sat(c):
                    all_sat = False
                    break
            if all_sat:
                return x
            
            # Find unsatisfied clauses
            unsat = [c for c in clauses if not is_sat(c)]
            if not unsat:
                return x
            
            # Choose a random unsatisfied clause
            clause = random.choice(unsat)
            
            # Choose which variable to flip
            # With 50% probability, pick a variable from the clause
            # With 50% probability, pick a random variable
            if random.random() < 0.5:
                var, _ = random.choice(clause)
            else:
                var = random.randint(0, n - 1)
            
            # Flip the variable
            x[var] = not x[var]
            
            # Update clause weights based on satisfaction changes
            # Increase weight for clauses that became unsatisfied
            # Decrease weight for clauses that became satisfied
            for idx, c in enumerate(clauses):
                was_sat = is_sat(c)
                # We already flipped, so check current state
                now_sat = is_sat(c)
                if not was_sat and now_sat:
                    weights[idx] = max(0.5, weights[idx] * 0.95)
                elif was_sat and not now_sat:
                    weights[idx] = min(10.0, weights[idx] * 1.1)
        
        return None

    def try_extend(n, adj, adj_bits, edges, max_restarts=5):
        """Try to add one vertex. Returns new (n, adj, adj_bits, edges) or None."""
        triangles = find_triangles(n, adj)
        indep_sets_5 = find_indep_sets_5_bitmask(n, adj_bits, max_sets=3000)
        
        clauses = []
        for tri in triangles:
            u, v, w = tri
            clauses.append([(u, -1), (v, -1), (w, -1)])
        for ind in indep_sets_5:
            u, v, w, x, y = ind
            clauses.append([(u, 1), (v, 1), (w, 1), (x, 1), (y, 1)])
        
        if not clauses:
            assignment = [False] * n
        else:
            assignment = None
            for _ in range(max_restarts):
                assignment = walk_sat_weighted(n, clauses, max_iters=5000)
                if assignment is not None:
                    break
            
            if assignment is None and not indep_sets_5:
                # Greedy fallback
                x = [True] * n
                changed = True
                while changed:
                    changed = False
                    for tri in triangles:
                        u, v, w = tri
                        if x[u] and x[v] and x[w]:
                            idx = random.choice([u, v, w])
                            x[idx] = False
                            changed = True
                assignment = x
        
        if assignment is None:
            return None
        
        # Verify
        valid = True
        for c in clauses:
            sat = False
            for var, sign in c:
                val = assignment[var]
                if sign == 1:
                    if val:
                        sat = True
                        break
                else:
                    if not val:
                        sat = True
                        break
            if not sat:
                valid = False
                break
        
        if not valid:
            return None
        
        new_n = n + 1
        new_edges = edges[:]
        for i in range(n):
            if assignment[i]:
                new_edges.append([i, new_n - 1])
        
        new_adj = build_adj(new_n, new_edges)
        new_adj_bits = build_adj_bit(new_n, new_edges)
        return (new_n, new_adj, new_adj_bits, new_edges)

    def local_search_polish(n, adj, adj_bits, edges, time_limit=8):
        """Try to add edges or remove edges to enable further extension.
        Returns improved (n, adj, adj_bits, edges) or original if no improvement."""
        start = time.time()
        best_n = n
        best_edges = edges[:]
        best_adj = [s.copy() for s in adj]
        best_adj_bits = adj_bits[:]
        
        # Strategy: try random edge removals, check validity, try to extend
        for attempt in range(30):
            if time.time() - start > time_limit:
                break
            if time.time() > deadline - 10:
                break
            
            # Pick a random edge to remove
            if not edges:
                break
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            # Create modified graph without this edge
            new_edges = edges[:]
            new_edges.pop(idx)
            
            # Check if still valid
            new_adj = build_adj(n, new_edges)
            new_adj_bits = build_adj_bit(n, new_edges)
            
            if not is_valid(n, new_edges):
                continue
            
            # Try to extend this modified graph
            result = try_extend(n, new_adj, new_adj_bits, new_edges, max_restarts=3)
            if result is not None:
                new_n, new_adj, new_adj_bits, new_edges = result
                if new_n > best_n:
                    best_n = new_n
                    best_edges = new_edges
                    best_adj = new_adj
                    best_adj_bits = new_adj_bits
            
            # Also try adding a random edge (carefully)
            if time.time() - start > time_limit:
                break
            # Pick two non-adjacent vertices and try adding edge
            non_adj_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    if j not in adj[i]:
                        non_adj_pairs.append((i, j))
            if non_adj_pairs:
                u, v = random.choice(non_adj_pairs)
                test_edges = edges + [[u, v]]
                test_adj = build_adj(n, test_edges)
                test_adj_bits = build_adj_bit(n, test_edges)
                
                if is_valid(n, test_edges):
                    # Try to extend
                    result = try_extend(n, test_adj, test_adj_bits, test_edges, max_restarts=3)
                    if result is not None:
                        new_n, new_adj, new_adj_bits, new_edges = result
                        if new_n > best_n:
                            best_n = new_n
                            best_edges = new_edges
                            best_adj = new_adj
                            best_adj_bits = new_adj_bits
        
        if best_n > n:
            return (best_n, best_adj, best_adj_bits, best_edges)
        return (n, adj, adj_bits, edges)

    # Main search
    best_g = None
    best_n = 0
    
    for q in [29, 17, 13, 5]:
        if time.time() > deadline - 20:
            break
        
        G = make_paley(q)
        if G is None:
            continue
        
        n = G['n']
        edges = G['edges']
        
        if not is_valid(n, edges):
            continue
        
        adj = build_adj(n, edges)
        adj_bits = build_adj_bit(n, edges)
        
        if n > best_n:
            best_g = {'n': n, 'edges': edges[:]}
            best_n = n
        
        # Extension loop
        extended = True
        while extended and time.time() < deadline - 15:
            result = try_extend(n, adj, adj_bits, edges, max_restarts=5)
            if result is None:
                extended = False
            else:
                n, adj, adj_bits, edges = result
                if n > best_n:
                    best_g = {'n': n, 'edges': edges[:]}
                    best_n = n
        
        # Local search polish to escape local optima
        if time.time() < deadline - 10:
            polished_n, polished_adj, polished_adj_bits, polished_edges = local_search_polish(
                n, adj, adj_bits, edges, time_limit=8
            )
            if polished_n > best_n:
                best_g = {'n': polished_n, 'edges': polished_edges}
                best_n = polished_n
            else:
                # Continue extending from polished state if possible
                n2, adj2, adj_bits2, edges2 = polished_n, polished_adj, polished_adj_bits, polished_edges
                extended2 = True
                while extended2 and time.time() < deadline - 5:
                    result = try_extend(n2, adj2, adj_bits2, edges2, max_restarts=5)
                    if result is None:
                        extended2 = False
                    else:
                        n2, adj2, adj_bits2, edges2 = result
                        if n2 > best_n:
                            best_g = {'n': n2, 'edges': edges2[:]}
                            best_n = n2
        
        if best_n >= 35:
            break  # Can't do much better than 35
    
    if best_g is None:
        return {'n': 1, 'edges': []}
    
    # Final verification
    if not is_valid(best_g['n'], best_g['edges']):
        return {'n': 1, 'edges': []}
    
    return best_g
