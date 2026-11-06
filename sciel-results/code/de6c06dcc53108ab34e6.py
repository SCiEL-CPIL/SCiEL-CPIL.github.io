import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin for 120s limit

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

    def find_indep_sets_5(n, adj):
        # Returns list of tuples of 5 vertices that form an independent set
        # Only used if alpha >= 5
        # Simple backtracking
        indep_sets = []
        
        def backtrack(start, current):
            if len(current) == 5:
                indep_sets.append(tuple(sorted(current)))
                return
            for v in range(start, n):
                # Check if v is independent from all in current
                if all(v not in adj[u] for u in current):
                    current.append(v)
                    backtrack(v + 1, current)
                    current.pop()
                    if len(indep_sets) > 1000: # Cap to avoid explosion
                        return

        backtrack(0, [])
        return indep_sets

    def is_valid(g):
        n = g['n']
        edges = g['edges']
        adj = build_adj(n, edges)
        
        # Check K4
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    for k in common:
                        if k > j:
                            # Check if i,j,k is triangle, and if any 4th vertex connects to all 3
                            # Actually, just check if there is a 4th vertex l such that l connected to i,j,k
                            # We can just check if the subgraph on {i,j,k} + neighbors has K4
                            # Simpler: Check all 4-combinations? O(n^4) is 35^4 ~ 1.5M. Acceptable.
                            pass
        
        # Efficient K4 check
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    for k in range(j+1, n):
                        if k in adj[i] and k in adj[j]:
                            # i,j,k is triangle. Check for 4th
                            common = adj[i] & adj[j] & adj[k]
                            if common:
                                return False
                                
        # Check alpha >= 6
        # Use a simple branch and bound or just check if find_indep_sets_5 finds many?
        # If we find an independent set of 6, return False.
        # We can reuse find_indep_sets_5 logic but for size 6?
        # Or just check max independent set size.
        # For n<=35, we can use a simple recursive check for independent set of size 6.
        
        def has_indep_6(start, current):
            if len(current) == 6:
                return True
            # Pruning: if start + (6 - len(current)) > n, return False
            if start + (6 - len(current)) > n:
                return False
            
            for v in range(start, n):
                if all(v not in adj[u] for u in current):
                    if has_indep_6(v + 1, current + [v]):
                        return True
            return False
        
        if has_indep_6(0, []):
            return False
            
        return True

    def walk_sat(n, clauses, max_iters=2000):
        # clauses: list of lists of (var, sign)
        # var is 0..n-1, sign is 1 or -1
        # x is list of bools
        x = [random.choice([True, False]) for _ in range(n)]
        
        def is_sat(clause):
            for var, sign in clause:
                val = x[var]
                if sign == 1:
                    if val: return True
                else:
                    if not val: return True
            return False

        for _ in range(max_iters):
            unsat = [c for c in clauses if not is_sat(c)]
            if not unsat:
                return x
            
            clause = random.choice(unsat)
            # Pick a variable to flip
            # 50% chance pick from clause, 50% random
            if random.random() < 0.5:
                var, _ = random.choice(clause)
            else:
                var = random.randint(0, n - 1)
            
            x[var] = not x[var]
            
        return None

    # 1. Initialize with P29
    G = make_paley(29)
    if G is None or not is_valid(G):
        G = make_paley(17)
        if G is None or not is_valid(G):
            # Fallback to trivial valid graph
            return {'n': 1, 'edges': []}

    n = G['n']
    edges = G['edges']
    adj = build_adj(n, edges)
    best_g = {'n': n, 'edges': edges}

    # 2. Extension Loop
    while time.time() < deadline:
        # Precompute constraints for new vertex
        triangles = find_triangles(n, adj)
        # Convert to sets for fast lookup in greedy? 
        # For walk_sat, we need clauses.
        
        # Check if we need independent set constraints
        # Only if alpha(G) >= 5
        # We can check if there are independent sets of size 5
        indep_sets_5 = find_indep_sets_5(n, adj)
        
        clauses = []
        
        # Triangle clauses: not (x_u and x_v and x_w)
        for tri in triangles:
            u, v, w = tri
            clauses.append([(u, -1), (v, -1), (w, -1)])
            
        # Independent set clauses: x_u or x_v or x_w or x_x or x_y
        for ind in indep_sets_5:
            u, v, w, x, y = ind
            clauses.append([(u, 1), (v, 1), (w, 1), (x, 1), (y, 1)])
            
        # Try to find assignment
        assignment = None
        
        # If no clauses, any assignment works (e.g., all False)
        if not clauses:
            assignment = [False] * n
        else:
            # Try WalkSAT
            assignment = walk_sat(n, clauses, max_iters=5000)
            
            # If WalkSAT fails, try a greedy heuristic specific to triangle constraints
            if assignment is None and not indep_sets_5:
                # Greedy: Start with all True, remove vertices causing triangle violations
                # x[i] = True means edge exists
                x = [True] * n
                changed = True
                while changed:
                    changed = False
                    for tri in triangles:
                        u, v, w = tri
                        if x[u] and x[v] and x[w]:
                            # Remove one
                            idx = random.choice([u, v, w])
                            x[idx] = False
                            changed = True
                assignment = x
                
        if assignment is not None:
            # Verify assignment
            valid = True
            for c in clauses:
                sat = False
                for var, sign in c:
                    val = assignment[var]
                    if sign == 1:
                        if val: sat = True; break
                    else:
                        if not val: sat = True; break
                if not sat:
                    valid = False
                    break
            
            if valid:
                # Add vertex
                new_n = n + 1
                new_edges = edges[:]
                for i in range(n):
                    if assignment[i]:
                        new_edges.append([i, new_n - 1])
                
                n = new_n
                edges = new_edges
                adj = build_adj(n, edges)
                best_g = {'n': n, 'edges': edges}
            else:
                break # Should not happen if walk_sat is correct, but safety
        else:
            break # Cannot extend

    return best_g

# EVOLVE-BLOCK-END
