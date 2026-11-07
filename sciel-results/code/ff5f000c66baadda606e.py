"""SAT-based approach: model bicolored-forest constraint via forbidding short bicolored cycles."""
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adjacency(edges):
    adj = [set() for _ in range(P['n'])]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_all_cycles_up_to_len(adj, n, max_len):
    """Find all simple cycles of length <= max_len in the graph."""
    cycles = []
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    for start in range(n):
        # DFS to find cycles starting from 'start'
        # Only visit vertices >= start to avoid duplicates
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > max_len:
                continue
            for nb in adj_list[node]:
                if nb < start:
                    continue
                if nb == start and len(path) >= 3:
                    cycles.append(list(path))
                elif nb not in path:
                    stack.append((nb, path + [nb]))
    return cycles


def _check_bicolored_forest(coloring, adj, n):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    colors_used = set(coloring)
    color_list = list(colors_used)
    m = len(color_list)
    
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    for i in range(m):
        for j in range(i + 1, m):
            c1, c2 = color_list[i], color_list[j]
            parent = list(range(n))
            has_cycle = False
            for u in range(n):
                if coloring[u] not in (c1, c2):
                    continue
                for v in adj[u]:
                    if v <= u:
                        continue
                    if coloring[v] not in (c1, c2):
                        continue
                    ru = find(u, parent)
                    rv = find(v, parent)
                    if ru == rv:
                        has_cycle = True
                        break
                    parent[ru] = rv
                if has_cycle:
                    break
            if has_cycle:
                return False
    return True


def _try_sat(k, adj, n, edges, deadline):
    """Try to find a valid k-coloring using SAT solver."""
    try:
        from pysat.solvers import Solver
    except ImportError:
        return None
    
    # Variable encoding: x[v][c] = 1 means vertex v has color c
    # x[v][c] -> variable index = v * k + c + 1
    # x[v][c] -> var(v, c) = v * k + c + 1
    
    def var(v, c):
        return v * k + c + 1
    
    solver = Solver()
    
    # Constraint 1: Each vertex gets exactly one color
    for v in range(n):
        # At least one color
        solver.add_clause([var(v, c) for c in range(k)])
        # At most one color
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                solver.add_clause([-var(v, c1), -var(v, c2)])
    
    # Constraint 2: Adjacent vertices must have different colors
    for u, v in edges:
        for c in range(k):
            solver.add_clause([-var(u, c), -var(v, c)])
    
    # Constraint 3: No bicolored cycle
    # For each cycle in the graph, for each pair of colors (c1, c2),
    # not all vertices in the cycle can be colored with only c1 or c2
    # i.e., at least one vertex must have a color different from both c1 and c2
    
    # Find all cycles up to a reasonable length
    # The graph has 75 edges and 50 vertices, so it's sparse
    # Let's find cycles up to length 8
    cycles = _find_all_cycles_up_to_len(adj, n, 8)
    
    for cycle in cycles:
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                # For this cycle to be bicolored with colors c1 and c2,
                # every vertex in the cycle must have color c1 or c2
                # We forbid this by requiring at least one vertex to NOT have c1 or c2
                # Clause: OR over vertices in cycle of (NOT x[v][c1] AND NOT x[v][c2])
                # This is not a single clause, so we need auxiliary variables
                # Alternative: for each vertex v in cycle, add clause (NOT x[v][c1] OR NOT x[v][c2]) is wrong
                # We need: NOT (AND over v in cycle of (x[v][c1] OR x[v][c2]))
                # = OR over v in cycle of (NOT x[v][c1] AND NOT x[v][c2])
                # Each (NOT x[v][c1] AND NOT x[v][c2]) is not a clause
                # Use Tseitin: introduce aux var a_v for (NOT x[v][c1] AND NOT x[v][c2])
                # Then clause: (a_v1 OR a_v2 OR ... OR a_vL)
                # And for each v: a_v -> (NOT x[v][c1]), a_v -> (NOT x[v][c2]), (x[v][c1] OR x[v][c2] OR NOT a_v)
                
                L = len(cycle)
                if L < 3:
                    continue
                
                # Introduce auxiliary variables for each vertex in the cycle
                aux_vars = []
                for v in cycle:
                    a = var(v, c1) * k + var(v, c2) + 1  # unique aux var
                    # Actually, let's use a simpler encoding
                    # a = (v * k + c1) * k + c2 + 1
                    a = (v * k + c1) * k + c2 + 1
                    aux_vars.append(a)
                    # a -> NOT x[v][c1]: (-a OR -x[v][c1])
                    solver.add_clause([-a, -var(v, c1)])
                    # a -> NOT x[v][c2]: (-a OR -x[v][c2])
                    solver.add_clause([-a, -var(v, c2)])
                    # (x[v][c1] OR x[v][c2] OR NOT a)
                    solver.add_clause([var(v, c1), var(v, c2), -a])
                
                # At least one aux var must be true
                solver.add_clause(aux_vars)
    
    # Solve
    result = solver.solve()
    if not result:
        solver.delete()
        return None
    
    model = solver.get_model()
    coloring = [0] * n
    for v in range(n):
        for c in range(k):
            if model[var(v, c) - 1] == 1:
                coloring[v] = c
                break
    
    solver.delete()
    
    # Verify
    valid = True
    for u, v in edges:
        if coloring[u] == coloring[v]:
            valid = False
            break
    if valid and _check_bicolored_forest(coloring, adj, n):
        return coloring
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges)
    
    # Try k from 3 upward using SAT
    for k in range(3, 8):
        if time.time() > deadline - 10.0:
            break
        
        result = _try_sat(k, adj, n, edges, deadline)
        if result is not None:
            return [int(c) for c in result]
    
    # Fallback: greedy coloring
    coloring = [0] * n
    for v in range(n):
        used = set()
        for nb in adj[v]:
            if nb < v:
                used.add(coloring[nb])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    
    # Verify and return
    valid = True
    for u, v in edges:
        if coloring[u] == coloring[v]:
            valid = False
            break
    if valid and _check_bicolored_forest(coloring, adj, n):
        return [int(c) for c in coloring]
    
    # Final fallback
    return [int(i) for i in range(n)]


# EVOLVE-BLOCK-END
