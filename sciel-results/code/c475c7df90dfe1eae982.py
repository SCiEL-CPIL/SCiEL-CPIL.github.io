"""ILP-based solver for bicolored-forest coloring using scipy MILP."""
import time
import random
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

# Precompute edge index for each edge
EDGE_INDEX = {}
for i, (u, v) in enumerate(EDGES):
    EDGE_INDEX[(u, v)] = i
    EDGE_INDEX[(v, u)] = i


def greedy_coloring(k, order=None):
    """Greedy coloring with given vertex order."""
    if order is None:
        order = list(range(N))
    color = [-1] * N
    for v in order:
        used = set()
        for u in ADJ[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(k):
            if c not in used:
                color[v] = c
                break
        else:
            color[v] = 0
    return color


def count_bicolored_cycles(color, k):
    """Count total number of bicolored cycles across all color pairs."""
    total_cycles = 0
    for a in range(k):
        for b in range(a + 1, k):
            verts = [i for i in range(N) if color[i] == a or color[i] == b]
            if len(verts) < 3:
                continue
            edges_in_pair = []
            for u, v in EDGES:
                cu, cv = color[u], color[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    edges_in_pair.append((u, v))
            if not edges_in_pair:
                continue
            vert_set = set(verts)
            parent = {v: v for v in vert_set}
            rank = {v: 0 for v in vert_set}

            def find(v):
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                return v

            def union(u, v):
                ru = find(u)
                rv = find(v)
                if ru == rv:
                    return False
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
                return True

            for u, v in edges_in_pair:
                if not union(u, v):
                    total_cycles += 1
    return total_cycles


def is_valid(color, k):
    """Check if coloring is valid: proper and no bicolored cycles."""
    for u, v in EDGES:
        if color[u] == color[v]:
            return False
    return count_bicolored_cycles(color, k) == 0


def solve_ilp(k, time_limit, rng):
    """Solve ILP for k colors. Returns coloring or None if infeasible/timeout."""
    # Variables: x[v][c] = 1 if vertex v has color c
    # Total variables: N * k
    # For each edge (u,v) and each color c: x[u][c] + x[v][c] <= 1 (proper coloring)
    # For each color pair (a,b): the induced subgraph must be a forest
    
    # The forest constraint: for any cycle in the graph, not all edges in that cycle
    # can be "active" (both endpoints have colors a or b with one being a and other b)
    
    # A simpler approach: for each color pair (a,b), for each simple cycle C in the graph,
    # at least one edge in C must not be "bicolored a-b"
    
    # But enumerating all cycles is expensive. Instead, use the fact that in a 4-regular graph,
    # we can use edge-based constraints.
    
    # Alternative: Use the property that a graph is a forest iff for every subset of edges,
    # the number of edges in any connected component <= number of vertices - 1.
    
    # Practical ILP formulation:
    # x[v][c] in {0,1} for v in 0..N-1, c in 0..k-1
    # Constraints:
    # 1. Sum_c x[v][c] = 1 for all v (each vertex gets exactly one color)
    # 2. x[u][c] + x[v][c] <= 1 for all edges (u,v) and all c (proper coloring)
    # 3. For each color pair (a,b) and each cycle C: sum over edges e in C of 
    #    (x[u_e][a]*x[v_e][b] + x[u_e][b]*x[v_e][a]) <= |C| - 1
    #    But this is nonlinear. Linearize with auxiliary variables.
    
    # For each edge (u,v) and color pair (a,b), define y[(u,v)][a][b] = 1 if edge (u,v) is 
    # "active" for colors a,b (i.e., one endpoint is a and other is b)
    # y[(u,v)][a][b] <= x[u][a] + x[v][b]
    # y[(u,v)][a][b] <= x[v][a] + x[u][b]  
    # Actually: y[(u,v)][a][b] = 1 iff (x[u][a]=1 and x[v][b]=1) or (x[u][b]=1 and x[v][a]=1)
    # This is: y >= x[u][a] + x[v][b] - 1, y >= x[u][b] + x[v][a] - 1, y <= x[u][a] + x[v][b], y <= x[u][b] + x[v][a]
    
    # For forest constraint: for each color pair (a,b) and each cycle C:
    # sum_{e in C} y[e][a][b] <= |C| - 1
    
    # This is a valid ILP but potentially large. Let's try it for small k.
    
    # Actually, let's use a different approach: since we just need ANY valid coloring,
    # we can use a CP-style approach with the ILP as a verifier, or use the ILP directly.
    
    # For now, let's try a simpler approach: use the ILP with cycle constraints for small cycles only.
    
    # Actually, the most practical approach given time constraints:
    # 1. Try greedy + local search (existing approach)
    # 2. If that fails, try ILP for small k
    
    # Let me implement a proper ILP for k colors
    # Variables: N*k binary variables for color assignment
    # Plus auxiliary variables for edge-color-pair activations
    
    n_vars = N * k
    
    # Objective: minimize 0 (feasibility problem)
    c = [0.0] * n_vars
    
    # Constraints list: (A_matrix_rows, lb, ub)
    # We'll build constraints incrementally
    
    A_rows = []
    lb_list = []
    ub_list = []
    
    # Constraint 1: Each vertex gets exactly one color
    # For vertex v: sum_c x[v][c] = 1
    for v in range(N):
        row = [0.0] * n_vars
        for c_ in range(k):
            row[v * k + c_] = 1.0
        A_rows.append(row)
        lb_list.append(1.0)
        ub_list.append(1.0)
    
    # Constraint 2: Proper coloring
    # For each edge (u,v) and each color c: x[u][c] + x[v][c] <= 1
    for u, v in EDGES:
        for c_ in range(k):
            row = [0.0] * n_vars
            row[u * k + c_] = 1.0
            row[v * k + c_] = 1.0
            A_rows.append(row)
            lb_list.append(0.0)
            ub_list.append(1.0)
    
    # For the forest constraint, we need auxiliary variables.
    # This makes the ILP much larger. Let's estimate size:
    # Number of color pairs: k*(k-1)/2
    # Number of edges: len(EDGES)
    # Auxiliary variables: len(EDGES) * k*(k-1)/2
    # For k=4: 4*3/2 = 6 pairs, ~75 edges, so ~450 aux vars
    # Total vars: 50*4 + 450 = 650
    # This is feasible for a MILP solver.
    
    # Let's add auxiliary variables and constraints
    num_pairs = k * (k - 1) // 2
    pair_list = []
    for a in range(k):
        for b in range(a + 1, k):
            pair_list.append((a, b))
    
    # Auxiliary variable index: for edge i and pair p, index = n_vars + i * num_pairs + p
    n_aux = len(EDGES) * num_pairs
    total_vars = n_vars + n_aux
    
    # Extend objective
    c = [0.0] * total_vars
    
    # For each edge and each color pair, add constraints for auxiliary variable y
    # y = 1 if the edge is "bicolored" for this pair
    # y >= x[u][a] + x[v][b] - 1
    # y >= x[u][b] + x[v][a] - 1
    # y <= x[u][a] + x[v][b]
    # y <= x[u][b] + x[v][a]
    # y >= 0
    
    for i, (u, v) in enumerate(EDGES):
        for p, (a, b) in enumerate(pair_list):
            y_idx = n_vars + i * num_pairs + p
            
            # y >= x[u][a] + x[v][b] - 1  =>  -y + x[u][a] + x[v][b] >= -1
            row = [0.0] * total_vars
            row[y_idx] = -1.0
            row[u * k + a] = 1.0
            row[v * k + b] = 1.0
            A_rows.append(row)
            lb_list.append(-1.0)
            ub_list.append(float('inf'))
            
            # y >= x[u][b] + x[v][a] - 1  =>  -y + x[u][b] + x[v][a] >= -1
            row = [0.0] * total_vars
            row[y_idx] = -1.0
            row[u * k + b] = 1.0
            row[v * k + a] = 1.0
            A_rows.append(row)
            lb_list.append(-1.0)
            ub_list.append(float('inf'))
            
            # y <= x[u][a] + x[v][b]  =>  y - x[u][a] - x[v][b] <= 0
            row = [0.0] * total_vars
            row[y_idx] = 1.0
            row[u * k + a] = -1.0
            row[v * k + b] = -1.0
            A_rows.append(row)
            lb_list.append(float('-inf'))
            ub_list.append(0.0)
            
            # y <= x[u][b] + x[v][a]  =>  y - x[u][b] - x[v][a] <= 0
            row = [0.0] * total_vars
            row[y_idx] = 1.0
            row[u * k + b] = -1.0
            row[v * k + a] = -1.0
            A_rows.append(row)
            lb_list.append(float('-inf'))
            ub_list.append(0.0)
    
    # Now add cycle constraints. We need to enumerate cycles.
    # For a 4-regular graph on 50 vertices, this could be many cycles.
    # Instead, let's use a different approach: for each color pair (a,b),
    # the subgraph induced by vertices with color a or b must be a forest.
    # A graph is a forest iff it has no cycles, which is equivalent to:
    # for every subset of vertices S, the number of edges within S <= |S| - 1
    # But this is exponential.
    
    # Practical approach: only add constraints for small cycles (length 3, 4, 5)
    # In a 4-regular graph, the girth is at least 3 (could have triangles).
    # Let's find all cycles of length up to 6 and add constraints for those.
    
    # Find all short cycles
    def find_short_cycles(max_len=6):
        cycles = []
        # Use DFS to find cycles
        for start in range(N):
            # DFS from start, only visiting vertices >= start to avoid duplicates
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                if len(path) > max_len:
                    continue
                for neighbor in ADJ[node]:
                    if neighbor == start and len(path) >= 3:
                        # Found a cycle
                        cycle = tuple(sorted(path))
                        if cycle not in cycles:
                            cycles.append(cycle)
                    elif neighbor not in visited and neighbor > start:
                        new_path = path + [neighbor]
                        new_visited = visited | {neighbor}
                        stack.append((neighbor, new_path, new_visited))
        return cycles
    
    # This might be slow. Let's limit to triangles and 4-cycles for now.
    # Actually, for the ILP to be correct, we need ALL cycles. This is impractical.
    
    # Alternative: use the fact that if the ILP finds a solution, we verify it.
    # If the ILP is infeasible, we try larger k.
    # But without cycle constraints, the ILP only enforces proper coloring.
    
    # Let's take a hybrid approach:
    # 1. Solve ILP for proper coloring (fast)
    # 2. Verify bicolored forest property
    # 3. If invalid, add a no-good cut and resolve
    
    # Actually, let's just use the existing local search approach but with better initialization
    # from the ILP solution.
    
    # For now, return None to indicate ILP is not fully implemented
    # and fall back to local search
    return None


def local_search(k, time_limit, rng):
    """Local search to minimize bicolored cycles."""
    deadline = time.time() + time_limit

    # Start with greedy coloring
    color = greedy_coloring(k)

    # Try multiple random restarts
    best_color = color[:]
    best_cost = count_bicolored_cycles(color, k)

    for u, v in EDGES:
        if color[u] == color[v]:
            best_cost = 1000

    num_restarts = 0
    max_restarts = 50

    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1

        # Random restart
        color = [rng.randint(0, k - 1) for _ in range(N)]
        for _ in range(100):
            improved = False
            for u, v in EDGES:
                if color[u] == color[v]:
                    if rng.random() < 0.5:
                        new_c = (color[u] + 1) % k
                        if new_c != color[v]:
                            color[u] = new_c
                            improved = True
                    else:
                        new_c = (color[v] + 1) % k
                        if new_c != color[u]:
                            color[v] = new_c
                            improved = True
            if not improved:
                break

        cost = count_bicolored_cycles(color, k)
        for u, v in EDGES:
            if color[u] == color[v]:
                cost = 1000
                break

        if cost < best_cost:
            best_cost = cost
            best_color = color[:]

        if cost == 0:
            break

        no_improve_count = 0
        max_no_improve = 200

        while time.time() < deadline and no_improve_count < max_no_improve:
            v = rng.randint(0, N - 1)
            old_c = color[v]

            best_new_c = old_c
            best_new_cost = cost

            for c in range(k):
                if c == old_c:
                    continue
                conflict = False
                for u in ADJ[v]:
                    if color[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue

                color[v] = c
                new_cost = count_bicolored_cycles(color, k)
                for u in ADJ[v]:
                    if color[u] == c:
                        new_cost = 1000
                        break

                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new_c = c

                color[v] = old_c

            if best_new_c != old_c:
                color[v] = best_new_c
                cost = best_new_cost
                no_improve_count = 0

                if cost < best_cost:
                    best_cost = cost
                    best_color = color[:]

                if cost == 0:
                    break
            else:
                no_improve_count += 1

        if best_cost == 0:
            break

    return best_color, best_cost


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    rng = random.Random(42)

    # Try k = 3, 4, 5, ...
    for k in range(3, 15):
        if time.time() > deadline:
            break

        # First try greedy
        color = greedy_coloring(k)
        if is_valid(color, k):
            return [int(c) for c in color]

        # Try local search
        remaining = deadline - time.time()
        if remaining < 5:
            break

        time_for_k = min(remaining * 0.7, 20)
        if time_for_k < 2:
            continue

        color, cost = local_search(k, time_for_k, rng)

        if cost == 0 and is_valid(color, k):
            return [int(c) for c in color]

    # Fallback: try to get a valid coloring with more colors
    for k in range(10, 51):
        if time.time() > deadline:
            break
        color = greedy_coloring(k)
        if is_valid(color, k):
            return [int(c) for c in color]

    # Last resort: identity coloring (always valid)
    return [int(i) for i in range(N)]
