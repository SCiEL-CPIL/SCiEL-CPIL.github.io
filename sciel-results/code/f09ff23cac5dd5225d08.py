import time
import random
import copy
import sys

# Increase recursion limit for deeper backtracking if needed
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = 115.0

    # 1. Seed: Greenwood-Gleason colouring of K16 (Clebsch) doubled to n=32
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    def get_seed_graph(n):
        graph = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                graph[i][j] = c
                graph[j][i] = c
        return graph

    current_n = 32
    graph = get_seed_graph(current_n)
    best_n = current_n
    best_graph = copy.deepcopy(graph)

    def is_valid_graph(g, n):
        """Check if the graph has no monochromatic triangles."""
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = g[i][j]
                for k in range(j + 1, n):
                    if g[i][k] == c_ij and g[j][k] == c_ij:
                        return False
        return True

    def add_vertex_with_cp(g, n):
        """
        Attempts to add a new vertex at index n to graph g (size n).
        Uses Constraint Propagation (AC-3) and Backtracking.
        Returns True if successful (g is updated to size n+1), False otherwise.
        """
        # Domains for edges (i, n)
        # dom[i] is a set of possible colors for edge (i, n)
        doms = [set([0, 1, 2, 3]) for _ in range(n)]
        
        # We need to enforce constraints:
        # For any i < j < n, if g[i][j] == c, then not (dom[i] contains c AND dom[j] contains c)
        # This is a binary constraint.
        
        # AC-3 Implementation
        # Queue of arcs (i, j) meaning "check if dom[i] is consistent with dom[j]"
        # The constraint between i and j is: 
        # There must exist a color c in dom[i] and d in dom[j] such that NOT (g[i][j] == c == d)
        # Actually, the constraint is: We cannot have c_i == c_j == g[i][j].
        # So for a fixed g[i][j] = C, the pair (c_i, c_j) is forbidden if c_i == C and c_j == C.
        # This means if C is in dom[i], then C must be removed from dom[j] ONLY IF C is the only option in dom[i]? No.
        # The constraint is satisfied if there is ANY assignment.
        # AC-3 removes values from dom[i] that have no support in dom[j].
        # Value c in dom[i] has support in dom[j] if there exists d in dom[j] such that (c, d) is a valid pair.
        # (c, d) is invalid ONLY IF c == g[i][j] AND d == g[i][j].
        # So, c in dom[i] is unsupported ONLY IF:
        # 1. c == g[i][j]
        # 2. AND d in dom[j] implies d == g[i][j] for all d? No.
        # If c == g[i][j], then we need some d in dom[j] such that d != g[i][j].
        # So c is unsupported if dom[j] == {g[i][j]}.
        
        # This is a very weak constraint for AC-3 to prune much unless domains are very small.
        # However, it helps.
        
        queue = []
        for i in range(n):
            for j in range(i + 1, n):
                queue.append((i, j))
                queue.append((j, i))
        
        def revise(i, j):
            """Check if any value in dom[i] is unsupported by dom[j] given g[i][j]."""
            c_g = g[i][j]
            removed = False
            # If c_g is in dom[i], it is supported if dom[j] has any value other than c_g.
            # So if dom[j] is a subset of {c_g}, then c_g in dom[i] is unsupported.
            if c_g in doms[i]:
                if len(doms[j]) == 1 and c_g in doms[j]:
                    doms[i].remove(c_g)
                    removed = True
            return removed
        
        while queue:
            i, j = queue.pop(0)
            if revise(i, j):
                if not doms[i]:
                    return False
                # Add all neighbors of i (except j) to queue
                for k in range(n):
                    if k != i and k != j:
                        queue.append((k, i))
                        
        # If we get here, domains are consistent (arc consistent).
        # Now we need to find a full assignment.
        # We use backtracking search.
        
        assignments = [-1] * n
        # Order variables by domain size (MRV)
        # But the constraints are static. 
        # We can just pick the variable with the smallest domain.
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # Find unassigned variable with smallest domain
            min_size = 5
            var = -1
            for i in range(n):
                if assignments[i] == -1:
                    if len(doms[i]) < min_size:
                        min_size = len(doms[i])
                        var = i
            if var == -1:
                return True # All assigned
            
            # Try each value in doms[var]
            # Heuristic: Try values that appear less frequently in neighbors?
            values = list(doms[var])
            random.shuffle(values)
            
            for c in values:
                assignments[var] = c
                # Check consistency with all previously assigned neighbors
                valid = True
                for k in range(n):
                    if k != var and assignments[k] != -1:
                        if g[k][var] == c and assignments[k] == c:
                            valid = False
                            break
                if valid:
                    if backtrack(idx + 1):
                        return True
                assignments[var] = -1
                
            return False
        
        if backtrack(0):
            # Success
            for i in range(n):
                row = g[i]
                row.append(assignments[i])
            g.append([0] * (n + 1))
            last_row = g[n]
            for i in range(n):
                last_row[i] = assignments[i]
            last_row[n] = 0
            return True
        return False

    attempts = 0
    max_attempts = 50000
    consecutive_failures = 0
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        if add_vertex_with_cp(graph, current_n):
            current_n += 1
            best_n = current_n
            best_graph = copy.deepcopy(graph)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            # Kick strategy
            if consecutive_failures > 5:
                old_graph = copy.deepcopy(graph)
                old_n = current_n
                num_edges = current_n * (current_n - 1) // 2
                num_flips = max(1, num_edges // 20)
                
                for _ in range(num_flips):
                    i = random.randint(0, current_n - 1)
                    j = random.randint(0, current_n - 1)
                    if i != j:
                        graph[i][j] = random.randint(0, 3)
                        graph[j][i] = graph[i][j]
                
                if is_valid_graph(graph, current_n):
                    pass # Keep the kick
                else:
                    graph = old_graph
                    current_n = old_n

    n = best_n
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(best_graph[i][j])
            
    return {'n': n, 'colors': colors}
