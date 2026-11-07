"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def verify(coloring):
        """Verify a coloring is valid."""
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        colors_used = set(coloring)
        for a in colors_used:
            for b in colors_used:
                if a >= b:
                    continue
                verts = set()
                for v in range(n):
                    if coloring[v] == a or coloring[v] == b:
                        verts.add(v)
                if len(verts) > 1:
                    visited = set()
                    for start in verts:
                        if start not in visited:
                            queue = deque([start])
                            visited.add(start)
                            ec = 0
                            vc = 0
                            while queue:
                                node = queue.popleft()
                                vc += 1
                                for nb in adj[node]:
                                    if nb in verts:
                                        ec += 1
                                        if nb not in visited:
                                            visited.add(nb)
                                            queue.append(nb)
                            if ec // 2 != vc - 1:
                                return False
        return True

    def count_violations(coloring):
        """Count the number of bicolored cycles (violation count)."""
        violations = 0
        for u, v in edges:
            if coloring[u] == coloring[v]:
                violations += 100  # Heavy penalty for adjacent same-color
        colors_used = set(coloring)
        for a in colors_used:
            for b in colors_used:
                if a >= b:
                    continue
                verts = set()
                for v in range(n):
                    if coloring[v] == a or coloring[v] == b:
                        verts.add(v)
                if len(verts) > 1:
                    visited = set()
                    for start in verts:
                        if start not in visited:
                            queue = deque([start])
                            visited.add(start)
                            ec = 0
                            vc = 0
                            while queue:
                                node = queue.popleft()
                                vc += 1
                                for nb in adj[node]:
                                    if nb in verts:
                                        ec += 1
                                        if nb not in visited:
                                            visited.add(nb)
                                            queue.append(nb)
                            if ec // 2 != vc - 1:
                                violations += 1
        return violations

    # Local search / hill climbing approach
    def local_search(k, max_iters=50000):
        """Local search to find valid k-coloring."""
        random.seed(42)
        best_coloring = None
        best_cost = float('inf')
        
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Random initial coloring
            coloring = [random.randint(0, k - 1) for _ in range(n)]
            cost = count_violations(coloring)
            
            if cost == 0:
                return coloring
            
            improved = True
            iter_count = 0
            while improved and iter_count < max_iters:
                if time.time() > deadline:
                    break
                improved = False
                iter_count += 1
                
                # Try changing each vertex's color
                for v in range(n):
                    old_color = coloring[v]
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        # Check if new color is valid for this vertex (no same-colored neighbor)
                        valid = True
                        for u in adj[v]:
                            if coloring[u] == new_color:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        coloring[v] = new_color
                        new_cost = count_violations(coloring)
                        
                        if new_cost < cost:
                            cost = new_cost
                            improved = True
                            if cost == 0:
                                return coloring
                        else:
                            coloring[v] = old_color
                
                # If no improvement, try random restart with perturbation
                if not improved:
                    # Perturb a few random vertices
                    for _ in range(3):
                        v = random.randint(0, n - 1)
                        coloring[v] = random.randint(0, k - 1)
                    cost = count_violations(coloring)
                    if cost == 0:
                        return coloring
            
            if cost < best_cost:
                best_cost = cost
                best_coloring = coloring[:]
        
        return best_coloring

    # Backtracking with constraint propagation
    def backtrack_solve(k):
        """Backtracking solver with forward checking."""
        color = [-1] * n
        
        # Vertex ordering: use degree-based ordering
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        # DSU for cycle detection
        parent = {}
        rank = {}
        
        # Initialize DSU
        for a in range(k):
            for b in range(a + 1, k):
                parent[(a, b)] = list(range(n))
                rank[(a, b)] = [0] * n
        
        def find_in(par, x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        def bt(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            v = order[idx]
            
            for c in range(k):
                # Check adjacency constraint
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                color[v] = c
                
                # Check and record unions
                union_stack = []
                conflict = False
                for u in adj[v]:
                    cu = color[u]
                    if cu >= 0 and cu != c:
                        a, b = (c, cu) if c < cu else (cu, c)
                        key = (a, b)
                        p = parent[key]
                        r = rank[key]
                        ra = find_in(p, v)
                        rb = find_in(p, u)
                        if ra == rb:
                            conflict = True
                            break
                        if r[ra] < r[rb]:
                            ra, rb = rb, ra
                        p[rb] = ra
                        if r[ra] == r[rb]:
                            r[ra] += 1
                        union_stack.append((key, ra, rb, r[ra]))
                
                if conflict:
                    # Undo unions
                    for key, ra, rb, old_rank in reversed(union_stack):
                        p = parent[key]
                        r = rank[key]
                        p[rb] = rb
                        r[ra] = old_rank
                    color[v] = -1
                    continue
                
                if bt(idx + 1):
                    return True
                
                # Undo unions
                for key, ra, rb, old_rank in reversed(union_stack):
                    p = parent[key]
                    r = rank[key]
                    p[rb] = rb
                    r[ra] = old_rank
                color[v] = -1
            
            return False
        
        if bt(0):
            return color[:]
        return None

    # Try backtracking first for small k
    for k in range(3, 5):
        if time.time() > deadline:
            break
        result = backtrack_solve(k)
        if result is not None:
            if verify(result):
                return [int(x) for x in result]

    # Try local search for k=3,4,5
    for k in range(3, 6):
        if time.time() > deadline:
            break
        result = local_search(k)
        if result is not None:
            if verify(result):
                return [int(x) for x in result]

    # Fallback: greedy with increasing colors
    best_result = None
    for k in range(4, 15):
        if time.time() > deadline:
            break
        color = [-1] * n
        for v in range(n):
            used = set()
            for u in adj[v]:
                if color[u] >= 0:
                    used.add(color[u])
            for c in range(k):
                if c not in used:
                    color[v] = c
                    break
            else:
                color[v] = k - 1

        if verify(color):
            if best_result is None or len(set(color)) < len(set(best_result)):
                best_result = color[:]
            break

    if best_result is not None:
        return [int(x) for x in best_result]

    # Last resort: trivial solution
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
