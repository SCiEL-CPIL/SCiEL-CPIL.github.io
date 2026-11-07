"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest(coloring, k):
    """Check that for every pair of colors (a,b), the induced subgraph is a forest."""
    n = P['n']
    # Group vertices by color
    color_groups = defaultdict(list)
    for i in range(n):
        color_groups[coloring[i]].append(i)
    
    # For each pair of colors, check the induced subgraph is a forest
    colors_used = list(color_groups.keys())
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            a, b = colors_used[i], colors_used[j]
            va = set(color_groups[a])
            vb = set(color_groups[b])
            # Build edges within va ∪ vb
            parent = {}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            for v in va | vb:
                parent[v] = v
            for u in va | vb:
                for v in _adj[u]:
                    if v > u and v in (va | vb):
                        if not union(u, v):
                            return False
    return True

def _is_valid(coloring, k):
    """Check all constraints: adjacent differ, and bicolored subgraphs are forests."""
    n = P['n']
    # Check adjacent vertices have different colors
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    # Check bicolored subgraphs are forests
    return _check_bicolor_forest(coloring, k)

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    adj = _build_adj()
    n = P['n']
    
    # Precompute adjacency sets for fast lookup
    adj_sets = [set(adj[i]) for i in range(n)]
    
    # Try increasing number of colors
    for k in range(2, 15):
        if time.time() > deadline:
            break
        best = None
        # Use randomized backtracking / local search
        # Try multiple random restarts
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            # Random initial coloring with k colors
            coloring = [random.randint(0, k - 1) for _ in range(n)]
            
            # Local search: try to fix violations
            improved = True
            stall_count = 0
            while improved and time.time() < deadline:
                improved = False
                # Check validity
                if _is_valid(coloring, k):
                    best = coloring[:]
                    break
                # Find a vertex to change
                # Prioritize vertices in conflicts
                for v in range(n):
                    if time.time() > deadline:
                        break
                    # Check if v is in a conflict
                    in_conflict = False
                    # Check adjacent same color
                    for u in adj[v]:
                        if coloring[u] == coloring[v]:
                            in_conflict = True
                            break
                    if not in_conflict:
                        # Check bicolored cycle involving v
                        cv = coloring[v]
                        for u in adj[v]:
                            cu = coloring[u]
                            if cu == cv:
                                continue
                            # Check if edge (v,u) creates a bicolored cycle
                            # This is expensive; approximate by checking if v is in a cycle
                            # For now, just try changing
                            pass
                    if in_conflict:
                        # Try changing v to a different color
                        for new_c in range(k):
                            if new_c == coloring[v]:
                                continue
                            # Check if new_c conflicts with neighbors
                            ok = True
                            for u in adj[v]:
                                if coloring[u] == new_c:
                                    ok = False
                                    break
                            if ok:
                                coloring[v] = new_c
                                improved = True
                                stall_count = 0
                                break
                        if improved:
                            break
                    elif not in_conflict:
                        # Not in obvious conflict, but might be in bicolored cycle
                        # Try small perturbation
                        if stall_count < 5:
                            new_c = random.randint(0, k - 1)
                            if new_c != coloring[v]:
                                ok = True
                                for u in adj[v]:
                                    if coloring[u] == new_c:
                                        ok = False
                                        break
                                if ok:
                                    coloring[v] = new_c
                                    improved = True
                                    break
                if not improved:
                    stall_count += 1
                    if stall_count > 20:
                        # Random perturbation
                        v = random.randint(0, n - 1)
                        new_c = random.randint(0, k - 1)
                        if new_c != coloring[v]:
                            ok = True
                            for u in adj[v]:
                                if coloring[u] == new_c:
                                    ok = False
                                    break
                            if ok:
                                coloring[v] = new_c
                                improved = True
                    if stall_count > 50:
                        break
            
            if best is not None:
                break
        if best is not None:
            # Verify
            if _is_valid(best, k):
                return [int(c) for c in best]
    
    # Fallback: return a valid 50-color solution
    return [int(i) for i in range(n)]

# Precompute adjacency for module-level use
_adj = _build_adj()
# EVOLVE-BLOCK-END
