"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

# Precompute adjacency
_ADJ = [[] for _ in range(P['n'])]
for u, v in P['edges']:
    _ADJ[u].append(v)
    _ADJ[v].append(u)

def _validate(colors):
    """Check proper coloring and no bicolored cycles using union-find."""
    n = P['n']
    # Proper coloring check
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    # Bicolored cycle check: for each pair of colors, induced subgraph must be a forest
    # Use union-find per color pair
    color_pairs = set()
    for u, v in P['edges']:
        cu, cv = colors[u], colors[v]
        if cu != cv:
            color_pairs.add((min(cu, cv), max(cu, cv)))
    
    for c1, c2 in color_pairs:
        # Union-find for vertices colored c1 or c2
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
                return x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False  # cycle detected
            parent[ra] = rb
            return True
        for u, v in P['edges']:
            cu, cv = colors[u], colors[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = P['n']
    best_colors = list(range(n))  # fallback: all distinct
    best_score = _count_distinct(best_colors)
    
    # Try 4-coloring first (grid-like structure suggests 4 colors may suffice)
    # The graph looks like a 5x10 grid with some diagonal connections
    # Let's try a 4-color pattern: color = (row % 2) * 2 + (col % 2)
    # But first, let's understand the structure better
    
    # Try various k values from 3 upward
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        if k >= best_score:
            break
        
        # Try multiple random initial assignments with local search
        found = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Random initial coloring
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Local search: repeatedly fix violations
            improved = True
            max_iters = 500
            iters = 0
            while improved and iters < max_iters and time.time() < deadline:
                improved = False
                iters += 1
                for i in range(n):
                    if time.time() > deadline:
                        break
                    # Check if vertex i has a conflict
                    # Conflict: same color as neighbor, or bicolored cycle
                    # For simplicity, first fix proper coloring violations
                    neighbor_colors = set(colors[v] for v in _ADJ[i])
                    if i in [v for v in range(n) if colors[v] == colors[i] and v != i]:
                        pass  # can't check this way
                    
                    # Check proper coloring
                    has_conflict = False
                    for v in _ADJ[i]:
                        if colors[v] == colors[i]:
                            has_conflict = True
                            break
                    
                    if has_conflict:
                        # Try changing to a different color
                        for new_c in range(k):
                            if new_c == colors[i]:
                                continue
                            if new_c in neighbor_colors:
                                continue
                            # Try this color
                            old_c = colors[i]
                            colors[i] = new_c
                            # Quick check: does this create a bicolored cycle?
                            # We'll do full validation later, but quick check:
                            # For each neighbor color, check if adding i creates cycle
                            # This is expensive; just try and validate at the end
                            # Actually, let's just accept and validate periodically
                            improved = True
                            break
                        else:
                            # Can't fix this vertex easily; try random
                            colors[i] = random.randint(0, k - 1)
                            improved = True
            
            # Validate the coloring
            if _validate(colors):
                score = _count_distinct(colors)
                if score < best_score:
                    best_score = score
                    best_colors = list(colors)
                if score <= k:
                    found = True
                    break
        
        if found:
            break
    
    # If we haven't found something better, try a more systematic approach
    if best_score > 4 and time.time() < deadline:
        # Try 4-coloring with backtracking
        colors = [-1] * n
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            # Try colors in order
            for c in range(4):
                # Check proper coloring
                ok = True
                for v in _ADJ[idx]:
                    if v < idx and colors[v] == c:
                        ok = False
                        break
                    elif v > idx:
                        # Will check later, but if v is already colored...
                        # Actually in backtracking, we only check colored neighbors
                        pass
                if not ok:
                    continue
                # Also check bicolored cycle constraint (expensive, skip for now)
                colors[idx] = c
                if backtrack(idx + 1):
                    return True
                colors[idx] = -1
            return False
        
        if backtrack(0):
            if _validate(colors):
                score = _count_distinct(colors)
                if score < best_score:
                    best_score = score
                    best_colors = list(colors)
    
    return [int(c) for c in best_colors]
# EVOLVE-BLOCK-END
