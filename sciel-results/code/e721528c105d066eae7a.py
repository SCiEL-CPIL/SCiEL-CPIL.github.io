import time
from collections import defaultdict
from itertools import product

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
    """Check if coloring is valid: adjacent differ and no bicolored cycle."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _get_vertex(r, c, n_rows=5, n_cols=10):
    """Get vertex index from row and column (with column wrap)."""
    return r * n_cols + (c % n_cols)

def _check_row_pair_valid(row_a, row_b, n_cols=10):
    """
    Check if two adjacent rows (as tuples of length n_cols) are compatible.
    Returns True if no bicolored cycle is formed between them and adjacency is valid.
    """
    # Check adjacency: for each column c, row_a[c] and row_b[c] must differ
    # Also row_a[c] and row_b[(c+1)%n_cols] must differ
    # Also row_a[c] and row_b[(c-1)%n_cols] must differ
    for c in range(n_cols):
        if row_a[c] == row_b[c]:
            return False
        if row_a[c] == row_b[(c + 1) % n_cols]:
            return False
        if row_a[c] == row_b[(c - 1) % n_cols]:
            return False
    
    # Check bicolored cycles between the two rows
    # For each color pair (ca, cb) where ca from row_a and cb from row_b,
    # check if the bipartite subgraph induced by these colors is a forest
    
    # Collect all edges between row_a and row_b for each color pair
    pair_edges = defaultdict(list)
    for c in range(n_cols):
        ca = row_a[c]
        # Edges from row_a[c] to row_b[c], row_b[c+1], row_b[c-1]
        for dc in [0, 1, -1]:
            cb = row_b[(c + dc) % n_cols]
            if ca != cb:
                pair = (min(ca, cb), max(ca, cb))
                # Vertex indices in the combined 2-row subgraph
                # row_a vertices: 0..n_cols-1, row_b vertices: n_cols..2*n_cols-1
                va = c
                vb = n_cols + ((c + dc) % n_cols)
                pair_edges[pair].append((va, vb))
    
    # Check each color pair subgraph is a forest using union-find
    for pair, elist in pair_edges.items():
        n_verts = 2 * n_cols
        parent = list(range(n_verts))
        rank = [0] * n_verts
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    
    return True

def _check_full_coloring_valid(coloring, edges):
    """Full validation of a complete coloring."""
    return _check_coloring(coloring, edges)

def _enumerate_row_patterns(k, n_cols=10):
    """
    Enumerate all valid row patterns for k colors.
    A row pattern is a tuple of length n_cols where adjacent columns differ
    (including wrap-around).
    Returns list of valid patterns.
    """
    patterns = []
    # Generate all possible colorings of a cycle of length n_cols with k colors
    # such that adjacent vertices have different colors
    def gen_row(pos, current):
        if pos == n_cols:
            # Check wrap-around: last and first must differ
            if current[0] != current[-1]:
                patterns.append(tuple(current))
            return
        for c in range(k):
            if pos > 0 and current[pos - 1] == c:
                continue
            current.append(c)
            gen_row(pos + 1, current)
            current.pop()
    
    gen_row(0, [])
    return patterns

def _solve_with_dp(k, n_rows=5, n_cols=10, deadline=None):
    """
    Use dynamic programming over row patterns to find a valid coloring
    using at most k colors.
    """
    if deadline and time.time() > deadline:
        return None
    
    # Generate all valid row patterns for k colors
    row_patterns = _enumerate_row_patterns(k, n_cols)
    
    if not row_patterns:
        return None
    
    # Pre-compute compatibility between row patterns
    # compat[i][j] = True if row_patterns[i] and row_patterns[j] are compatible
    n_pat = len(row_patterns)
    compat = [[False] * n_pat for _ in range(n_pat)]
    
    for i in range(n_pat):
        for j in range(n_pat):
            if _check_row_pair_valid(row_patterns[i], row_patterns[j], n_cols):
                compat[i][j] = True
    
    # DP: dp[row][pattern_idx] = True if we can assign row 'row' pattern 'pattern_idx'
    # and have a valid coloring for rows 0..row
    # We want to minimize the number of distinct colors used
    
    # Track which colors are used in each pattern
    pattern_colors = [set(row_patterns[i]) for i in range(n_pat)]
    
    # For minimizing distinct colors, we need to track the union of colors used
    # This makes it more complex. Let's first check if any valid coloring exists,
    # then try to minimize colors.
    
    # First, check if a valid sequence of rows exists
    # dp[r][i] = list of color sets achievable for rows 0..r ending with pattern i
    # To keep it manageable, track the frozenset of colors used
    
    # Actually, let's just find any valid sequence first, then optimize
    # dp[r][i] = True if valid sequence exists for rows 0..r ending with pattern i
    dp = [dict() for _ in range(n_rows)]
    parent = [dict() for _ in range(n_rows)]
    
    # Initialize row 0
    for i in range(n_pat):
        dp[0][i] = frozenset(pattern_colors[i])
    
    # Fill DP
    for r in range(1, n_rows):
        for j in range(n_pat):
            best_colors = None
            best_prev = None
            for i in range(n_pat):
                if i in dp[r-1] and compat[i][j]:
                    colors = dp[r-1][i] | pattern_colors[j]
                    if best_colors is None or len(colors) < len(best_colors):
                        best_colors = colors
                        best_prev = i
            if best_prev is not None:
                dp[r][j] = best_colors
                parent[r][j] = best_prev
    
    # Find the best ending pattern for the last row
    best_j = None
    best_color_count = k + 1
    for j in range(n_pat):
        if j in dp[n_rows - 1]:
            if len(dp[n_rows - 1][j]) < best_color_count:
                best_color_count = len(dp[n_rows - 1][j])
                best_j = j
    
    if best_j is None:
        return None
    
    # Reconstruct the sequence
    seq = [0] * n_rows
    seq[n_rows - 1] = best_j
    for r in range(n_rows - 2, -1, -1):
        seq[r] = parent[r + 1][seq[r + 1]]
    
    # Build the full coloring
    coloring = [0] * (n_rows * n_cols)
    for r in range(n_rows):
        for c in range(n_cols):
            coloring[r * n_cols + c] = row_patterns[seq[r]][c]
    
    # Verify the full coloring
    if _check_full_coloring_valid(coloring, P['edges']):
        return coloring
    
    return None

def _try_structured_patterns(k, n_rows=5, n_cols=10):
    """Try simple structured patterns for k colors."""
    # Pattern: color = (a*r + b*c) % k
    for a in range(k):
        for b in range(k):
            coloring = [(a * (i // n_cols) + b * (i % n_cols)) % k for i in range(n_rows * n_cols)]
            if _check_full_coloring_valid(coloring, P['edges']):
                return coloring
    return None

def _local_search_improve(coloring, adj, n, deadline, target_k):
    """Try to reduce colors via local search."""
    import random
    random.seed(42)
    coloring = coloring[:]
    
    # Relabel to use colors 0..target_k-1
    distinct = sorted(set(coloring))
    if len(distinct) <= target_k:
        return coloring if _check_coloring(coloring, P['edges']) else None
    
    color_map = {c: i % target_k for i, c in enumerate(distinct)}
    coloring = [color_map[c] for c in coloring]
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        if _check_coloring(coloring, P['edges']):
            if len(set(coloring)) <= target_k:
                return coloring
            # Try to merge two colors
            from collections import Counter
            cnt = Counter(coloring)
            colors_used = sorted(set(coloring))
            if len(colors_used) > target_k:
                c1 = min(colors_used, key=lambda c: cnt[c])
                c2 = min((c for c in colors_used if c != c1), key=lambda c: cnt[c])
                new_coloring = [c1 if c == c2 else c for c in coloring]
                if _check_coloring(new_coloring, P['edges']):
                    coloring = new_coloring
                else:
                    new_coloring = [c2 if c == c1 else c for c in coloring]
                    if _check_coloring(new_coloring, P['edges']):
                        coloring = new_coloring
        else:
            # Find a violation and fix it
            v = random.randint(0, n - 1)
            old_c = coloring[v]
            for new_c in range(target_k):
                if new_c == old_c:
                    continue
                ok = True
                for nb in adj[v]:
                    if coloring[nb] == new_c:
                        ok = False
                        break
                if not ok:
                    continue
                coloring[v] = new_c
                if _check_coloring(coloring, P['edges']):
                    break
                coloring[v] = old_c
    
    return coloring if _check_coloring(coloring, P['edges']) else None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(edges, n)
    
    # Try increasing k from 2 upward
    for k in range(2, 10):
        if time.time() > deadline - 5:
            break
        
        # First try structured patterns (very fast)
        result = _try_structured_patterns(k)
        if result is not None:
            if _check_full_coloring_valid(result, edges):
                return [int(c) for c in result]
        
        # Try DP over row patterns
        if time.time() < deadline - 10:
            result = _solve_with_dp(k, deadline=deadline - 5)
            if result is not None:
                if _check_full_coloring_valid(result, edges):
                    return [int(c) for c in result]
    
    # Fallback: greedy coloring with many colors
    coloring = [0] * n
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    for v in order:
        used = set()
        for nb in adj[v]:
            used.add(coloring[nb])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    
    # Try to reduce
    result = _local_search_improve(coloring, adj, n, deadline, 4)
    if result is not None:
        return [int(c) for c in result]
    
    return [int(c) for c in coloring]
