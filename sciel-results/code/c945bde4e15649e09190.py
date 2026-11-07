"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = [[] for _ in range(50)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _is_valid(coloring):
    # Check adjacent vertices differ
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycle using union-find
    # For each pair of colors, check if the induced subgraph is a forest
    color_sets = {}
    for i, c in enumerate(coloring):
        if c not in color_sets:
            color_sets[c] = []
        color_sets[c].append(i)
    
    colors_used = list(color_sets.keys())
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            c1, c2 = colors_used[i], colors_used[j]
            vertices = set(color_sets[c1] + color_sets[c2])
            # Union-find on these vertices
            parent = {v: v for v in vertices}
            rank = {v: 0 for v in vertices}
            
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
            
            for v in color_sets[c1]:
                for w in P['edges']:
                    if w[0] == v and w[1] in vertices:
                        if not union(v, w[1]):
                            return False
                    if w[1] == v and w[0] in vertices:
                        if not union(v, w[0]):
                            return False
    return True

def construct():
    deadline = time.time() + 85.0
    adj = _build_adj()
    
    # Try k from 3 upward
    for k in range(3, 15):
        if time.time() > deadline:
            break
        
        # Generate all valid row patterns: 5-tuples of colors in 0..k-1
        # where adjacent (including wrap: pos 4 and pos 0) differ
        valid_rows = []
        for row in itertools.product(range(k), repeat=5):
            ok = True
            for i in range(5):
                if row[i] == row[(i + 1) % 5]:
                    ok = False
                    break
            if ok:
                valid_rows.append(row)
        
        if not valid_rows:
            continue
        
        # For two adjacent rows r, s (r is row i, s is row i+1),
        # they are compatible if for all j: r[j] != s[j]
        # Additionally, we need to check bicolored cycle constraint.
        # The bicolored cycle constraint is global, so we handle it in the final check.
        
        # DP: dp[row_idx][row_pattern] = True if we can reach this state
        # For 10 rows, we need to check wrap-around (row 9 adjacent to row 0)
        
        # Precompute compatibility: compat[a][b] = True if row a can be followed by row b
        n_patterns = len(valid_rows)
        compat = [[False] * n_patterns for _ in range(n_patterns)]
        for a in range(n_patterns):
            for b in range(n_patterns):
                ok = True
                for j in range(5):
                    if valid_rows[a][j] == valid_rows[b][j]:
                        ok = False
                        break
                compat[a][b] = ok
        
        # DP over 10 rows
        # State: (current_row_idx, current_pattern_idx)
        # We need the full sequence to check wrap-around and bicolored cycles
        
        # For small k, the number of valid rows is manageable
        # k=3: C(3,5-cycle colorings) = 3*2*2*2*1 = 24 (actually chromatic polynomial of C5)
        # P(C5, k) = (k-1)^5 + (k-1) = for k=3: 32+2=34... let me just enumerate
        
        # DP: dp[i][p] = list of possible previous patterns (for backtracking)
        # Actually, let's just do BFS/DFS with memoization
        
        # dp[i][p] = True if we can color rows 0..i with row i = pattern p
        dp = [set() for _ in range(10)]
        for p in range(n_patterns):
            dp[0].add(p)
        
        for i in range(1, 10):
            for prev_p in dp[i - 1]:
                for curr_p in range(n_patterns):
                    if compat[prev_p][curr_p]:
                        dp[i].add(curr_p)
        
        if not dp[9]:
            continue
        
        # Check wrap-around: row 9 must be compatible with row 0
        # We need to find a full sequence row_0, ..., row_9 such that
        # compat[row_i][row_{i+1}] for all i, and compat[row_9][row_0]
        
        # Backtrack to find valid sequences
        # For efficiency, let's try to find one sequence
        
        def backtrack(i, curr_p, seq):
            if time.time() > deadline:
                return None
            if i == 9:
                # Check wrap: row 9 compatible with row 0
                if compat[curr_p][seq[0]]:
                    return seq + [curr_p]
                return None
            for next_p in dp[i + 1]:
                if compat[curr_p][next_p]:
                    result = backtrack(i + 1, next_p, seq + [curr_p])
                    if result is not None:
                        return result
            return None
        
        found = None
        for start_p in sorted(dp[0]):
            if time.time() > deadline:
                break
            result = backtrack(0, start_p, [])
            if result is not None:
                found = result
                break
        
        if found is None:
            continue
        
        # Build coloring
        coloring = []
        for row_idx in range(10):
            row = valid_rows[found[row_idx]]
            coloring.extend(row)
        
        # Final validation
        if _is_valid(coloring):
            return [int(c) for c in coloring]
    
    # Fallback: greedy or return something valid
    # Use a simple proper coloring with enough colors
    coloring = list(range(50))
    return [int(c) for c in coloring]

# EVOLVE-BLOCK-END
