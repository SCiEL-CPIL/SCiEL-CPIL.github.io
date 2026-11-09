"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Union-Find for bicolored cycle detection
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n

        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True

    def check_validity(coloring, max_colors):
        """Check if coloring is valid: adjacent vertices differ and bicolored subgraphs are forests."""
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False

        # Check bicolored subgraphs are forests
        color_pairs = set()
        for u, v in edges:
            c1, c2 = coloring[u], coloring[v]
            if c1 != c2:
                if c1 < c2:
                    color_pairs.add((c1, c2))
                else:
                    color_pairs.add((c2, c1))

        for c1, c2 in color_pairs:
            uf = UnionFind(n)
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not uf.union(u, v):
                        return False

        return True

    def greedy_mrv_randomized(max_colors, seed):
        """
        Greedy coloring with MRV heuristic and random tie-breaking.
        Returns a valid coloring or None.
        """
        random.seed(seed)
        coloring = [-1] * n
        uncolored = set(range(n))
        
        while uncolored:
            # Pick vertex with fewest available colors
            best_v = None
            best_available = None
            best_count = max_colors + 1
            
            # To add randomness to tie-breaking, we iterate in random order
            # but we must find the MIN count. 
            # Strategy: Collect candidates with min count, then pick random from them.
            min_count_found = max_colors + 1
            candidates = []
            
            for v in uncolored:
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                if len(available) < min_count_found:
                    min_count_found = len(available)
                    candidates = [(v, available)]
                elif len(available) == min_count_found:
                    candidates.append((v, available))
                    
            if not candidates:
                return None
                
            # If any candidate has 0 available, we fail
            if min_count_found == 0:
                return None
                
            # Pick random from candidates
            v, available = random.choice(candidates)
            coloring[v] = random.choice(available)
            uncolored.discard(v)
            
        return coloring

    best_coloring = None
    best_count = n + 1

    # Try 3 colors first with many restarts
    if time.time() < deadline:
        for attempt in range(5000):
            if time.time() > deadline - 10: # Leave time for check
                break
            coloring = greedy_mrv_randomized(3, attempt)
            if coloring is not None:
                if check_validity(coloring, 3):
                    used = len(set(coloring))
                    if used < best_count:
                        best_count = used
                        best_coloring = coloring[:]
                    if best_count == 3:
                        break

    # Try 4 colors if 3 failed
    if best_count > 4 and time.time() < deadline:
        for attempt in range(5000):
            if time.time() > deadline - 10:
                break
            coloring = greedy_mrv_randomized(4, 10000 + attempt)
            if coloring is not None:
                if check_validity(coloring, 4):
                    used = len(set(coloring))
                    if used < best_count:
                        best_count = used
                        best_coloring = coloring[:]
                    if best_count == 4:
                        break

    # Try 5 colors if 4 failed
    if best_count > 5 and time.time() < deadline:
        for attempt in range(5000):
            if time.time() > deadline - 10:
                break
            coloring = greedy_mrv_randomized(5, 20000 + attempt)
            if coloring is not None:
                if check_validity(coloring, 5):
                    used = len(set(coloring))
                    if used < best_count:
                        best_count = used
                        best_coloring = coloring[:]
                    if best_count == 5:
                        break

    # Fallback: assign unique colors (always valid)
    if best_coloring is None:
        best_coloring = list(range(n))
        best_count = n

    # Convert to proper Python ints
    result = [int(c) for c in best_coloring]

    # Verify one final time
    for i, c in enumerate(result):
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49

    # Final validity check
    for u, v in edges:
        assert result[u] != result[v], f"Adjacent vertices {u},{v} have same color {result[u]}"

    # Check bicolored forests
    for c1 in range(best_count):
        for c2 in range(c1 + 1, best_count):
            uf = UnionFind(n)
            for u, v in edges:
                cu, cv = result[u], result[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    assert uf.union(u, v), f"Bicolored cycle in colors {c1},{c2}"

    return result
# EVOLVE-BLOCK-END
