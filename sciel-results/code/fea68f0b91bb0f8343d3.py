"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def construct():
    deadline = time.time() + 145  # 5s safety margin

    def _decode_edge(idx, n):
        cum = 0
        for i in range(n - 1):
            cnt = n - 1 - i
            if cum + cnt > idx:
                return (i, i + 1 + (idx - cum))
            cum += cnt
        return (0, 1)

    def _edge_idx(i, j, n):
        idx = 0
        for k in range(i):
            idx += n - 1 - k
        idx += j - i - 1
        return idx

    def _check_c6_with_edge(n, colors, c, i, j, edge_idx):
        """Check if adding edge (i,j) with color c creates a C6 using this edge."""
        # Need path of length 5 from i to j in color-c subgraph (excluding (i,j))
        # i - v1 - v2 - v3 - v4 - j
        adj = [set() for _ in range(n)]
        for idx2, col in enumerate(colors):
            if col != c or idx2 == edge_idx:
                continue
            ii, jj = _decode_edge(idx2, n)
            adj[ii].add(jj)
            adj[jj].add(ii)

        # DFS for path of exactly 5 edges from i to j
        def dfs(cur, target, depth, visited):
            if time.time() > deadline:
                return True
            if depth == 5:
                return target in adj[cur]
            for nxt in adj[cur]:
                if nxt in visited or nxt == target:
                    continue
                visited.add(nxt)
                if dfs(nxt, target, depth + 1, visited):
                    return True
                visited.remove(nxt)
            return False

        return dfs(i, j, 1, {i})

    def backtrack(n, start_idx, colors):
        if time.time() > deadline:
            return False
        if start_idx == len(colors):
            return True
        i, j = _decode_edge(start_idx, n)
        for c in range(4):
            colors[start_idx] = c
            if _check_c6_with_edge(n, colors, c, i, j, start_idx):
                colors[start_idx] = -1
                continue
            if backtrack(n, start_idx + 1, colors):
                return True
        colors[start_idx] = -1
        return False

    def full_validate(n, colors):
        """Full validation: check no C6 in any color."""
        for c in range(4):
            adj = [set() for _ in range(n)]
            for idx, col in enumerate(colors):
                if col != c:
                    continue
                ii, jj = _decode_edge(idx, n)
                adj[ii].add(jj)
                adj[jj].add(ii)
            # Check all 6-subsets for a Hamiltonian cycle
            for subset in __import__('itertools').combinations(range(n), 6):
                sub = set(subset)
                ok = False
                first = subset[0]
                rest = list(subset[1:])
                for perm in permutations(rest):
                    if (perm[0] in adj[first] and
                        perm[1] in adj[perm[0]] and
                        perm[2] in adj[perm[1]] and
                        perm[3] in adj[perm[2]] and
                        perm[4] in adj[perm[3]] and
                        first in adj[perm[4]]):
                        ok = True
                        break
                if ok:
                    return False
        return True

    # Try to find coloring for n=17 (known to exist), then try n=18
    for n in [18, 17]:
        if time.time() > deadline - 5:
            break
        num_edges = n * (n - 1) // 2
        colors = [-1] * num_edges
        # Symmetry breaking: fix color of (0,1) to 0
        colors[_edge_idx(0, 1, n)] = 0
        if backtrack(n, 1, colors):
            if full_validate(n, colors):
                return {'n': n, 'colors': colors}

    # Fallback: n=9 seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

# EVOLVE-BLOCK-END
