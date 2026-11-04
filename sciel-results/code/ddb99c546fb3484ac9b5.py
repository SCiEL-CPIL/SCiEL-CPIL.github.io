"""Combined program: LUW fallback + algebraic constructions + fast verification."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # --- LUW fallback ---
    def _luw_graph(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # --- Fast girth verification using BFS ---
    def _check_girth(adj, n, g_limit):
        """Return True if girth >= g_limit. Uses BFS from each vertex."""
        max_depth = (g_limit - 1) // 2
        for start in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g_limit:
                            return False
        return True

    # --- Generate 3-regular graph from connection set on Z_n ---
    def _circulant_3reg(n, conn):
        """Build 3-regular circulant graph on Z_n with connection set conn (3 elements, all != 0, symmetric)."""
        if n < 4:
            return None
        # conn must have exactly 3 distinct nonzero elements, and be symmetric: if a in conn then n-a in conn
        # For 3-regular: one element must be self-inverse (a = n-a, i.e., a = n/2) or we need pairs
        # Actually for 3-regular circulant: need |conn| = 3 with symmetry
        # Options: {a, b, n-a} where b = n-b (b=n/2), or {a, n-a, b} where b=n/2
        # So one connection must be n/2 (requires n even)
        # Or: {a, b, c} where the set is closed under negation mod n
        # For 3 elements closed under negation: one must be self-negating (n/2)
        # So n must be even, and one connection is n/2
        if n % 2 != 0:
            return None
        adj = [[] for _ in range(n)]
        edges = []
        for a in conn:
            a = a % n
            if a == 0:
                return None
        # Check symmetry
        conn_set = set(conn)
        for a in conn_set:
            if (-a) % n not in conn_set:
                return None
        for i in range(n):
            for a in conn:
                j = (i + a) % n
                if j not in adj[i]:
                    adj[i].append(j)
                    adj[j].append(i)
                    if i < j:
                        edges.append((i, j))
        # Verify all degrees are 3
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return edges, adj

    # --- Generate bipartite 3-regular graph from two connection sets ---
    def _bipartite_cayley(n_half, connA, connB):
        """Bipartite graph: left {0..n_half-1}, right {n_half..2*n_half-1}.
        Left i connects to right (i+a) mod n_half for a in connA.
        Right j connects to left (j-n_half+b) mod n_half for b in connB.
        For 3-regular: |connA| = |connB| = 3."""
        n = 2 * n_half
        if len(connA) != 3 or len(connB) != 3:
            return None
        adj = [[] for _ in range(n)]
        edges = []
        for i in range(n_half):
            for a in connA:
                j = n_half + (i + a) % n_half
                if j not in adj[i]:
                    adj[i].append(j)
                    adj[j].append(i)
                    edges.append((i, j))
        for j in range(n_half):
            for b in connB:
                i = (j + b) % n_half
                if i not in adj[n_half + j]:
                    adj[n_half + j].append(i)
                    adj[i].append(n_half + j)
                    if i < n_half + j:
                        edges.append((i, n_half + j))
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return edges, adj

    # --- Moore bound for k=3, g=16 ---
    # n >= 1 + k * sum_{i=0}^{g/2-1} (k-1)^i = 1 + 3*(1+2+4+...+128) = 1 + 3*255 = 766
    # But task says 512. Let me use the stated bound.
    moore_bound = 512

    # Compute LUW as fallback
    n_luw, edges_luw = _luw_graph(k, g)

    best_n = n_luw
    best_edges = edges_luw

    if k == 3 and g == 16:
        # Try circulant constructions
        # For 3-regular circulant on Z_n (n even): connections must be {a, b, n/2} or similar
        # where the set is symmetric. So one element is n/2, and {a, n-a} for some a.
        # conn = {a, n-a, n/2}
        # Try various n and a values
        # Girth of circulant Z_n with connections {1, a, n/2}:
        # Shortest cycle: need to find shortest non-backtracking closed walk
        
        # Systematic search over n and connection parameters
        # n must be even, >= moore_bound
        # Try n from moore_bound to min(n_luw, 1200) in steps of 2
        # For each n, try a from 2 to n//2 - 1
        
        # But this is too many combinations. Let's be smarter.
        # For girth 16, we need no cycles of length 3,4,5,...,15.
        # Circulant with {1, a, n/2}:
        # - Triangle: 1 + 1 + (n-2) = n? No. 1 + a + (n-1-a) = n. Always a cycle of length 3 if 1+a+(n-1-a) mod n = 0... 
        #   Actually cycle: 0 -> 1 -> 1+a -> 1+a+(n-1-a) = n = 0. Length 3. So {1, a, n/2} always has triangle!
        #   Wait: 0->1 (step 1), 1->1+a (step a), 1+a->0 (step -1-a = n-1-a). 
        #   But n-1-a is not in {1, a, n/2} unless n-1-a = 1 or a or n/2.
        #   So no triangle necessarily. Let me reconsider.
        #   Edges from x: x+1, x+a, x+n/2 (mod n).
        #   Triangle: x -> x+1 -> x+1+a -> x. Need (1+a) - 0 = 1+a to be in {1,a,n/2} mod n. 
        #   1+a = 1 => a=0 (no). 1+a=a => 1=0 (no). 1+a=n/2 => possible.
        #   So triangle exists iff 1+a ≡ n/2 (mod n), i.e., a = n/2 - 1.
        
        # This is getting complex. Let's just try many and verify.
        
        # Strategy: try n values, for each try a few connection sets, verify girth.
        # Limit total work to fit in time budget.
        
        max_n_try = min(n_luw - 2, 1000)
        
        # Try circulant graphs: conn = {a, n-a, n/2} for various a
        for n in range(moore_bound, max_n_try + 1, 2):
            if time.time() > deadline - 5:
                break
            if n % 2 != 0:
                continue
            # conn = {a, n-a, n/2}
            # a should be such that a != n/2, a != 0, n-a != 0, n-a != n/2
            # So a in {2, 3, ..., n/2 - 1}
            # Try a few values of a
            # Heuristic: for large girth, a should be "independent" from 1 and n/2
            # Try a = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
            a_candidates = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
            # Also try a = n//4, n//3, etc.
            a_candidates.append(n // 4)
            a_candidates.append(n // 3)
            a_candidates.append(n // 5)
            a_candidates.append(n // 7)
            
            tried = set()
            for a in a_candidates:
                if time.time() > deadline - 5:
                    break
                a = a % n
                if a <= 0 or a >= n:
                    continue
                if a == n // 2:
                    continue
                na = (n - a) % n
                if na == 0 or na == n // 2:
                    continue
                conn = [a, na, n // 2]
                conn_key = frozenset(conn)
                if conn_key in tried:
                    continue
                tried.add(conn_key)
                
                result = _circulant_3reg(n, conn)
                if result is None:
                    continue
                edges, adj = result
                if _check_girth(adj, n, g):
                    best_n = n
                    best_edges = edges
                    return best_n, best_edges

        # Try bipartite Cayley graphs
        # n_half from moore_bound//2 to max_n_try//2
        for n_half in range(moore_bound // 2, max_n_try // 2 + 1):
            if time.time() > deadline - 5:
                break
            # connA and connB are 3-element subsets of Z_{n_half}
            # Try structured choices
            # connA = {1, a, b}, connB = {1, c, d}
            # For girth, we need no short cycles.
            # Try a few parameter combinations
            param_sets = [
                ([1, 2, 4], [1, 3, 5]),
                ([1, 2, 5], [1, 3, 6]),
                ([1, 3, 7], [1, 2, 5]),
                ([1, 4, 9], [1, 3, 8]),
                ([1, 5, 11], [1, 4, 7]),
                ([1, 2, 6], [1, 3, 7]),
                ([1, 3, 8], [1, 2, 5]),
                ([1, 4, 10], [1, 3, 9]),
            ]
            for connA, connB in param_sets:
                if time.time() > deadline - 5:
                    break
                connA_mod = [x % n_half for x in connA]
                connB_mod = [x % n_half for x in connB]
                if len(set(connA_mod)) != 3 or len(set(connB_mod)) != 3:
                    continue
                if 0 in connA_mod or 0 in connB_mod:
                    continue
                result = _bipartite_cayley(n_half, connA_mod, connB_mod)
                if result is None:
                    continue
                edges, adj = result
                n = 2 * n_half
                if _check_girth(adj, n, g):
                    best_n = n
                    best_edges = edges
                    return best_n, best_edges

    # Fallback to LUW
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
