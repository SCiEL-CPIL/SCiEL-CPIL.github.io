"""Construct a 4-regular graph of girth 10 using cyclic (Z_n-regular) construction.

Strategy:
- For k=4, g=10, we use a cyclic 4-regular graph on n vertices.
- A cyclic 4-regular graph is defined by 2 connection offsets {a, b} (and their negatives),
  so vertex i connects to i±a and i±b (mod n).
- We binary search on n (from Moore bound up to a reasonable upper bound) to find the
  smallest n where the non-backtracking girth is at least g.
- Girth verification: BFS in the non-backtracking line graph (directed edges) from every
  directed edge; the shortest cycle in the non-backtracking sense gives the girth.
  Actually, for undirected graphs, the girth is the length of the shortest cycle.
  We can verify by checking that no cycle of length < g exists.
  
  Efficient girth check: For a 4-regular graph, we can do a BFS from each vertex up to
  depth g//2 = 5, tracking the path to detect backtracking. But that's O(n * 4^5) which
  is fine for n up to ~320.

  Better: Use the standard approach — for each directed edge (u,v), do a BFS in the
  non-backtracking sense (can't go back to u from v) up to depth g-2, and check if we
  can reach an edge that closes a cycle. Actually, the simplest correct method:
  
  For each vertex, do a BFS tracking the parent to avoid immediate backtracking, up to
  depth g-1. If we find a vertex at depth d < g that is already visited via a different
  path of length d' such that d + d' + 1 >= ... this gets complicated.
  
  Standard method: The girth of a graph can be found by checking for cycles of length
  3, 4, ..., g-1. For a 4-regular graph on n vertices, we can do:
  - For each vertex v, do a BFS up to depth floor(g/2). Track for each visited node the
    depth and the path. If we encounter a node at depth d that was first seen at depth d'
    with d + d' < g, and the paths don't share vertices (except possibly the start), then
    we have a cycle of length d + d'. But we need to be careful about backtracking.
  
  Simpler and correct: Enumerate all non-backtracking walks of length up to g-1 from each
  directed edge. A cycle of length L corresponds to a non-backtracking walk of length L
  that starts and ends at the same directed edge (or vertex, with the first and last
  edges matching). 
  
  Actually, the cleanest method for small graphs: 
  For each vertex s, do a BFS where the state is (current_vertex, previous_vertex) to
  prevent backtracking. Start from (s, None) — but we need to handle the start carefully.
  Alternatively, start from each directed edge (u, v) and do a non-backtracking BFS
  (next step cannot go back to u). If we reach (v, u) after L-1 steps from (u, v) where
  L < g, we have a cycle of length L. But this overcounts.
  
  Let's use a well-known approach: 
  girth = minimum over all pairs of vertices (u, v) of the shortest cycle through u and v,
  but that's O(n^2 * BFS).
  
  Given n ≤ 320 and g = 10, depth 5 BFS from each vertex is cheap. Let me do:
  For each vertex s:
    Do a BFS up to depth 5, where each state is (node, parent) to avoid backtracking.
    Record for each (node, parent) the depth. If we reach a node w at depth d where
    (w, s) was visited at depth d' and d + d' < g, we have a cycle. But (w, s) means
    we came from s to w, so the cycle would be s->...->w->s which is length d+1.
    
  This is getting messy. Let me just do a direct check: for each possible cycle length
  L from 3 to g-1, check if any cycle of length L exists. For L=3: check triangles.
  For L=4: check 4-cycles. Etc. For n=320 and L up to 9, this is feasible with smart
  enumeration.
  
  Actually, the simplest correct approach for verification: 
  Build the adjacency list. For each vertex, do a BFS in the "non-backtracking" sense
  up to depth g-1. The state is (current, previous). Start: for each neighbor of s,
  state (neighbor, s) at depth 1. Then expand: from (u, v), go to all neighbors w of u
  where w != v. If w == s and depth+1 < g, we found a cycle of length depth+1.
  Also, if we reach a state (u, v) that was already seen at a smaller depth, we can
  prune.
  
  For n=320, branching factor 3 (since we exclude backtracking), depth 9: 3^9 = 19683
  per start vertex, times 320 vertices = ~6.3M states. That's fine in Python if done
  efficiently, but might be slow. Let's cap the search at n ≤ 400 and optimize.
  
  To speed up: we can use a set of visited (node, prev) pairs per BFS, and stop early
  if we find a cycle shorter than g.

Let's implement this.
"""

import sys
from collections import deque

def construct(k, g):
    if k != 4 or g != 10:
        # Fallback to a simple construction for other parameters
        # For now, return a trivial valid graph if possible, else use the old method
        # We'll handle only k=4, g=10 optimally
        return _fallback(k, g)
    
    # Moore bound for 4-regular graph of girth 10:
    # Lower bound: 1 + k + k(k-1) + k(k-1)^2 + ... up to depth floor(g/2)-1
    # For girth g, the Moore bound is:
    # If g is even: n >= 1 + k * sum_{i=0}^{g/2-2} (k-1)^i
    # If g is odd: n >= 1 + k * sum_{i=0}^{g//2-1} (k-1)^i
    # For k=4, g=10 (even): n >= 1 + 4*(1 + 3 + 9 + 27 + 81) = 1 + 4*121 = 485? 
    # Wait, that doesn't match. Let me recompute.
    # Actually the Moore bound for girth g:
    # For even g = 2r: n >= 2 * sum_{i=0}^{r-1} (k-1)^i * k? No.
    # Standard Moore bound: n >= k * ( (k-1)^{floor(g/2)-1} - 1 ) / (k-2) for even g? 
    # Let me just use the known lower bound of 243 from the problem statement.
    
    n_min = 243  # known lower bound
    n_max = 400  # search up to 400; known construction has 320
    
    # Binary search on n
    lo, hi = n_min, n_max
    best_n = None
    best_offsets = None
    
    for n in range(n_min, n_max + 1):
        # Try to find offsets a, b such that the cyclic 4-regular graph on n vertices
        # with connections ±a, ±b has girth >= 10.
        # Constraints:
        # - a, b in {1, ..., n-1}, a != b, a != n-a, b != n-b (to avoid double edges)
        # - Actually, we need {a, n-a, b, n-b} to be 4 distinct values mod n.
        # - Also a != 0, b != 0, a != b, a != n-b, etc.
        # - The graph must be simple: no loops (a != 0, b != 0), no repeated edges
        #   (a != b, a != n-b, b != n-a, a != n-a unless n=2a which we exclude).
        
        found = False
        # We can restrict a < b and a < n-a, b < n-b to reduce symmetry, but let's
        # just iterate and check conditions.
        for a in range(1, n // 2 + 1):
            if 2 * a == n:
                continue  # would create a 2-cycle effectively (edge appears twice)
            for b in range(a + 1, n // 2 + 1):
                if 2 * b == n:
                    continue
                if b == n - a:
                    continue
                # Check that the four offsets are distinct
                offsets = {a, n - a, b, n - b}
                if len(offsets) != 4:
                    continue
                # Now check girth
                if _check_girth(n, a, b, g):
                    found = True
                    best_n = n
                    best_offsets = (a, b)
                    break
            if found:
                break
        if found:
            break
    
    if best_n is None:
        # Fallback: use a known construction or return a valid but larger graph
        # Let's try a few more n values or use a different approach
        # For now, return the Lazebnik-Ustimenko-Woldar style construction as fallback
        return _fallback(4, 10)
    
    # Build the edge list
    a, b = best_offsets
    edges = []
    for i in range(best_n):
        for off in [a, b]:
            j = (i + off) % best_n
            if i < j:
                edges.append((i, j))
            else:
                edges.append((j, i))
    # Deduplicate
    edges = list(set(edges))
    return best_n, edges


def _check_girth(n, a, b, g):
    """Check if the cyclic 4-regular graph on n vertices with offsets ±a, ±b
    has girth >= g. Returns True if girth >= g."""
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        for off in (a, b, n - a, n - b):
            j = (i + off) % n
            adj[i].append(j)
    
    # For each vertex, do a non-backtracking BFS up to depth g-1
    # State: (current, previous), depth
    # Start: for each neighbor of s, state (neighbor, s) at depth 1
    # If we reach (s, prev) at depth d where d < g, we have a cycle of length d+1? 
    # Actually, reaching (s, prev) means we went s -> ... -> s, which is a cycle of
    # length d. Wait: state (u, v) means we are at u, having come from v.
    # If we start at (nbr, s) at depth 1, and later reach (s, x) at depth d, that
    # means we have a walk s -> ... -> s of length d. That's a cycle of length d.
    # We need d < g to indicate a violation.
    
    for s in range(n):
        visited = set()
        queue = deque()
        for nbr in adj[s]:
            state = (nbr, s)
            visited.add(state)
            queue.append((nbr, s, 1))
        
        while queue:
            u, prev, depth = queue.popleft()
            if depth + 1 >= g:
                continue  # no need to expand further
            for w in adj[u]:
                if w == prev:
                    continue  # non-backtracking
                new_state = (w, u)
                if new_state in visited:
                    continue
                # Check if we've returned to s
                if w == s:
                    # Cycle of length depth + 1
                    if depth + 1 < g:
                        return False
                    else:
                        continue  # don't add to queue
                visited.add(new_state)
                queue.append((w, u, depth + 1))
    
    return True


def _fallback(k, g):
    """Fallback using the Lazebnik-Ustimenko-Woldar construction."""
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
