"""Construct 4-regular graph of girth 10 with minimal vertices."""
import random
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110  # 10s safety margin
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use a simple construction
        return _fallback(k, g)
    
    # Target: find 4-regular graph of girth >= 10 with as few vertices as possible
    # Moore bound for (4,10): 2*(1 + 3 + 9 + 27 + 81) = 2*121 = 242, so n >= 243
    # Known best: 320
    
    best_n = None
    best_edges = None
    
    # Try different starting sizes, starting from lower bound
    # Moore bound: for even girth g, n >= 2*(1 + (k-1) + (k-1)^2 + ... + (k-1)^(g/2 - 1))
    # For k=4, g=10: 2*(1+3+9+27+81) = 242, so n >= 243 (and n must be even for 4-regular)
    
    moore = 2 * sum((k-1)**i for i in range(g//2))
    min_n = moore + 1 if moore % 2 else moore  # n must be even for k-regular
    if min_n % 2:
        min_n += 1
    
    # Try a range of n values, starting from known good sizes
    # We'll try n = 320 first (known to work), then try to find smaller
    
    for n in [320, 316, 312, 308, 304, 300, 296, 292, 288, 284, 280, 276, 272, 268, 264, 260, 256, 252, 248, 244]:
        if time.time() > deadline - 5:
            break
        if n < min_n:
            continue
        
        # Try to find a 4-regular graph of girth >= 10 on n vertices
        result = _search_4reg_girth10(n, deadline)
        if result is not None:
            edges = result
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
                # If we found a small one, we can try to go smaller
                # But for now, accept and break early if we're at a good size
                if n <= 320:
                    # We found one at or below the known best, try a few more smaller
                    # but don't spend too much time
                    pass
    
    if best_edges is None:
        # Fallback: return the LZW construction
        return _lzw_fallback(k, g)
    
    return best_n, best_edges


def _fallback(k, g):
    """Fallback for non-(4,10) cases using LZW."""
    return _lzw_fallback(k, g)


def _lzw_fallback(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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


def _has_short_cycle(adj, n, g):
    """Check if graph has any cycle of length < g. Returns True if such a cycle exists."""
    # For each vertex, do BFS up to depth (g//2 - 1) and check for back-edges
    # that would form a cycle shorter than g
    max_depth = g // 2 - 1  # For g=10, max_depth=4, checking cycles up to length 8
    # Actually we need to check cycles of length 4,5,6,7,8,9
    # A cycle of length L is found by a path of length L-1 from v to a neighbor of v
    # More efficiently: BFS from v, track depth. If we find a vertex w at depth d
    # that is adjacent to a vertex at depth d-1 (and w is not the parent), we have
    # a cycle of length 2d or 2d+1.
    
    for v in range(n):
        # BFS from v
        visited = [-1] * n  # depth, -1 = unvisited
        visited[v] = 0
        queue = deque([v])
        
        while queue:
            u = queue.popleft()
            du = visited[u]
            if du >= max_depth:
                continue
            for w in adj[u]:
                if visited[w] == -1:
                    visited[w] = du + 1
                    queue.append(w)
                else:
                    # Check if this forms a short cycle
                    # If w is adjacent to u and w is at depth d, then cycle length is d + du + 1
                    # But we need to be careful: this is the edge (u,w) which might be the tree edge
                    if w != v:  # skip the root
                        cycle_len = du + visited[w] + 1
                        if cycle_len < g:
                            return True
                    # Also check: if visited[w] < du, the edge (u,w) creates a cycle
                    # of length du - visited[w] + 1
                    if visited[w] < du:
                        cycle_len = du - visited[w] + 1
                        if cycle_len < g:
                            return True
    return False


def _count_short_cycles(adj, n, g):
    """Count the number of cycles of length < g. Used as an objective function."""
    count = 0
    max_depth = g // 2 - 1
    for v in range(n):
        visited = [-1] * n
        visited[v] = 0
        queue = deque([v])
        
        while queue:
            u = queue.popleft()
            du = visited[u]
            if du >= max_depth:
                continue
            for w in adj[u]:
                if visited[w] == -1:
                    visited[w] = du + 1
                    queue.append(w)
                else:
                    if w != v:
                        cycle_len = du + visited[w] + 1
                        if cycle_len < g:
                            count += 1
                    if visited[w] < du:
                        cycle_len = du - visited[w] + 1
                        if cycle_len < g:
                            count += 1
    return count


def _search_4reg_girth10(n, deadline):
    """Try to find a 4-regular graph of girth >= 10 on n vertices via local search."""
    if n % 2 != 0:
        return None
    if n < 8:
        return None
    
    # Generate a random 4-regular graph
    adj = _random_4reg(n)
    if adj is None:
        return None
    
    # Check initial girth
    if not _has_short_cycle(adj, n, 10):
        return _adj_to_edges(adj, n)
    
    # Local search: try to eliminate short cycles via edge swaps
    best_adj = [list(row) for row in adj]
    best_count = _count_short_cycles(best_adj, n, 10)
    
    max_iterations = 5000
    for it in range(max_iterations):
        if time.time() > deadline - 5:
            break
        if best_count == 0:
            return _adj_to_edges(best_adj, n)
        
        # Pick two random edges and try to swap them
        # Remove edges (a,b) and (c,d), add (a,c) and (b,d) or (a,d) and (b,c)
        # This maintains 4-regularity
        
        # Get all edges
        edges = []
        for i in range(n):
            for j in best_adj[i]:
                if i < j:
                    edges.append((i, j))
        
        if len(edges) < 2:
            break
        
        # Try random swaps
        found_improvement = False
        for _ in range(50):  # try 50 random swaps per iteration
            if time.time() > deadline - 5:
                break
            
            # Pick two random edges
            e1, e2 = random.sample(edges, 2)
            a, b = e1
            c, d = e2
            
            # Check they're disjoint
            if len({a, b, c, d}) < 4:
                continue
            
            # Try swap: remove (a,b) and (c,d), add (a,c) and (b,d)
            # Check no duplicate edges
            if best_adj[a][c] or best_adj[b][d] or best_adj[a][d] or best_adj[b][c]:
                continue
            
            # Apply swap
            best_adj[a][b] = best_adj[b][a] = False
            best_adj[c][d] = best_adj[d][c] = False
            best_adj[a][c] = best_adj[c][a] = True
            best_adj[b][d] = best_adj[d][b] = True
            
            new_count = _count_short_cycles(best_adj, n, 10)
            
            if new_count < best_count:
                best_count = new_count
                found_improvement = True
                if best_count == 0:
                    return _adj_to_edges(best_adj, n)
                break
            else:
                # Revert
                best_adj[a][c] = best_adj[c][a] = False
                best_adj[b][d] = best_adj[d][b] = False
                best_adj[a][b] = best_adj[b][a] = True
                best_adj[c][d] = best_adj[d][c] = True
        
        if not found_improvement:
            # Try a few more with different random edges
            pass
    
    # Final check
    if best_count == 0:
        return _adj_to_edges(best_adj, n)
    return None


def _random_4reg(n):
    """Generate a random 4-regular graph on n vertices. Returns adjacency matrix or None."""
    if n < 4 or n % 2 != 0:
        return None
    
    # Use configuration model: each vertex has 4 stubs
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    adj = [[False] * n for _ in range(n)]
    used = [False] * len(stubs)
    
    for i in range(0, len(stubs), 2):
        a = stubs[i]
        # Find a j > i such that stubs[j] is not a and (a, stubs[j]) is not already an edge
        found = False
        for j in range(i + 1, len(stubs)):
            b = stubs[j]
            if a == b:
                continue  # no loops
            if adj[a][b]:
                continue  # no multi-edges
            adj[a][b] = adj[b][a] = True
            found = True
            break
        if not found:
            return None
    
    # Verify 4-regular
    for i in range(n):
        if sum(adj[i]) != 4:
            return None
    
    return adj


def _adj_to_edges(adj, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    return edges

# EVOLVE-BLOCK-END
