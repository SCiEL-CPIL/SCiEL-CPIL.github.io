"""Rewritten program: maximize edges in a triangle-free, non-3-colorable graph on 16 vertices."""
import time
import sys

N = 16
TIME_LIMIT = 110.0  # safety margin before 120s wall-clock kill


def _start_time():
    return time.time()


def _elapsed(t0):
    return time.time() - t0


def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix (list of bitmasks) is triangle-free."""
    for i in range(n):
        ai = adj[i]
        # For each pair of neighbors of i, check if they are adjacent
        j = (ai & -ai).bit_length() - 1
        while ai:
            j_bit = ai & -ai
            j = j_bit.bit_length() - 1
            rest = ai & ~(j_bit)
            if adj[j] & rest:
                return False
            ai = rest
    return True


def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable using backtracking with bitmasks."""
    color = [-1] * n
    # Order vertices by degree (descending) for better pruning
    degs = [adj[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda x: -degs[x])
    
    # Precompute neighbor masks in order
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Determine forbidden colors
        forbidden = 0
        nb = adj[v]
        for c in range(3):
            # Check if any neighbor has color c
            # We need to check which neighbors are already colored
            # Actually, let's use a different approach
            pass
        # Better: track which colors are used by neighbors
        used = 0
        nb = adj[v]
        # Iterate through neighbors
        m = nb
        while m:
            bit = m & -m
            u = bit.bit_length() - 1
            if color[u] != -1:
                used |= (1 << color[u])
            m = m & (m - 1)
        
        for c in range(3):
            if not (used & (1 << c)):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)


def _is_3_colorable_fast(adj, n):
    """Faster 3-colorability check using DSATUR-like ordering."""
    color = [-1] * n
    # Use a recursive approach with forward checking
    # Order by degree descending
    degs = [adj[i].bit_count() for i in range(n)]
    order = list(range(n))
    order.sort(key=lambda x: -degs[x])
    
    # Precompute: for each vertex in order, its neighbors that come before it in order
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Find which colors are forbidden by already-colored neighbors
        used = 0
        nb = adj[v]
        m = nb
        while m:
            bit = m & -m
            u = bit.bit_length() - 1
            if pos[u] < idx:  # already colored
                if color[u] != -1:
                    used |= (1 << color[u])
            m = m & (m - 1)
        
        # Try available colors
        for c in range(3):
            if not (used & (1 << c)):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)


def construct():
    t0 = time.time()
    
    # Start with a known valid graph: Mycielski of C5 (Grötzsch) on 11 vertices + 5 isolated
    # Actually, let's start with the seed and improve via local search
    
    # Build initial graph: Grötzsch graph on vertices 0-10
    # C5: 0-1-2-3-4-0
    # Mycielski: add vertices 5-9 (copies of 0-4) and vertex 10 (universal to 5-9)
    edges = []
    # Cycle
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # Copies
    for i in range(5):
        u = i
        w = (i + 1) % 5
        edges.append([u, 5 + w])
        edges.append([w, 5 + u])
    # Universal vertex
    for i in range(5):
        edges.append([5 + i, 10])
    
    # Convert to adjacency bitmask
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Verify initial graph is valid
    assert _is_triangle_free(adj, N), "Initial graph has a triangle"
    assert not _is_3_colorable_fast(adj, N), "Initial graph is 3-colorable"
    
    best_edges = len(edges)
    best_adj = list(adj)
    
    # All possible edges
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Local search: try to add edges one at a time
    improved = True
    while improved:
        improved = False
        for i, (u, v) in enumerate(all_pairs):
            if _elapsed(t0) > TIME_LIMIT:
                break
            if adj[u] & (1 << v):
                continue  # edge already exists
            # Check if adding edge (u,v) keeps graph triangle-free
            # Triangle would exist if there's a common neighbor w of u and v
            common = adj[u] & adj[v]
            if common:
                continue  # would create a triangle
            # Try adding the edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            # Check 3-colorability
            if not _is_3_colorable_fast(adj, N):
                # Valid! Keep the edge
                best_edges += 1
                best_adj = list(adj)
                improved = True
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        if _elapsed(t0) > TIME_LIMIT:
            break
    
    # Now try to remove some edges and add others (2-opt style)
    # First, build current edge list from best_adj
    current_edges = set()
    for u in range(N):
        m = best_adj[u]
        while m:
            bit = m & -m
            v = bit.bit_length() - 1
            if v > u:
                current_edges.add((u, v))
            m = m & (m - 1)
    
    # Try 2-opt: remove one edge, add two new edges
    edge_list = list(current_edges)
    improved = True
    while improved and _elapsed(t0) < TIME_LIMIT:
        improved = False
        for i in range(len(edge_list)):
            if _elapsed(t0) > TIME_LIMIT:
                break
            e1 = edge_list[i]
            for j in range(i + 1, len(edge_list)):
                if _elapsed(t0) > TIME_LIMIT:
                    break
                e2 = edge_list[j]
                # Remove e1 and e2
                adj2 = list(best_adj)
                u1, v1 = e1
                u2, v2 = e2
                adj2[u1] &= ~(1 << v1)
                adj2[v1] &= ~(1 << u1)
                adj2[u2] &= ~(1 << v2)
                adj2[v2] &= ~(1 << u2)
                
                # Try to add new edges
                for a, b in all_pairs:
                    if _elapsed(t0) > TIME_LIMIT:
                        break
                    if (a, b) in current_edges or (b, a) in current_edges:
                        continue
                    if a == b:
                        continue
                    # Check triangle-free for new edge
                    if adj2[a] & (1 << b):
                        continue  # already there
                    common = adj2[a] & adj2[b]
                    if common:
                        continue
                    # Try adding
                    adj2[a] |= (1 << b)
                    adj2[b] |= (1 << a)
                    if _is_triangle_free(adj2, N) and not _is_3_colorable_fast(adj2, N):
                        # Count edges
                        cnt = 0
                        for x in range(N):
                            cnt += adj2[x].bit_count()
                        cnt //= 2
                        if cnt > best_edges:
                            best_edges = cnt
                            best_adj = adj2
                            # Rebuild edge list
                            current_edges = set()
                            for x in range(N):
                                m = best_adj[x]
                                while m:
                                    bit = m & -m
                                    y = bit.bit_length() - 1
                                    if y > x:
                                        current_edges.add((x, y))
                                    m = m & (m - 1)
                            edge_list = list(current_edges)
                            improved = True
                        adj2[a] &= ~(1 << b)
                        adj2[b] &= ~(1 << a)
    
    # Final: try greedy addition again from the best state
    adj = list(best_adj)
    for u, v in all_pairs:
        if _elapsed(t0) > TIME_LIMIT:
            break
        if adj[u] & (1 << v):
            continue
        common = adj[u] & adj[v]
        if common:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _is_3_colorable_fast(adj, N):
            best_edges += 1
            best_adj = list(adj)
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    # Convert best_adj to edge list
    result = []
    for u in range(N):
        m = best_adj[u]
        while m:
            bit = m & -m
            v = bit.bit_length() - 1
            if v > u:
                result.append([u, v])
            m = m & (m - 1)
    
    return result


if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
