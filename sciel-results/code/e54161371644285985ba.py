import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    n = len(points)  # 31

    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q

    # Build polarity graph using bitsets for speed
    adj = [0] * n
    edges = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            if dot(pi, points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])

    if len(edges) != 90:
        # Fallback: shouldn't happen, but ensure valid output
        return _fill_to_91(edges, n)

    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))

    def is_edge(i, j):
        return (i, j) in edge_set or (j, i) in edge_set

    def new_c4(i, j, adj):
        """Count new 4-cycles created by adding edge (i,j).
        A new 4-cycle is i-a-b-j-i where a~i, b~j, a~b, a!=b, a!=j, b!=i.
        Since (i,j) is not yet an edge, a!=j and b!=i are automatic.
        So we count pairs (a,b) with a in N(i), b in N(j), a!=b, a~b.
        """
        ai = adj[i] & ~(1 << j)  # neighbors of i excluding j (j not neighbor anyway)
        aj = adj[j] & ~(1 << i)  # neighbors of j excluding i
        cnt = 0
        x = ai
        while x:
            lb = x & (-x)
            a = lb.bit_length() - 1
            # Count neighbors of a that are in aj, excluding a itself
            cnt += (adj[a] & aj).bit_count()
            x ^= lb
        return cnt

    def total_c4(adj):
        """Exact count of 4-cycles in the graph."""
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    total += (adj[i] & adj[j]).bit_count()
        return total // 2

    def c4_removed_by_edge(x, y, adj):
        """Count 4-cycles that use edge (x,y) in the current graph.
        These are 4-cycles x-a-b-y-x where a~x, b~y, a~b, a!=b, a!=y, b!=x.
        """
        ax = adj[x] & ~(1 << y)
        ay = adj[y] & ~(1 << x)
        cnt = 0
        xx = ax
        while xx:
            lb = xx & (-xx)
            a = lb.bit_length() - 1
            cnt += (adj[a] & ay).bit_count()
            xx ^= lb
        return cnt

    # Find the non-edge that adds the fewest new 4-cycles
    best_edge = None
    best_cost = None
    
    for i in range(n):
        if time.time() > deadline - 30:
            break
        for j in range(i + 1, n):
            if is_edge(i, j):
                continue
            c = new_c4(i, j, adj)
            if best_cost is None or c < best_cost:
                best_cost = c
                best_edge = (i, j)
            if best_cost == 0:
                break
        if best_cost == 0:
            break

    if best_edge is None:
        # Fallback: pick any non-edge
        for i in range(n):
            for j in range(i + 1, n):
                if not is_edge(i, j):
                    best_edge = (i, j)
                    break
            if best_edge:
                break

    if best_edge is None:
        return _fill_to_91(edges, n)

    i, j = best_edge
    edges.append([i, j])
    adj[i] |= (1 << j)
    adj[j] |= (1 << i)
    edge_set.add((i, j))
    edge_set.add((j, i))

    # Local search: try to improve by swapping edges
    if time.time() < deadline - 20:
        cur_total = total_c4(adj)
        
        for iteration in range(50):
            if time.time() > deadline - 20:
                break
            
            improved = False
            # Try removing each existing edge and adding a better non-edge
            for e_idx in range(len(edges)):
                if time.time() > deadline - 20:
                    break
                    
                x, y = edges[e_idx]
                
                # Count C4s using edge (x,y) before removing it
                removed_c4s = c4_removed_by_edge(x, y, adj)
                
                # Temporarily remove edge (x,y)
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
                edge_set.discard((x, y))
                edge_set.discard((y, x))
                
                # Find best non-edge to add (one that adds fewest new C4s)
                best_new = None
                best_new_cost = None
                
                for a in range(n):
                    if time.time() > deadline - 20:
                        break
                    for b in range(a + 1, n):
                        if (a, b) in edge_set or (b, a) in edge_set:
                            continue
                        if (a, b) == (x, y) or (a, b) == (y, x):
                            continue
                        
                        c = new_c4(a, b, adj)
                        if best_new_cost is None or c < best_new_cost:
                            best_new_cost = c
                            best_new = (a, b)
                        if best_new_cost == 0:
                            break
                    if best_new_cost == 0:
                        break
                
                if best_new is not None:
                    a, b = best_new
                    # Delta: remove (x,y) eliminates removed_c4s, add (a,b) creates best_new_cost
                    # Net change = best_new_cost - removed_c4s
                    if best_new_cost < removed_c4s:
                        # Accept the swap
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        edge_set.add((a, b))
                        edge_set.add((b, a))
                        edges[e_idx] = [a, b]
                        cur_total += best_new_cost - removed_c4s
                        improved = True
                    else:
                        # Revert: restore old edge
                        adj[x] |= (1 << y)
                        adj[y] |= (1 << x)
                        edge_set.add((x, y))
                        edge_set.add((y, x))
                else:
                    # No non-edge found, restore
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    edge_set.add((x, y))
                    edge_set.add((y, x))
                
                if not improved and time.time() > deadline - 40:
                    break
            
            if not improved:
                break

    return edges[:91]

def _fill_to_91(edges, n):
    """Fallback: ensure exactly 91 edges."""
    if len(edges) >= 91:
        return edges[:91]
    
    existing = set()
    for e in edges:
        existing.add(tuple(sorted(e)))
    
    for i in range(n):
        for j in range(i + 1, n):
            if len(edges) >= 91:
                break
            if (i, j) not in existing:
                edges.append([i, j])
                existing.add((i, j))
        if len(edges) >= 91:
            break
    
    return edges[:91]
