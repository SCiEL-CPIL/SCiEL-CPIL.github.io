import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    q = 5
    n = q * q  # 25 vertices in the 2D grid
    # We need 31 vertices, so we'll use Z_5 x Z_5 plus 6 extra vertices
    # Actually, let's stick with the polarity graph approach but use spectral methods for optimization
    
    # Build the polarity graph of PG(2,5) - this gives 90 edges
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

    # Build polarity graph using bitsets
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
        return _fill_to_91(edges, n)

    # Build adjacency matrix for spectral methods
    A = np.zeros((n, n), dtype=np.float64)
    for e in edges:
        A[e[0]][e[1]] = 1
        A[e[1]][e[0]] = 1

    # Spectral C4 count: For a graph with adjacency matrix A,
    # the number of C4s = (1/8) * (tr(A^4) - 4*tr(A^2) + 2*tr(A) + 4*sum(A^2) - 4*sum(A))
    # Actually: C4 = (1/8)*(tr(A^4) - 4*tr(A^2) - 2*tr(A) + 4*tr(A^2))
    # Let me use the correct formula:
    # Number of closed walks of length 4 = tr(A^4)
    # This includes: 4*C4 + 2*(number of 2-cycles counted twice) + 4*(stars) + ...
    # Correct formula: C4 = (tr(A^4) - 2*m - 4*sum_v C(deg(v),2)) / 8
    # where m = number of edges
    
    def count_c4_spectral(A):
        """Count C4s using spectral method."""
        # tr(A^4)
        A2 = A @ A
        A4 = A2 @ A2
        tr_A4 = np.trace(A4)
        
        # sum of degrees
        deg = A.sum(axis=1)
        m = int(deg.sum() / 2)
        
        # sum_v C(deg(v), 2)
        sum_c2 = np.sum(deg * (deg - 1) / 2)
        
        # C4 = (tr(A^4) - 2*m - 4*sum_c2) / 8
        c4 = (tr_A4 - 2*m - 4*sum_c2) / 8
        return int(round(c4))

    def count_c4_direct(adj, n):
        """Direct count of C4s using bitsets."""
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    total += (adj[i] & adj[j]).bit_count()
        return total // 2

    # Verify spectral count matches direct count
    c4_direct = count_c4_direct(adj, n)
    c4_spectral = count_c4_spectral(A)
    # They should be equal or very close

    # Now add the 91st edge to minimize C4s
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))

    def new_c4_cost(i, j, adj):
        """Count new C4s created by adding edge (i,j)."""
        ai = adj[i] & ~(1 << j)
        aj = adj[j] & ~(1 << i)
        if ai.bit_count() > aj.bit_count():
            ai, aj = aj, ai
        cnt = 0
        x = ai
        while x:
            lb = x & (-x)
            a = lb.bit_length() - 1
            cnt += (adj[a] & aj).bit_count()
            x ^= lb
        return cnt

    # Find best edge to add
    best_edge = None
    best_cost = None
    
    for i in range(n):
        if time.time() > deadline - 30:
            break
        for j in range(i + 1, n):
            if (i, j) in edge_set:
                continue
            c = new_c4_cost(i, j, adj)
            if best_cost is None or c < best_cost:
                best_cost = c
                best_edge = (i, j)
            if best_cost == 0:
                break
        if best_cost == 0:
            break

    if best_edge is None:
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
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
    A[i][j] = 1
    A[j][i] = 1

    # Local search: try edge swaps to reduce C4s
    if time.time() < deadline - 25:
        cur_c4 = count_c4_direct(adj, n)
        
        def c4s_using_edge(u, v, adj):
            """Count C4s that use edge (u,v)."""
            au = adj[u] & ~(1 << v)
            av = adj[v] & ~(1 << u)
            if au.bit_count() > av.bit_count():
                au, av = av, au
            cnt = 0
            x = au
            while x:
                lb = x & (-x)
                a = lb.bit_length() - 1
                cnt += (adj[a] & av).bit_count()
                x ^= lb
            return cnt

        improved = True
        while improved and time.time() < deadline - 25:
            improved = False
            
            for e_idx in range(len(edges)):
                if time.time() > deadline - 25:
                    break
                
                x, y = edges[e_idx]
                
                # Count C4s destroyed by removing edge (x,y)
                destroyed = c4s_using_edge(x, y, adj)
                
                # Temporarily remove edge (x,y)
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
                edge_set.discard((x, y))
                edge_set.discard((y, x))
                A[x][y] = 0
                A[y][x] = 0
                
                # Find best non-edge to add
                best_new = None
                best_new_cost = None
                
                for a in range(n):
                    if time.time() > deadline - 25:
                        break
                    for b in range(a + 1, n):
                        if (a, b) in edge_set or (b, a) in edge_set:
                            continue
                        
                        c = new_c4_cost(a, b, adj)
                        if best_new_cost is None or c < best_new_cost:
                            best_new_cost = c
                            best_new = (a, b)
                        if c < destroyed:
                            break
                    if best_new_cost is not None and best_new_cost < destroyed:
                        break
                
                if best_new is not None and best_new_cost < destroyed:
                    a, b = best_new
                    # Add the new edge
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    edge_set.add((a, b))
                    edge_set.add((b, a))
                    A[a][b] = 1
                    A[b][a] = 1
                    edges[e_idx] = [a, b]
                    cur_c4 += best_new_cost - destroyed
                    improved = True
                else:
                    # Revert
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    edge_set.add((x, y))
                    edge_set.add((y, x))
                    A[x][y] = 1
                    A[y][x] = 1
        
        # Final C4 count
        final_c4 = count_c4_direct(adj, n)

    # Cleanup and return
    seen = set()
    final_edges = []
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            seen.add(key)
            final_edges.append([e[0], e[1]])
    
    if len(final_edges) < 91:
        return _fill_to_91(final_edges, n)
    
    return final_edges[:91]

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
