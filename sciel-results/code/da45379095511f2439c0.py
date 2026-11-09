"""construct() returns 91 edges on vertices 0..30 minimizing C4s.
Uses PG(2,5) polarity graph for C4-free core, then local search via edge swaps.
"""
import numpy as np
import time
import random

def _count_c4(adj_sets):
    """Count C4s given adjacency as list of sets."""
    total = 0
    n = len(adj_sets)
    for u in range(n):
        for v in range(u + 1, n):
            cn = len(adj_sets[u] & adj_sets[v])
            total += cn * (cn - 1) // 2
    return total // 4

def _count_c4_fast(adj_matrix):
    """Count C4s using adjacency matrix. adj_matrix is a 31x31 boolean numpy array."""
    # C4 count = (1/4) * sum over all pairs (u,v) of C(common_neighbors(u,v), 2)
    # Equivalent: (1/4) * (sum_{u,v} (A^2)_{uv} choose 2) ... 
    # Actually: number of C4s = (1/4) * sum_{u < v} C(d(u)∩d(v), 2)
    # But a cleaner formula: for each pair of vertices u,v, the number of length-2 paths
    # between them is (A^2)[u,v]. The number of C4s containing edge... no.
    # 
    # Number of C4s = (1/4) * sum_{all ordered pairs (u,v), u≠v} C((A^2)[u,v], 2) / 2
    # Let me use the standard: C4_count = (1/4) * sum_{u,v} C(common(u,v), 2)
    # where common(u,v) = number of common neighbors of u and v.
    # But each C4 is counted 4 times (once per vertex as the "top"), and for each
    # top vertex, the two bottom vertices form a pair, so each C4 contributes 1 to
    # C(common(u,v),2) for the pair of opposite vertices... 
    #
    # Standard formula: #C4 = (1/4) * sum_{u < v} C(|N(u) ∩ N(v)|, 2)
    # Wait no. Each C4 has 4 vertices. For each pair of opposite vertices in the C4,
    # they share exactly 2 common neighbors (the other two vertices of the C4).
    # Each C4 has 2 pairs of opposite vertices. So sum_{u<v} C(common(u,v), 2) counts
    # each C4 exactly 2 times (once for each pair of opposite vertices).
    # But C(common(u,v), 2) counts the number of ways to choose 2 common neighbors,
    # which for a C4 is exactly 1 per pair of opposite vertices.
    # So #C4 = (1/2) * sum_{u < v} C(common(u,v), 2)
    # Hmm, let me just use the direct method.
    
    A = adj_matrix
    # A^2 gives number of length-2 paths
    A2 = A @ A
    total = 0
    for u in range(31):
        for v in range(u + 1, 31):
            cn = int(A2[u, v])
            total += cn * (cn - 1) // 2
    return total // 4

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    
    q = 5
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Try multiple bilinear forms to find a C4-free 90-edge subgraph
    # Form: x1*x2 + a*y1*y2 + b*z1*z2 mod 5
    best_edges = None
    best_c4 = float('inf')
    
    for a in range(5):
        for b in range(5):
            if a == 0 and b == 0:
                continue
            edges = []
            for i in range(31):
                for j in range(i + 1, 31):
                    x1, y1, z1 = points[i]
                    x2, y2, z2 = points[j]
                    val = (x1 * x2 + a * y1 * y2 + b * z1 * z2) % 5
                    if val == 0:
                        edges.append((i, j))
            
            if len(edges) < 91:
                continue
            
            # Check if a 90-edge subset is C4-free by checking the full graph
            adj = [set() for _ in range(31)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            c4_full = _count_c4(adj)
            
            if c4_full == 0 and len(edges) >= 90:
                # C4-free graph with at least 90 edges
                best_edges = edges
                best_c4 = 0
                break
            elif c4_full < best_c4 and len(edges) >= 90:
                best_c4 = c4_full
                best_edges = edges
        
        if best_c4 == 0:
            break
    
    if best_edges is None:
        # Fallback: use the original form
        edges = []
        for i in range(31):
            for j in range(i + 1, 31):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                val = (x1 * x2 + y1 * y2 - z1 * z2) % 5
                if val == 0:
                    edges.append((i, j))
        best_edges = edges
    
    # Now we have best_edges. We need exactly 91 edges.
    # If len(best_edges) == 90, add one edge optimally.
    # If len(best_edges) > 90, we need to pick 91 edges minimizing C4s.
    
    if len(best_edges) == 90:
        # Add the 91st edge to minimize new C4s
        adj = [set() for _ in range(31)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        edge_set = set()
        for u, v in best_edges:
            edge_set.add((min(u,v), max(u,v)))
        
        best_add_edge = None
        best_add_c4 = float('inf')
        
        for u in range(31):
            for v in range(u + 1, 31):
                if (u, v) in edge_set:
                    continue
                # Count new C4s from adding (u,v)
                common = adj[u] & adj[v]
                # Each pair of common neighbors (a,b) where a~b forms a C4
                # Actually, adding (u,v) creates a C4 for each pair (a,b) with
                # a in N(u), b in N(v), a~b, a≠b. But a and b are common neighbors.
                # For each common neighbor a, and each other common neighbor b where a~b,
                # we get a C4: u-a-v-b-u. But we need to be careful.
                # Adding edge (u,v): new C4s = number of length-3 paths from u to v
                # that don't use edge (u,v). A length-3 path u-a-b-v where a~b.
                # a must be in N(u), b must be in N(v), and a~b.
                new_c4 = 0
                for a in adj[u]:
                    for b in adj[v]:
                        if a == b:
                            continue
                        if b in adj[a]:
                            new_c4 += 1
                # Each C4 is counted once (u-a-b-v with a in N(u)\{v}, b in N(v)\{u})
                # But wait, a C4 u-a-v-b-u: a in N(u), v in N(a), b in N(v), u in N(b)
                # When we add (u,v), the C4 is u-a-b-v-u where a in N(u), b in N(v), a~b.
                # This is counted once for each such (a,b) pair.
                # But actually each C4 has 4 vertices, and we're fixing u and v as the
                # newly connected pair. So each new C4 is counted exactly once.
                if new_c4 < best_add_c4:
                    best_add_c4 = new_c4
                    best_add_edge = (u, v)
                if new_c4 == 0:
                    break
            if best_add_c4 == 0:
                break
        
        if best_add_edge:
            u, v = best_add_edge
            best_edges.append((u, v))
    
    elif len(best_edges) > 90:
        # Pick 91 edges from the larger set to minimize C4s
        # Start with all edges, then remove edges that are in C4s
        adj = [set() for _ in range(31)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        c4_count = _count_c4(adj)
        
        # Greedily remove edges that are in the most C4s until we have 91
        while len(best_edges) > 91:
            if time.time() > deadline - 10:
                break
            # For each edge, count how many C4s it participates in
            # An edge (u,v) is in a C4 if there exists a path of length 3 from u to v
            # not using edge (u,v).
            # Simpler: for edge (u,v), the number of C4s containing it =
            # number of pairs (a,b) with a in N(u)\{v}, b in N(v)\{u}, a~b, a≠b
            # But we need to be careful not to double count.
            
            # Use a different approach: for each edge, temporarily remove it and
            # see how much C4 count drops. But that's expensive.
            
            # Alternative: just pick 91 edges randomly and hope, or use greedy.
            # Let's try: for each edge, compute its "C4 participation"
            edge_c4_count = {}
            for u, v in best_edges:
                cnt = 0
                for a in adj[u]:
                    if a == v:
                        continue
                    for b in adj[v]:
                        if b == u:
                            continue
                        if a == b:
                            continue
                        if b in adj[a]:
                            cnt += 1
                edge_c4_count[(u, v)] = cnt
            
            # Remove the edge with highest C4 participation
            max_edge = max(edge_c4_count, key=edge_c4_count.get)
            best_edges.remove(max_edge)
            u, v = max_edge
            adj[u].discard(v)
            adj[v].discard(u)
    
    # Ensure exactly 91 edges
    if len(best_edges) > 91:
        best_edges = best_edges[:91]
    elif len(best_edges) < 91:
        # Add more edges
        adj = [set() for _ in range(31)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        edge_set = set()
        for u, v in best_edges:
            edge_set.add((min(u, v), max(u, v)))
        
        while len(best_edges) < 91:
            best_e = None
            best_add = float('inf')
            for u in range(31):
                for v in range(u + 1, 31):
                    if (u, v) in edge_set:
                        continue
                    common = adj[u] & adj[v]
                    new_c4 = 0
                    for a in adj[u]:
                        for b in adj[v]:
                            if a != b and b in adj[a]:
                                new_c4 += 1
                    if new_c4 < best_add:
                        best_add = new_c4
                        best_e = (u, v)
                    if new_c4 == 0:
                        break
                if best_add == 0:
                    break
            if best_e is None:
                # Just add any non-edge
                for u in range(31):
                    for v in range(u + 1, 31):
                        if (u, v) not in edge_set:
                            best_e = (u, v)
                            break
                    if best_e:
                        break
            u, v = best_e
            best_edges.append(best_e)
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
    
    # Local search: swap edges to reduce C4s
    adj = [set() for _ in range(31)]
    for u, v in best_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    current_c4 = _count_c4(adj)
    best_edges_set = set()
    for u, v in best_edges:
        best_edges_set.add((min(u, v), max(u, v)))
    
    # Local search via edge swaps
    random.seed(42)
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 15 and no_improve < max_no_improve:
        # Pick a random edge to remove
        u1, v1 = random.choice(best_edges)
        # Pick a random non-edge to add
        candidates = []
        for u in range(31):
            for v in range(u + 1, 31):
                if (u, v) not in best_edges_set:
                    candidates.append((u, v))
        
        if not candidates:
            break
        
        improved = False
        for _ in range(100):  # Try up to 100 random swaps
            if time.time() > deadline - 15:
                break
            u2, v2 = random.choice(candidates)
            
            # Check if swap is valid (u1,v1) != (u2,v2) and (u1,v1) not in non-edges
            if (min(u1,v1), max(u1,v1)) == (min(u2,v2), max(u2,v2)):
                continue
            
            # Remove (u1,v1), add (u2,v2)
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            
            new_c4 = _count_c4(adj)
            
            if new_c4 < current_c4:
                current_c4 = new_c4
                best_edges_set.discard((min(u1,v1), max(u1,v1)))
                best_edges_set.add((min(u2,v2), max(u2,v2)))
                best_edges.remove((u1, v1))
                best_edges.append((u2, v2))
                improved = True
                no_improve = 0
            else:
                # Revert
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].discard(v2)
                adj[v2].discard(u2)
                no_improve += 1
        
        if not improved:
            no_improve += 10
    
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result

def _greedy_construct():
    """Fallback greedy construction."""
    import random
    random.seed(42)
    
    edges = []
    adj = [set() for _ in range(31)]
    
    remaining = []
    for i in range(31):
        for j in range(i+1, 31):
            remaining.append([i, j])
    
    random.shuffle(remaining)
    
    for _ in range(91):
        best_e = None
        best_add = float('inf')
        for e in remaining:
            u, v = e
            if v in adj[u]:
                continue
            add = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        add += 1
            if add < best_add:
                best_add = add
                best_e = e
            if add == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append(best_e)
        adj[u].add(v)
        adj[v].add(u)
        remaining.remove(best_e)
    
    while len(edges) < 91:
        for e in remaining:
            u, v = e
            if v not in adj[u]:
                edges.append(e)
                adj[u].add(v)
                adj[v].add(u)
                remaining.remove(e)
                break
        if not remaining:
            break
    
    return edges[:91]

def _greedy_construct():
    """Fallback greedy construction."""
    import random
    random.seed(42)
    
    def count_c4(edge_list):
        adj_local = [set() for _ in range(31)]
        for u, v in edge_list:
            adj_local[u].add(v)
            adj_local[v].add(u)
        total = 0
        for u in range(31):
            for v in range(u+1, 31):
                cn = len(adj_local[u] & adj_local[v])
                total += cn * (cn - 1) // 2
        return total // 4
    
    edges = []
    adj = [set() for _ in range(31)]
    
    # Greedy: add edges in order of minimum new C4s
    remaining = []
    for i in range(31):
        for j in range(i+1, 31):
            remaining.append([i, j])
    
    random.shuffle(remaining)
    
    for _ in range(91):
        best_e = None
        best_add = float('inf')
        for e in remaining:
            u, v = e
            if v in adj[u]:
                continue
            # Count added C4s
            add = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        add += 1
            if add < best_add:
                best_add = add
                best_e = e
            if add == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append(best_e)
        adj[u].add(v)
        adj[v].add(u)
        remaining.remove(best_e)
    
    while len(edges) < 91:
        for e in remaining:
            u, v = e
            if v not in adj[u]:
                edges.append(e)
                adj[u].add(v)
                adj[v].add(u)
                remaining.remove(e)
                break
        if not remaining:
            break
    
    return edges[:91]
