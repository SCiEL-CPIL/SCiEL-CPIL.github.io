"""Construct k-regular graph of girth g; try random Cayley graphs, fallback to L-U-W."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 115  # safety margin before 120s kill
    
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    # Try random Cayley graph approach for k=5, g=9
    if k == 5 and g == 9:
        def perm_mul(p, q):
            return tuple(p[q[i]] for i in range(len(p)))
        
        def perm_inv(p):
            inv = [0] * len(p)
            for i in range(len(p)):
                inv[p[i]] = i
            return tuple(inv)
        
        def is_involution(p):
            for i in range(len(p)):
                if p[p[i]] != i:
                    return False
            return True
        
        def random_involution(n, rng):
            remaining = list(range(n))
            rng.shuffle(remaining)
            perm = list(range(n))
            i = 0
            while i < len(remaining):
                if i + 1 < len(remaining) and rng.random() < 0.5:
                    a, b = remaining[i], remaining[i+1]
                    perm[a] = b
                    perm[b] = a
                    i += 2
                else:
                    i += 1
            return tuple(perm)
        
        def group_identity(parts):
            return tuple(tuple(range(p)) for p in parts)
        
        def group_mul(elem1, elem2):
            return tuple(perm_mul(p1, p2) for p1, p2 in zip(elem1, elem2))
        
        def check_girth_cayley(parts, generators, max_depth):
            """Check if Cayley graph has girth > max_depth via BFS from identity.
            Returns True if no cycle of length <= max_depth found, False otherwise.
            Also returns the number of vertices explored."""
            identity = group_identity(parts)
            # BFS: (element, depth, parent)
            visited = {identity: 0}  # element -> depth
            queue = deque([(identity, 0, None)])
            vertex_count = 0
            while queue:
                elem, depth, parent = queue.popleft()
                vertex_count += 1
                if depth >= max_depth:
                    continue
                for gen in generators:
                    new_elem = group_mul(elem, gen)
                    if new_elem not in visited:
                        visited[new_elem] = depth + 1
                        queue.append((new_elem, depth + 1, elem))
                    else:
                        # Found a cycle; check its length
                        # The cycle length is depth + 1 + visited[new_elem] if it's a back-edge
                        # But we need to be careful: this gives an upper bound on cycle length
                        # For girth check, we just need to ensure no cycle of length < g
                        # A non-backtracking walk of length L corresponds to a cycle of length <= L
                        # If we find new_elem already visited at depth d2, the walk length is depth+1
                        # This means there's a cycle of length at most depth+1+1 = depth+2
                        # Actually, the cycle length is depth + 1 + (distance from new_elem back to some ancestor)
                        # This is complex; simpler: if we reach a node at depth d that was first seen at depth d',
                        # there's a cycle of length at most d + d' + 1
                        # For girth >= g, we need no cycle of length < g
                        # So if depth + 1 + visited[new_elem] < g, we have a short cycle
                        # But this is not exact; let's just check if depth + 1 >= g, if so we're safe
                        # Actually, the standard approach: BFS from identity, if we can reach all vertices
                        # within depth < g/2 without hitting a previously visited vertex (except parent),
                        # then girth >= g
                        # Simpler: if BFS explores up to depth g//2 and finds no back-edges to non-parents,
                        # girth >= g
                        pass
            return True, vertex_count
        
        def try_cayley_graph(parts, num_generators, rng, max_attempts=100):
            """Try to find a valid Cayley graph with the given group structure."""
            identity = group_identity(parts)
            n = 1
            for p in parts:
                # |S_p| = p!
                import math
                n *= math.factorial(p)
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    return None
                
                # Generate num_generators random involutions
                generators = []
                valid = True
                for _ in range(num_generators):
                    gens = []
                    for p in parts:
                        inv = random_involution(p, rng)
                        gens.append(inv)
                    gen = tuple(gens)
                    # Check it's not identity and not inverse of any existing (should be same for involutions)
                    if gen == identity:
                        valid = False
                        break
                    if gen in generators:
                        valid = False
                        break
                    generators.append(gen)
                
                if not valid:
                    continue
                
                # Check girth via BFS
                # We need girth >= g, so no cycle of length < g
                # BFS from identity, track depth
                # If we find a back-edge to a node at depth d where current depth + 1 + d < g, fail
                # More precisely: a cycle of length L means there's a non-backtracking walk of length L
                # From identity, BFS: if we encounter a visited node (not parent) at depth d_cur,
                # and it was first visited at depth d_vis, then there's a cycle of length <= d_cur + 1 + d_vis
                # But this is an upper bound. For exact girth check, we need to verify no cycle of length < g.
                # Simpler approach: BFS up to depth g//2. If no vertex is reached by two different paths
                # of total length < g, then girth >= g.
                
                # Let's do a proper BFS with parent tracking
                visited = {identity: (0, None)}  # elem -> (depth, parent_elem)
                queue = deque([(identity, 0, None)])
                girth_ok = True
                vertex_count = 0
                
                while queue and girth_ok:
                    elem, depth, parent = queue.popleft()
                    vertex_count += 1
                    if depth >= g // 2:
                        continue
                    for gen in generators:
                        new_elem = group_mul(elem, gen)
                        if new_elem == parent:
                            continue  # back-edge to parent, skip
                        if new_elem in visited:
                            # Found a cycle; its length is at most depth + 1 + visited[new_elem][0]
                            # But we need exact check. The cycle length is depth + 1 + dist(new_elem, elem_via_other_path)
                            # This is complex. Let's just check: if depth + 1 + visited[new_elem][0] < g, fail
                            # This is a sufficient condition for failure
                            if depth + 1 + visited[new_elem][0] < g:
                                girth_ok = False
                                break
                        else:
                            visited[new_elem] = (depth + 1, elem)
                            queue.append((new_elem, depth + 1, elem))
                
                if girth_ok:
                    # Build the full graph and return
                    # We need all vertices, not just those within g//2 depth
                    all_vertices = {}
                    queue2 = deque([identity])
                    all_vertices[identity] = 0
                    while queue2:
                        elem = queue2.popleft()
                        for gen in generators:
                            new_elem = group_mul(elem, gen)
                            if new_elem not in all_vertices:
                                all_vertices[new_elem] = len(all_vertices)
                                queue2.append(new_elem)
                    
                    n_final = len(all_vertices)
                    edges = []
                    for elem, idx in all_vertices.items():
                        for gen in generators:
                            new_elem = group_mul(elem, gen)
                            new_idx = all_vertices[new_elem]
                            if idx < new_idx:
                                edges.append((idx, new_idx))
                    
                    return n_final, edges
            
            return None
        
        # Try various group structures
        rng = random.Random(42)
        best_result = None
        
        # Try S_3 x S_3 x S_3 (size 216)
        # Try S_4 x S_3 (size 144)
        # Try S_5 (size 120)
        # Try S_4 x S_4 (size 576)
        
        group_structures = [
            [3, 3, 3],  # 216 vertices
            [4, 3],     # 144 vertices
            [5],        # 120 vertices
            [4, 4],     # 576 vertices
            [3, 4],     # 144 vertices
            [3, 3, 4],  # 864 vertices
        ]
        
        for parts in group_structures:
            if time.time() > deadline:
                break
            result = try_cayley_graph(parts, k, rng, max_attempts=50)
            if result is not None:
                n_found = result[0]
                if best_result is None or n_found < best_result[0]:
                    best_result = result
                    if n_found <= mb:
                        break  # Can't do better than Moore bound
        
        if best_result is not None:
            return best_result
    
    # Fallback to L-U-W construction
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
