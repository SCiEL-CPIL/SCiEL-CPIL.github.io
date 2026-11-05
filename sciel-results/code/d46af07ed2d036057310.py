import time
import math
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    if k != 3:
        return _luw_construct(k, g)

    # Petersen graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
    
    edge_map = {}
    for idx, (u, v) in enumerate(petersen_edges):
        edge_map[(u, v)] = (idx, 1)
        edge_map[(v, u)] = (idx, -1)

    # Generate all closed non-backtracking walks of length 5 to g-1
    # Store as sign vectors of length 15
    def generate_constraints(girth):
        """Generate sign vectors for all closed non-backtracking walks of length < girth."""
        # Build walks incrementally
        # A walk is stored as (start, current, prev) for efficiency
        # But we need the full sequence to compute the sign vector
        
        # Alternative: store walks as tuples of edge indices with signs
        # Walk of length L: sequence of (edge_idx, sign) pairs
        # Non-backtracking: consecutive edges don't cancel
        
        # Let's store walks as (start_vertex, list_of_(edge_idx, sign))
        # and track current vertex and previous vertex
        
        # For efficiency, let's just enumerate directly
        # Number of walks of length L in 3-reg graph: 3 * 2^(L-1)
        # For L=15: ~50K, manageable
        
        sign_vectors = []
        
        # Build walks as (v0, v1, ..., vL) but store compactly
        # Walk state: (current_vertex, prev_vertex, list_of_signs)
        # Start: for each directed edge (u,v): state = (v, u, [sign_of_edge])
        
        # Let's use a different approach: DFS with memoization
        # Actually, let's just do iterative deepening
        
        # For each length L from 5 to girth-1:
        # Generate all non-backtracking walks of length L and check if closed
        
        # State: (current, prev, sign_vector_so_far)
        # But sign_vector is a list, expensive to copy
        
        # Better: store the walk as a list of (idx, sign) and use tuples
        # For L up to 15, this is fine
        
        # Let's generate all walks of each length
        # walks_L = list of (start, end, prev, terms_tuple)
        # where terms_tuple is a tuple of (idx, sign) for each step
        
        # Length 1:
        w = []
        for u in range(10):
            for v in petersen_adj[u]:
                idx, sign = edge_map[(u, v)]
                w.append((u, v, u, ((idx, sign),)))
        
        for L in range(2, girth):
            if L < 5:
                # Still need to build up
                nw = []
                for (s, e, p, terms) in w:
                    for nb in petersen_adj[e]:
                        if nb == p:
                            continue
                        idx, sign = edge_map[(e, nb)]
                        nw.append((s, nb, e, terms + ((idx, sign),)))
                w = nw
                continue
            
            # L >= 5: check for closed walks
            nw = []
            for (s, e, p, terms) in w:
                for nb in petersen_adj[e]:
                    if nb == p:
                        continue
                    idx, sign = edge_map[(e, nb)]
                    new_terms = terms + ((idx, sign),)
                    if nb == s:
                        # Closed walk!
                        vec = [0] * 15
                        for (i, sg) in new_terms:
                            vec[i] += sg
                        sign_vectors.append(tuple(vec))
                    else:
                        nw.append((s, nb, e, new_terms))
            w = nw
        
        return sign_vectors
    
    constraints = generate_constraints(g)
    
    # Now search for m and voltage assignment
    # For voltages v[0..14] in Z_m: for all vec in constraints, dot(vec, v) % m != 0
    
    random.seed(12345)
    
    best_n = float('inf')
    best_edges = None
    
    # Try m from 50 to 200
    for m in range(50, 250):
        if time.time() > deadline - 15:
            break
        if 10 * m >= best_n:
            continue
        
        found = False
        # Try a few assignments
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            if attempt == 0:
                v = [i % m for i in range(15)]
            elif attempt == 1:
                v = [(2*i+1) % m for i in range(15)]
            elif attempt == 2:
                v = [(3*i+2) % m for i in range(15)]
            else:
                v = [random.randint(0, m-1) for _ in range(15)]
            
            # Check constraints
            valid = True
            for vec in constraints:
                total = 0
                for i in range(15):
                    if vec[i] != 0:
                        total += vec[i] * v[i]
                if total % m == 0:
                    valid = False
                    break
            
            if valid:
                # Build the lift
                n = 10 * m
                edges = []
                for idx, (u, vv) in enumerate(petersen_edges):
                    volt = v[idx]
                    for gg in range(m):
                        eu = u * m + gg
                        ev = vv * m + (gg + volt) % m
                        edges.append((eu, ev))
                
                # Verify girth
                if _verify_girth(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                    found = True
                    break
        
        if found:
            # Found valid for this m, try smaller? No, we're going up.
            # Actually we want smallest n, so we should try from small m up.
            # If we found one, we can stop since larger m gives larger n.
            # But we started from m=50, so the first valid m gives smallest n.
            break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback
    return _luw_construct(k, g)


def _verify_girth(n, edges, g):
    """Verify graph is 3-regular and has girth >= g."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            node = queue[qi]
            qi += 1
            for nb in adj[node]:
                if nb == parent[node]:
                    continue
                if dist[nb] != -1:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < g:
                        return False
                else:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
    return True


def _luw_construct(k, g):
    d = max(3, (g - 2) // 2)
    q = k
    
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
        r = (c - 5) % 4
        c0 = c - r
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
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    min_cycle = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            node = queue[qi]
            qi += 1
            for nb in adj[node]:
                if nb == parent[node]:
                    continue
                if dist[nb] != -1:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if cycle_len < g:
                        return None
                else:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
        if min_cycle < g:
            return None

    return n, edges
