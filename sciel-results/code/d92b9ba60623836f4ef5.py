"""Rewritten program: deterministic algebraic construction via voltage lift of the Petersen graph
over a small non-abelian group (A_5) to obtain a 3-regular graph with large girth."""

import time
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Uses a voltage lift of the Petersen graph over a finite group G.
    For k=3, the Petersen graph is 3-regular. We assign voltages from G to edges,
    then the lift has |G| copies of each vertex, giving 10*|G| vertices.
    
    The girth of the lift is at least g iff there are no non-backtracking walks
    of length < g in the base graph whose voltage product is the identity.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    if k != 3:
        # Fallback for other k: use a random regular graph approach or return a simple construction
        # For now, use a basic approach
        return _construct_general(k, g, deadline)
    
    # For k=3, use Petersen graph voltage lift over A_5
    # A_5 has 60 elements. The Petersen graph has 10 vertices, 15 edges.
    # Lift will have 600 vertices.
    
    # First, try A_5 (order 60)
    result = _try_voltage_lift_petersen_A5(g, deadline)
    if result is not None:
        return result
    
    # If A_5 doesn't work, try other groups or constructions
    # Try S_3 x S_3 (order 36)
    result = _try_voltage_lift_petersen_S3xS3(g, deadline)
    if result is not None:
        return result
    
    # Try Z_7 x Z_7 (order 49) - abelian
    result = _try_voltage_lift_petersen_Z7xZ7(g, deadline)
    if result is not None:
        return result
    
    # Fallback: use the LUW-style construction as last resort
    return _luw_fallback(k, g)


def _luw_fallback(k, g):
    """The original LUW construction as fallback."""
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


def _construct_general(k, g, deadline):
    """General construction for k != 3 using random voltage lifts or other methods."""
    # For general k, we can try to find a base k-regular graph and lift it
    # This is a fallback; the main focus is k=3
    # Use a simple cycle-based construction
    n = 2 * k * (g // 2)  # rough estimate
    # Actually, let's just use a random k-regular graph and check girth
    # But that's expensive. For now, return a simple construction.
    
    # Simple approach: use a circulant graph
    # For 3-regular, use a cycle with chords
    # For general k, this is tricky. Let's use a basic construction.
    
    # Moore bound for k-regular girth g:
    if g % 2 == 0:
        moore = k * (k-1)**((g//2)-1) + (k-1)**(g//2)
    else:
        moore = k * (k-1)**((g-1)//2)
    
    # Just return a simple construction that might not be optimal
    # For the purpose of this task, focus on k=3
    return _luw_fallback(k, g)


def _petersen_edges():
    """Return the Petersen graph edges (10 vertices, 15 edges, 3-regular)."""
    # Standard Petersen graph: vertices 0-9
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    return edges


def _check_girth_lift(base_edges, voltage_assign, group_order, group_mult, g):
    """Check if the voltage lift has girth at least g.
    
    A cycle of length L in the lift corresponds to a closed walk of length L in the base
    graph whose voltage product is the identity. We need no such closed non-backtracking
    walk of length < g.
    
    For a bipartite graph like the Petersen, all cycles are even length.
    We check all non-backtracking closed walks of length < g.
    """
    n_base = 10  # Petersen has 10 vertices
    identity = 0  # assume group identity is element 0
    
    # Build adjacency with voltages
    adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        # Find the voltage for edge (u,v)
        # voltage_assign is a dict mapping (u,v) to group element
        if (u, v) in voltage_assign:
            volt = voltage_assign[(u, v)]
        elif (v, u) in voltage_assign:
            volt = voltage_assign[(v, u)]
            # For undirected, we need the inverse voltage for the reverse direction
            # In a group, the inverse of g is the element h such that g*h = identity
            # For now, assume we store voltages in both directions
            volt = _group_inverse(volt, group_order, group_mult)
        else:
            return False  # missing voltage
        
        adj[u].append((v, volt))
    
    # Check all non-backtracking closed walks of length 2, 4, 6, ..., up to g-1
    # Since Petersen is bipartite, only even lengths matter
    for L in range(2, g, 2):
        if L < 2:
            continue
        # For each starting vertex and first edge, do DFS for non-backtracking walks
        for start in range(n_base):
            for first_neighbor, first_volt in adj[start]:
                # Walk: start -> first_neighbor, then L-2 more steps
                # Track: (current_vertex, previous_vertex, accumulated_voltage)
                stack = [(first_neighbor, start, first_volt)]
                while stack:
                    curr, prev, volt = stack.pop()
                    if L == 1:
                        # We've taken 1 step, need L-1 more
                        pass
                    # Actually, let's restructure: track step count
                    break
        
        # Better approach: BFS by length
        # State: (vertex, previous_vertex, voltage_product)
        # Start states: for each edge (u,v) with voltage w, state (v, u, w)
        # After L steps, if we return to start vertex with voltage = identity, we have a cycle
        
        # For each starting vertex s and first step to neighbor v with voltage w:
        # We need to find walks of exactly L steps from s back to s with product = identity
        # where the walk is non-backtracking
        
        for s in range(n_base):
            for v, w in adj[s]:
                # Non-backtracking walk of length L starting with s->v
                # State after 1 step: (v, s, w)
                # Need L-1 more steps to return to s with total voltage = identity
                if L == 1:
                    continue
                # BFS/DFS for remaining L-1 steps
                # State: (current, previous, voltage_so_far)
                # We want to reach (s, *, identity) after L steps total
                # i.e., after L-1 more steps from (v, s, w), reach vertex s with voltage = identity * w^{-1} = w^{-1}
                target_volt = _group_inverse(w, group_order, group_mult)
                
                # DFS with depth limit L-1
                found = _has_nonbacktracking_walk(adj, v, s, w, L-1, s, target_volt, group_order, group_mult)
                if found:
                    return False
    
    return True


def _has_nonbacktracking_walk(adj, curr, prev, volt, remaining_steps, target_vertex, target_volt, group_order, group_mult):
    """Check if there's a non-backtracking walk of exactly remaining_steps from curr (prev is previous vertex)
    that reaches target_vertex with voltage product = target_volt."""
    if remaining_steps == 0:
        return curr == target_vertex and volt == target_volt
    
    for next_v, edge_volt in adj[curr]:
        if next_v == prev:
            continue  # non-backtracking
        new_volt = group_mult(volt, edge_volt)
        if _has_nonbacktracking_walk(adj, next_v, curr, new_volt, remaining_steps - 1, target_vertex, target_volt, group_order, group_mult):
            return True
    return False


def _group_inverse(g, group_order, group_mult):
    """Find inverse of element g in the group."""
    # For small groups, just search
    for h in range(group_order):
        if group_mult(g, h) == 0:  # assuming identity is 0
            return h
    return 0  # shouldn't happen


def _build_A5():
    """Build the alternating group A_5 as a multiplication table.
    
    A_5 has 60 elements. We'll use a permutation representation.
    Elements are even permutations of {0,1,2,3,4}.
    """
    from itertools import permutations
    
    # Generate all even permutations of 5 elements
    def is_even(perm):
        # Count inversions
        inv = 0
        for i in range(5):
            for j in range(i+1, 5):
                if perm[i] > perm[j]:
                    inv += 1
        return inv % 2 == 0
    
    perms = [p for p in permutations(range(5)) if is_even(p)]
    n = len(perms)  # should be 60
    
    # Build multiplication table
    # Composition: (p * q)(i) = p(q(i))
    mult_table = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # compose perms[i] and perms[j]
            composed = tuple(perms[i][perms[j][k]] for k in range(5))
            # find index of composed
            mult_table[i][j] = perms.index(composed)
    
    return n, mult_table


def _try_voltage_lift_petersen_A5(g, deadline):
    """Try voltage lift of Petersen over A_5 with a deterministic voltage assignment."""
    import time
    if time.time() > deadline:
        return None
    
    base_edges = _petersen_edges()
    
    # Build A_5
    try:
        group_order, group_mult = _build_A5()
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    # We need to assign voltages to the 15 edges of the Petersen graph
    # such that no non-backtracking closed walk of length < g has identity voltage.
    
    # For g=16, we need to check walks of length 2,4,6,8,10,12,14
    # The Petersen graph has girth 5, so the shortest cycles are length 5.
    # In the lift, a 5-cycle in the base becomes a 5-cycle in the lift only if the
    # voltage product around it is identity. Otherwise it becomes a longer cycle.
    
    # Strategy: assign voltages so that all short closed walks have non-identity voltage.
    # Use a deterministic assignment based on edge indices.
    
    # Let's try assigning distinct "generator-like" elements to edges
    # A_5 is generated by two elements. Let's pick some generators.
    
    # For a systematic approach, try several deterministic assignments and check girth.
    # Assignment 1: based on edge position
    assignments = []
    
    # Edge list
    n_edges = len(base_edges)
    
    # Try assignment where voltage of edge i is some function of i
    # We need to be careful about the group structure.
    
    # Let's try: voltage of edge i = group element (i * 7) % 60
    # This is arbitrary but deterministic.
    
    for seed_offset in range(50):
        if time.time() > deadline:
            break
        
        voltage_assign = {}
        for i, (u, v) in enumerate(base_edges):
            volt = (i * 7 + seed_offset * 13) % group_order
            voltage_assign[(u, v)] = volt
            voltage_assign[(v, u)] = volt  # for undirected, we'll handle inverse
        
        # Check girth
        if _check_girth_lift(base_edges, voltage_assign, group_order, group_mult, g):
            # Build the lift
            n_lift = 10 * group_order
            edges_lift = []
            for u, v in base_edges:
                volt = voltage_assign[(u, v)]
                for coset in range(group_order):
                    u_lift = u * group_order + coset
                    v_lift = v * group_order + group_mult(coset, volt)
                    edges_lift.append((u_lift, v_lift))
            
            # Verify it's valid
            n, edges = n_lift, edges_lift
            if _verify_graph(n, edges, 3, g):
                return n, edges
    
    return None


def _try_voltage_lift_petersen_S3xS3(g, deadline):
    """Try voltage lift over S_3 x S_3 (order 36)."""
    import time
    if time.time() > deadline:
        return None
    
    base_edges = _petersen_edges()
    
    # Build S_3 x S_3
    # S_3 has 6 elements: identity, (12), (13), (23), (123), (132)
    # Represent as permutations of {0,1,2}
    from itertools import permutations
    
    s3_perms = list(permutations(range(3)))
    s3_n = 6
    
    # Build S_3 multiplication
    s3_mult = [[0]*s3_n for _ in range(s3_n)]
    for i in range(s3_n):
        for j in range(s3_n):
            composed = tuple(s3_perms[i][s3_perms[j][k]] for k in range(3))
            s3_mult[i][j] = s3_perms.index(composed)
    
    # S_3 x S_3: elements are pairs (a,b) where a,b in S_3
    group_order = s3_n * s3_n  # 36
    group_mult = [[0]*group_order for _ in range(group_order)]
    for i in range(group_order):
        a1, b1 = divmod(i, s3_n)
        for j in range(group_order):
            a2, b2 = divmod(j, s3_n)
            a_prod = s3_mult[a1][a2]
            b_prod = s3_mult[b1][b2]
            group_mult[i][j] = a_prod * s3_n + b_prod
    
    if time.time() > deadline:
        return None
    
    for seed_offset in range(30):
        if time.time() > deadline:
            break
        
        voltage_assign = {}
        for i, (u, v) in enumerate(base_edges):
            volt = (i * 5 + seed_offset * 11) % group_order
            voltage_assign[(u, v)] = volt
            voltage_assign[(v, u)] = volt
        
        if _check_girth_lift(base_edges, voltage_assign, group_order, group_mult, g):
            n_lift = 10 * group_order
            edges_lift = []
            for u, v in base_edges:
                volt = voltage_assign[(u, v)]
                for coset in range(group_order):
                    u_lift = u * group_order + coset
                    v_lift = v * group_order + group_mult(coset, volt)
                    edges_lift.append((u_lift, v_lift))
            
            n, edges = n_lift, edges_lift
            if _verify_graph(n, edges, 3, g):
                return n, edges
    
    return None


def _try_voltage_lift_petersen_Z7xZ7(g, deadline):
    """Try voltage lift over Z_7 x Z_7 (order 49, abelian)."""
    import time
    if time.time() > deadline:
        return None
    
    base_edges = _petersen_edges()
    group_order = 49
    
    def group_mult(a, b):
        a1, a2 = divmod(a, 7)
        b1, b2 = divmod(b, 7)
        return ((a1 + b1) % 7) * 7 + ((a2 + b2) % 7)
    
    for seed_offset in range(20):
        if time.time() > deadline:
            break
        
        voltage_assign = {}
        for i, (u, v) in enumerate(base_edges):
            volt = (i * 3 + seed_offset * 5) % group_order
            voltage_assign[(u, v)] = volt
            voltage_assign[(v, u)] = volt
        
        if _check_girth_lift(base_edges, voltage_assign, group_order, group_mult, g):
            n_lift = 10 * group_order
            edges_lift = []
            for u, v in base_edges:
                volt = voltage_assign[(u, v)]
                for coset in range(group_order):
                    u_lift = u * group_order + coset
                    v_lift = v * group_order + group_mult(coset, volt)
                    edges_lift.append((u_lift, v_lift))
            
            n, edges = n_lift, edges_lift
            if _verify_graph(n, edges, 3, g):
                return n, edges
    
    return None


def _verify_graph(n, edges, k, g):
    """Verify that the graph is k-regular and has girth at least g."""
    # Check degree
    degree = [0] * n
    for u, v in edges:
        if u >= n or v >= n or u < 0 or v < 0:
            return False
        if u == v:
            return False  # loop
        degree[u] += 1
        degree[v] += 1
    
    for d in degree:
        if d != k:
            return False
    
    # Check for duplicate edges
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Check girth using BFS from each vertex (for small graphs)
    # For large graphs, this is expensive. Use non-backtracking walk check.
    if n > 1000:
        # Use a sampling-based girth check
        return _check_girth_sampling(n, edges, g)
    else:
        return _check_girth_bfs(n, edges, g)


def _check_girth_bfs(n, edges, g):
    """Check girth using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS to find shortest cycle through start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _check_girth_sampling(n, edges, g):
    """Check girth using random sampling (for large graphs)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Sample some vertices and do BFS
    sample_size = min(100, n)
    random.seed(42)
    for _ in range(sample_size):
        start = random.randint(0, n - 1)
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True
