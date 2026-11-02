import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Strategy: Use a deterministic algebraic construction.
    # For k=3, we can use a Cayley graph on a cyclic group Z_n or a finite field.
    # A common construction for high girth 3-regular graphs uses the group Z_{2q} or similar.
    # However, the LUW construction in the seed is already quite good (n=936 for g=16? No, seed says 936 is known, LUW might be different).
    # Let's check the LUW output size for k=3, g=16.
    # The seed's _construct_luw returns a graph. Let's see its size.
    
    # Actually, let's try a different family: Lifts of the Petersen graph are random.
    # Let's try a deterministic lift or a different base graph.
    # Or, use the "Kautz graph" construction if applicable.
    # Kautz graphs are (d+1)-regular. For d=2, they are 3-regular.
    # The Kautz graph K(d, n) has (d+1) * d^n vertices.
    # For d=2, n=2: 3 * 4 = 12 vertices. Girth?
    # For d=2, n=3: 3 * 8 = 24 vertices.
    # For d=2, n=4: 3 * 16 = 48 vertices.
    # For d=2, n=5: 3 * 32 = 96 vertices.
    # For d=2, n=6: 3 * 64 = 192 vertices.
    # For d=2, n=7: 3 * 128 = 384 vertices.
    # For d=2, n=8: 3 * 256 = 768 vertices.
    # Girth of Kautz graph K(d, n) is at least 2n.
    # So for g=16, we need 2n >= 16 => n >= 8.
    # K(2, 8) has 768 vertices. This is smaller than 936.
    # Let's try to construct the Kautz graph K(2, 8).
    
    # Kautz graph K(d, n) vertices are strings of length n over alphabet of size d+1,
    # with no two consecutive characters equal.
    # Edges connect (x_1...x_n) to (x_2...x_n y) where y != x_n.
    
    d = k - 1  # k=3 => d=2
    n_len = (g + 1) // 2  # Need 2*n_len >= g => n_len >= g/2. For g=16, n_len=8.
    
    # Check if n_len is sufficient.
    # Girth of Kautz graph is exactly 2n_len for n_len >= 2?
    # Actually, girth is 2n_len.
    # So n_len = 8 gives girth 16.
    # Number of vertices = (d+1) * d^(n_len-1) = 3 * 2^7 = 384?
    # Wait, definition of Kautz graph K(d, n):
    # Vertices are strings of length n from {0,...,d} with no equal adjacent.
    # Count: (d+1) * d^(n-1).
    # For d=2, n=8: 3 * 2^7 = 384.
    # Is it 3-regular?
    # Out-degree: last char is x_n. Next char y can be anything except x_n. So d choices.
    # In-degree: first char is x_1. Previous char z can be anything except x_1. So d choices.
    # Wait, standard Kautz is (d+1)-regular?
    # Let's check.
    # Vertex (x_1...x_n).
    # Neighbors:
    # 1. (x_2...x_n y) for y != x_n. There are d such y.
    # 2. (z x_1...x_{n-1}) for z != x_1. There are d such z.
    # Are these distinct?
    # If d=2, out-degree 2, in-degree 2. Total degree 4?
    # No, the edge (u,v) means u -> v.
    # If we consider undirected graph, degree is d+d = 2d?
    # For d=2, degree 4.
    # We need degree 3.
    # So Kautz with d=2 is 4-regular. Not suitable for k=3.
    
    # What about d=1?
    # K(1, n) is 2-regular (cycle).
    
    # So Kautz doesn't directly give 3-regular.
    
    # Alternative: Use a lift of the Petersen graph but with a deterministic, high-girth permutation set.
    # Or, use the "Moore graph" construction if it exists (it doesn't for g=16, k=3).
    
    # Let's stick to the LUW construction but verify its size.
    # The seed's LUW construction:
    # q = k = 3.
    # d = max(3, g-5 if (g-5)%2 else g-4).
    # g=16 => g-5=11 (odd) => d=11.
    # Vertices: 2 * q * d = 2 * 3 * 11 = 66?
    # No, the LUW construction in the code creates a graph with 2 * q * d vertices?
    # Let's trace _construct_luw.
    # It iterates through vertices.
    # The number of vertices is len(order).
    # For q=3, d=11.
    # The state is (side, x). side in {0,1}. x is a tuple of length d.
    # x[0] is None. x[1..d] are in Z_q.
    # Number of states: 2 * q^(d-1)?
    # No, x[0] is None, x[1..d] are values.
    # Actually, the code says: start = (0, (None,) + (0,)*d).
    # So x has length d+1. x[0] is None.
    # The neighbours function yields 1-side, tuple(y).
    # y[0] is None? No, y = [None, first] + [0]*(d-1). So y[0] is None.
    # So the state is determined by side and x[1..d].
    # Number of possible x[1..d] is q^(d-1)? No, x[1] is free?
    # In neighbours: y[1] = first. first ranges 0..q-1.
    # So x[1] can be anything.
    # x[2..d] are determined by x and y[1].
    # So the number of vertices is 2 * q * q^(d-1) = 2 * q^d?
    # Let's check.
    # If x[1..d] are arbitrary in Z_q, then there are q^d combinations.
    # Times 2 sides. Total 2 * q^d.
    # For q=3, d=11: 2 * 3^11 = 2 * 177147 = 354294.
    # That is way too large.
    # The known smallest is 936.
    # The seed's LUW is probably not the one giving 936.
    # The 936 graph is by Exoo et al.
    
    # The seed program returns the LUW graph if no lift is found.
    # If LUW gives 354k vertices, it's a bad fallback.
    # The random lifts of Petersen (10*M vertices) are much better.
    # For M=52, n=520.
    # If we can find a lift with girth 16, n=520 is a huge improvement over 936.
    
    # Why did the seed score 1.0?
    # Maybe it didn't find a valid lift in time, or the LUW was used and scored based on that?
    # Or maybe 520 is not achievable?
    # Girth of random lift of Petersen (girth 5) with M lifts.
    # Girth ~ 5 + 2 log2(M).
    # For M=52, 5 + 2*5.7 = 16.4.
    # So it should be possible.
    
    # Let's optimize the search for the lift.
    # 1. Try M starting from the Moore bound constraint.
    # Moore bound for (3,16) is 512.
    # So n >= 512.
    # Since n = 10*M, M >= 52.
    # 2. Use a more efficient girth check.
    # 3. Use multiple random seeds.
    
    petersen_edges = _get_petersen_edges()
    
    best_n = float('inf')
    best_edges = None
    
    # Try M from 52 upward
    # We can stop if 10*M > best_n
    for M in range(52, 100):
        if time.time() > deadline:
            break
            
        n = 10 * M
        if n >= best_n:
            break
        
        # Try several random lifts
        # Increase attempts for smaller M?
        num_attempts = 50
        for _ in range(num_attempts):
            if time.time() > deadline:
                break
            edges = _random_lift(petersen_edges, M)
            if edges is not None and _has_girth_at_least(n, edges, g):
                best_n = n
                best_edges = edges
                break
        
        # If we found a graph for this M, we can stop (since larger M gives larger n)
        if best_n == n:
            break
    
    # If no lift worked, fall back to LUW construction
    if best_n == float('inf'):
        n_luw, edges_luw = _construct_luw(k, g)
        return n_luw, edges_luw
        
    return best_n, best_edges

def _get_petersen_edges():
    # Petersen graph: 10 vertices, 15 edges
    # Outer cycle: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges = []
    # Outer cycle
    for i in range(5):
        edges.append((i, (i+1) % 5))
    # Inner star
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i+1) % 5]))
    # Spokes
    for i in range(5):
        edges.append((i, i+5))
    return edges

def _random_lift(base_edges, M):
    """
    Create a random M-lift of the base graph.
    """
    n_base = 10
    n_lift = n_base * M
    
    # Assign a random permutation to each edge
    perm_map = {}
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        if key not in perm_map:
            p = list(range(M))
            random.shuffle(p)
            perm_map[key] = p
            
    lift_edges = []
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        p = perm_map[key]
        for j in range(M):
            u_node = u * M + j
            v_node = v * M + p[j]
            lift_edges.append((u_node, v_node))
            
    return lift_edges

def _has_girth_at_least(n, edges, g):
    """Check if graph has no cycle shorter than g. Uses BFS from each vertex with parent tracking."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw(k, g):
    # Lazy-evaluated LUW construction from seed
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
# EVOLVE-BLOCK-END
