import time
import random
import math
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    if k == 3:
        # Primary: Voltage lift of Petersen with improved search (inspiration 1)
        # Fallback: Greedy configuration model (strategy directive)
        # Final fallback: LUW / Petersen lift from current program
        result = _voltage_lift_petersen_improved(k, g, deadline, start_time)
        if result is not None:
            return result
        
        result = _greedy_configuration_model(k, g, deadline, start_time)
        if result is not None:
            return result
            
        # Last resort: original Petersen lift from current program
        result = _construct_petersen_lift(g, deadline, start_time)
        if result is not None:
            return result
            
        return _construct_luw(k, g, d=11)
    else:
        return _construct_luw(k, g, d=7)

def _voltage_lift_petersen_improved(k, g, deadline, start_time):
    """
    Voltage lifting of the Petersen graph with improved search.
    Uses numpy for constraint checking and simulated annealing for voltage assignment.
    """
    if k != 3 or g != 16:
        return None

    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    adj = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Generate all non-backtracking closed walks of length [5, g-1]
    max_L = g - 1
    walk_vectors = set()

    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]
        while stack:
            current, prev, length, counts = stack.pop()

            if length >= max_L:
                continue

            for next_v in adj[current]:
                if next_v == prev:
                    continue
                e_idx = edge_index[(current, next_v)]
                u, v = pet_edges[e_idx]
                d = 1 if current == u else -1

                new_counts = counts.copy()
                new_counts[e_idx] += d
                new_l = length + 1

                if next_v == start_v and new_l >= 5:
                    walk_vectors.add(tuple(new_counts))

                stack.append((next_v, current, new_l, new_counts))

    if not walk_vectors:
        return None

    # Filter out zero vectors and duplicates
    unique_constraints = set()
    for c in walk_vectors:
        if all(x == 0 for x in c):
            continue
        first_non_zero = -1
        for i in range(15):
            if c[i] != 0:
                first_non_zero = i
                break
        if first_non_zero != -1:
            if c[first_non_zero] < 0:
                c = tuple(-x for x in c)
            unique_constraints.add(c)

    C = np.array(list(unique_constraints), dtype=np.int64)
    if C.size == 0:
        return None

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def solve_voltage(m, max_iters=10000):
        # Initialize with random values
        v = np.random.randint(1, m, size=15).astype(np.int64)

        def cost(v):
            residues = np.mod(C @ v, m)
            return np.sum(residues == 0)

        best_v = v.copy()
        best_cost = cost(v)

        if best_cost == 0:
            return v, 0

        # Simulated annealing with adaptive temperature
        T = 1.0
        cooling = 0.9999
        min_T = 0.01

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Pick a random variable to change
            j = random.randint(0, 14)
            old_val = v[j]
            v[j] = random.randint(1, m - 1)

            new_cost = cost(v)
            delta = new_cost - best_cost

            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_v = v.copy()
                if new_cost == 0:
                    return v, 0
            else:
                v[j] = old_val

            T *= cooling

        return best_v, best_cost

    # Try primes in increasing order starting from 53
    # Moore bound for (3,16) is 512. n = 10*m. So m >= 52.
    # Smallest prime >= 52 is 53.
    m = 53
    while m < 2000 and time.time() < deadline - 20:
        if is_prime(m):
            # Try multiple restarts with different strategies
            for restart in range(30):
                if time.time() > deadline - 15:
                    break

                v, cost = solve_voltage(m, max_iters=5000)

                if cost == 0:
                    # Construct graph
                    n = 10 * m
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = v[e_idx]
                        for layer in range(m):
                            u_lift = u * m + layer
                            v_lift = v_base * m + (layer + vol) % m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))

                    # Verify
                    if _verify_girth_fast(n, edges, g):
                        return (n, edges)

        m += 1

    return None

def _verify_girth_fast(n, edges, min_girth):
    """
    Fast girth verification using BFS from every vertex.
    Returns True if girth >= min_girth.
    """
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    for i in range(n):
        if len(adj[i]) != 3:
            return False
            
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb == parent[node]:
                    continue
                if dist[nb] != -1:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        return False
                else:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
    return True

def _greedy_configuration_model(k, g, deadline, start_time):
    """
    Greedy growth via configuration model with dynamic edge rejection.
    Attempts to build a 3-regular graph with girth >= 16 directly.
    """
    if k != 3:
        return None

    # Try a few candidate sizes
    for n in [512, 520, 530, 540, 550, 560]:
        if time.time() > deadline - 10:
            break
            
        # Generate stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        random.shuffle(stubs)
        edges = set()
        adj = [[] for _ in range(n)]
        valid = True
        
        # Greedy matching with girth check
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
                
            # Check if adding this edge creates a short cycle
            if _would_create_short_cycle(u, v, adj, g):
                valid = False
                break
                
            edges.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
            
        if valid:
            # Final verification
            if _verify_girth_fast(n, list(edges), g):
                return n, list(edges)
                
    return None

def _would_create_short_cycle(u, v, adj, girth_req):
    """
    Check if adding edge (u, v) to the current graph creates a cycle shorter than girth_req.
    This is an approximation: we check if there's a path between u and v of length < girth_req - 1.
    """
    # BFS from u to see if we can reach v in < girth_req - 1 steps
    dist = {u: 0}
    queue = deque([u])
    
    while queue:
        node = queue.popleft()
        if dist[node] >= girth_req - 1:
            break
        for nb in adj[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                if nb == v:
                    return dist[nb] + 1 < girth_req
                queue.append(nb)
                
    return False

def _construct_petersen_lift(g, deadline, start_time):
    # Petersen graph edges
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    n_edges_base = 15
    
    base_adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(pet_edges):
        base_adj[u].append((v, i))
        base_adj[v].append((u, i))

    # Generate all closed non-backtracking walks of length 3..g-1 in Petersen
    def _get_closed_nb_walks(max_len):
        walks = set()
        for start_v in range(n_base):
            stack = []
            for next_v, e_id in base_adj[start_v]:
                u, v = pet_edges[e_id]
                sign = 1 if u == start_v else -1
                stack.append((next_v, e_id, 1, [(e_id, sign)]))
            while stack:
                curr_v, prev_e, length, path = stack.pop()
                if length >= max_len:
                    continue
                if curr_v == start_v and length >= 3:
                    walks.add(tuple(path))
                for next_v, e_id in base_adj[curr_v]:
                    if e_id == prev_e:
                        continue
                    u, v = pet_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    new_path = path + [(e_id, sign)]
                    stack.append((next_v, e_id, length + 1, new_path))
        return list(walks)

    closed_walks = _get_closed_nb_walks(g)

    # Precompute sparse coefficient vectors
    walk_coeffs = []
    for w in closed_walks:
        coeff = [0] * n_edges_base
        for e_id, sign in w:
            coeff[e_id] += sign
        sparse = [(eid, c) for eid, c in enumerate(coeff) if c != 0]
        if sparse:
            walk_coeffs.append(sparse)

    seen = set()
    unique_walk_coeffs = []
    for sparse in walk_coeffs:
        key = tuple(sorted(sparse))
        if key not in seen:
            seen.add(key)
            unique_walk_coeffs.append(sparse)
    walk_coeffs = unique_walk_coeffs

    best_n = float('inf')
    best_edges = None

    # Search for voltage assignments
    # Moore bound for k=3, g=16 is 512. n = 10*m.
    # m >= 52.
    m_candidates = list(range(52, 100)) + list(range(100, 200, 2)) + list(range(200, 400, 5))
    
    random.seed(42)

    for m in m_candidates:
        if time.time() > deadline - 20.0:
            break
        candidate_n = n_base * m
        if candidate_n >= best_n:
            continue

        max_tries = 20000 if m < 60 else 8000
        found = False

        for _ in range(max_tries):
            if time.time() > deadline - 10.0:
                break

            volts = [random.randint(1, m - 1) for _ in range(n_edges_base)]
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
                if g_val == 1:
                    break
            if g_val != 1:
                continue

            valid = True
            for sparse in walk_coeffs:
                s = 0
                for eid, c in sparse:
                    s += c * volts[eid]
                if s % m == 0:
                    valid = False
                    break

            if valid:
                lift_edges = []
                for i, (u, v) in enumerate(pet_edges):
                    vol = volts[i]
                    for t in range(m):
                        u_lift = u * m + t
                        v_lift = v * m + (t + vol) % m
                        lift_edges.append((u_lift, v_lift))

                if _verify_girth(candidate_n, lift_edges, g):
                    best_n = candidate_n
                    best_edges = lift_edges
                    found = True
                    break

        if found:
            break

    if best_edges is not None:
        return best_n, best_edges

    return _construct_luw(k=3, g=g, d=11)

def _verify_girth(n, edges, girth_req):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != 3:
            return False
            
    for start in range(n):
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        while queue:
            curr, prev, dist = queue.popleft()
            if dist >= girth_req:
                continue
            if curr == start and dist >= 3:
                return False
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if (nb, curr) not in visited:
                    visited.add((nb, curr))
                    queue.append((nb, curr, dist + 1))
    return True

def _construct_luw(k, g, d=None, variant=0):
    if d is None:
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
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
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
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
