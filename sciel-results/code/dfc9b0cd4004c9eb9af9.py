import time
import math
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0

    def remaining():
        return deadline - time.time()

    # ---------------- fast girth verification (non-backtracking BFS with early exit) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        min_cycle = float('inf')

        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = {}
            for nb in adj[s]:
                dist[(nb, s)] = 1
                if nb == s:
                    min_cycle = min(min_cycle, 1)
                    return min_cycle

            queue = deque([(nb, s) for nb in adj[s]])
            while queue:
                if time.time() > time_limit:
                    return -1
                node, prev = queue.popleft()
                d = dist[(node, prev)]
                if d >= g - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s:
                        cycle_len = d + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        state = (nb, node)
                        if state not in dist:
                            dist[state] = d + 1
                            queue.append((nb, node))
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Cycle-avoidant configuration model construction ----------------
    def cycle_avoidant_construct():
        if k != 3 or g != 16:
            return None

        # Moore bound for (3, 16): 2 + 3*sum_{i=0}^{7} 2^i = 2 + 3*(2^8-1) = 2 + 3*255 = 767
        # But we can try smaller n. The Moore bound is a lower bound.
        # For 3-regular girth g, n >= 2 + k*sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i
        # For g=16: floor(15/2)-1 = 6, so n >= 2 + 3*sum_{i=0}^{5} 2^i = 2 + 3*63 = 191
        # Wait, let me recalculate. The Moore bound for girth g (even) is:
        # n >= 2 + k * sum_{i=0}^{g/2 - 2} (k-1)^i
        # For k=3, g=16: n >= 2 + 3 * sum_{i=0}^{6} 2^i = 2 + 3*(2^7 - 1) = 2 + 3*127 = 383
        
        # Let's try n from a reasonable lower bound upward
        # The known best is 936, so we should be able to find something smaller with good search
        
        # Try various n values, starting from smaller ones
        # n must be even for 3-regular graph
        candidate_ns = []
        for n in range(200, 1000, 2):
            candidate_ns.append(n)
        
        # Sort by n (we want smallest)
        candidate_ns.sort()
        
        for n in candidate_ns:
            if remaining() < 30:
                break
            
            # Try multiple attempts for each n
            max_attempts = 200 if n < 400 else 100
            
            for attempt in range(max_attempts):
                if remaining() < 30:
                    break
                
                # Build adjacency list using configuration model
                # Each vertex has k=3 stubs
                # We'll pair stubs to create edges
                
                # Represent stubs as (vertex, stub_index)
                stubs = []
                for v in range(n):
                    for s in range(k):
                        stubs.append((v, s))
                
                random.shuffle(stubs)
                
                # Track which stubs are used
                used = [False] * len(stubs)
                edges = []
                adj = [[] for _ in range(n)]
                
                # For each vertex, track its neighbors to avoid multi-edges and self-loops
                # Also need to check for short cycles
                
                success = True
                edge_count = 0
                target_edge_count = n * k // 2  # 3n/2 edges
                
                # We'll process stubs in order, pairing each free stub with another
                for i in range(len(stubs)):
                    if used[i]:
                        continue
                    
                    v1, s1 = stubs[i]
                    
                    # Find a valid partner for this stub
                    found = False
                    attempts = 0
                    max_partner_attempts = 50
                    
                    while not found and attempts < max_partner_attempts:
                        # Pick a random unused stub
                        j = random.randint(0, len(stubs) - 1)
                        if used[j]:
                            attempts += 1
                            continue
                        
                        v2, s2 = stubs[j]
                        
                        # Check: no self-loop
                        if v1 == v2:
                            attempts += 1
                            continue
                        
                        # Check: no multi-edge (v1, v2) already connected
                        if v2 in adj[v1]:
                            attempts += 1
                            continue
                        
                        # Check: adding this edge doesn't create a cycle shorter than g
                        # We need to check if there's a path from v1 to v2 of length < g-1
                        # that doesn't use the edge we're about to add
                        # More precisely: if there's a non-backtracking walk from v1 to v2 of length < g-1
                        # that doesn't go through the edge (v1,v2), then adding the edge creates a cycle of length < g
                        
                        # Use BFS to check if there's a short path
                        # We need to be careful: the path should not use the edge (v1,v2)
                        # Since we haven't added it yet, any existing path from v1 to v2 would create a cycle
                        
                        # Actually, we need to check: is there a path from v1 to v2 of length L where 1 <= L <= g-2?
                        # If yes, adding edge (v1,v2) creates a cycle of length L+1 <= g-1 < g
                        # But we also need to make sure the path doesn't backtrack in a way that creates a shorter cycle
                        # For simplicity, let's just check if there's any path of length < g-1
                        
                        # BFS from v1, tracking distance
                        if v1 == v2:
                            attempts += 1
                            continue
                        
                        # Check if there's a path of length 1 to g-2 from v1 to v2
                        # Use BFS
                        dist = {v1: 0}
                        queue = deque([v1])
                        found_short_path = False
                        while queue:
                            node = queue.popleft()
                            d = dist[node]
                            if d >= g - 1:
                                continue
                            for nb in adj[node]:
                                if nb not in dist:
                                    dist[nb] = d + 1
                                    if nb == v2:
                                        found_short_path = True
                                        break
                                    queue.append(nb)
                            if found_short_path:
                                break
                        
                        if found_short_path:
                            attempts += 1
                            continue
                        
                        # This pairing is valid
                        used[i] = True
                        used[j] = True
                        edges.append((v1, v2))
                        adj[v1].append(v2)
                        adj[v2].append(v1)
                        edge_count += 1
                        found = True
                    
                    if not found:
                        success = False
                        break
                
                if not success or edge_count != target_edge_count:
                    continue
                
                # Verify the graph is valid
                # Check all vertices have degree exactly k
                valid = True
                for v in range(n):
                    if len(adj[v]) != k:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Verify girth
                gc = check_girth(n, edges, g, deadline - 15)
                if gc >= g:
                    return (n, edges)
        
        return None

    # ---------------- LUW construction (guaranteed fallback) ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = _luw_ops(q)
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

    # ---------------- Petersen voltage lift with vectorized constraint checking ----------------
    def petersen_lift_construct():
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
        
        coeff_vectors = set()
        max_walk_len = g - 1
        
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            
            while stack:
                v, prev_v, l, counts = stack.pop()
                
                if l >= max_walk_len:
                    continue
                
                for next_v in range(n_base):
                    if next_v == prev_v:
                        continue
                    if next_v == v:
                        continue
                    
                    e_idx = edge_index.get((v, next_v))
                    if e_idx is None:
                        continue
                    
                    new_counts = counts.copy()
                    u_edge, v_edge = pet_edges[e_idx]
                    if v == u_edge and next_v == v_edge:
                        new_counts[e_idx] += 1
                    else:
                        new_counts[e_idx] -= 1
                    
                    new_l = l + 1
                    if next_v == start_v and new_l > 0:
                        coeff_vectors.add(tuple(new_counts))
                    
                    stack.append((next_v, v, new_l, new_counts))
        
        unique_coeffs = set()
        for c in coeff_vectors:
            if all(x == 0 for x in c):
                continue
            for i in range(15):
                if c[i] != 0:
                    if c[i] < 0:
                        c = tuple(-x for x in c)
                    unique_coeffs.add(c)
                    break
        coeff_list = list(unique_coeffs)
        
        if len(coeff_list) > 0:
            coeff_matrix = np.array(coeff_list, dtype=np.int64)
            n_constraints = len(coeff_matrix)
        else:
            coeff_matrix = np.zeros((0, 15), dtype=np.int64)
            n_constraints = 0

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

        primes = []
        candidate = 2
        while len(primes) < 200:
            if is_prime(candidate):
                primes.append(candidate)
            candidate += 1
        
        best_n = float('inf')
        best_edges = None
        
        for m in primes:
            if time.time() > deadline - 15:
                break
            
            n_candidate = 10 * m
            if n_candidate >= best_n:
                continue
            
            max_attempts = 600 if m < 100 else 300
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 15:
                    break
                
                voltages = np.array([random.randint(1, m - 1) for _ in range(15)], dtype=np.int64)
                
                for repair_step in range(100):
                    if time.time() > deadline - 15:
                        break
                    
                    if n_constraints > 0:
                        sums = (coeff_matrix @ voltages) % m
                        violated_indices = np.where(sums == 0)[0]
                        
                        if len(violated_indices) == 0:
                            edges = _build_lifted_graph(pet_edges, voltages.tolist(), m)
                            return n_candidate, edges
                        
                        viol_idx = random.choice(violated_indices.tolist())
                        c = coeff_list[viol_idx]
                        
                        candidates = [i for i in range(15) if c[i] != 0]
                        if not candidates:
                            break
                        j = random.choice(candidates)
                        
                        delta = random.randint(1, m - 1)
                        voltages[j] = (voltages[j] + delta) % m
                        if voltages[j] == 0:
                            voltages[j] = 1
                    else:
                        edges = _build_lifted_graph(pet_edges, voltages.tolist(), m)
                        return n_candidate, edges
        
        return None

    def _build_lifted_graph(pet_edges, voltages, m):
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
        
        edges = set()
        for u in range(n_base):
            for v in range(n_base):
                if u == v:
                    continue
                e_idx = edge_index.get((u, v))
                if e_idx is None:
                    continue
                
                vol = voltages[e_idx]
                u_edge, v_edge = pet_edges[e_idx]
                if u == u_edge and v == v_edge:
                    sign = 1
                else:
                    sign = -1
                
                for layer in range(m):
                    u_node = layer * n_base + u
                    target_layer = (layer + sign * vol) % m
                    v_node = target_layer * n_base + v
                    
                    a, b = min(u_node, v_node), max(u_node, v_node)
                    edges.add((a, b))
        
        return list(edges)

    # ---------------- Margulis-LUB Ramanujan graph construction ----------------
    def lub_construct():
        if k != 3 or g != 16:
            return None

        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(math.isqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True

        best_result = None

        primes = []
        for n in range(2, 500):
            if is_prime(n):
                primes.append(n)

        candidates = []
        for i, p in enumerate(primes):
            for q in primes[i:]:
                n = p * q
                if n >= 512 and n <= 2000:
                    candidates.append((n, p, q))

        candidates.sort()

        for n, p, q in candidates:
            if remaining() < 30:
                break

            edges = []
            for x in range(p):
                for y in range(q):
                    idx = x * q + y
                    nx = (x + 1) % p
                    nidx = nx * q + y
                    if idx < nidx:
                        edges.append((idx, nidx))
                    nx = (x - 1) % p
                    nidx = nx * q + y
                    if idx < nidx:
                        edges.append((idx, nidx))
                    ny = (y + 1) % q
                    nidx = x * q + ny
                    if idx < nidx:
                        edges.append((idx, nidx))

            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            valid = True
            for i in range(n):
                if len(adj[i]) != 3:
                    valid = False
                    break
            if not valid:
                continue

            gc = check_girth(n, edges, g, deadline - 10)
            if gc >= g:
                return (n, edges)
            elif gc > 0:
                continue

        return None

    # Try cycle-avoidant construction first
    result = cycle_avoidant_construct()
    if result is not None:
        return result

    # Try Petersen voltage lift
    result = petersen_lift_construct()
    if result is not None:
        return result

    # Try Margulis-LUB construction
    result = lub_construct()
    if result is not None:
        return result

    # Fallback to the LUW construction.
    return luw_construct()
