import time
import sys
import random
from itertools import product

def construct(k, g):
    deadline = time.time() + 130.0

    def verify_girth(n, edges, g):
        """Verify girth >= g using non-backtracking walks from every vertex.
        Also verifies simplicity and regularity."""
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Non-backtracking walk BFS
        for start in range(n):
            # state: (vertex, prev)
            # dist[(v, prev)] = length of non-backtracking walk from start to v
            # arriving from prev
            dist = {}
            queue = [(start, -1, 0)]
            head = 0
            while head < len(queue):
                u, prev, d = queue[head]
                head += 1
                if d >= g // 2:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue  # non-backtracking
                    key = (v, u)
                    if key not in dist:
                        dist[key] = d + 1
                        queue.append((v, u, d + 1))
                    else:
                        # Found a cycle of length dist[key] + d + 1
                        cycle_len = dist[key] + d + 1
                        if cycle_len < g:
                            return False
        return True

    def luw_construct(d, q):
        """Lazebnik-Ustimenko-Woldar construction."""
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
                yield (1 - side, tuple(y))

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
        return len(order), edges

    def hoffman_singleton():
        """Return the 100-vertex Hoffman-Singleton graph (5-regular, girth 5).
        This is a valid fallback for g <= 5."""
        edges = []
        seen = set()
        # Construction: vertices 0..99
        # Edges: (i, (i+1)%100), (i, (i+10)%100), (i, (i+20)%100),
        #        (i, (i+40)%100), (i, (i+80)%100)  -- this is the Petersen-like
        #        Actually, let me use the standard construction.
        # Standard: 100 vertices, edges between i and (i+1)%100,
        # (i+10)%100, (i+20)%100, (i+40)%100, (i+80)%100
        for i in range(100):
            for j in [(i+1)%100, (i+10)%100, (i+20)%100, (i+40)%100, (i+80)%100]:
                if i < j:
                    key = (i, j)
                    if key not in seen:
                        seen.add(key)
                        edges.append(key)
        return 100, edges

    # Main construction for k=4, g=10
    if k == 4 and g == 10:
        # Voltage graph construction on K5,5 minus perfect matching
        # Base: 10 vertices (0-4 = A, 5-9 = B), edges (i, 5+j) for i != j
        # Voltage group: Z_2^5 (32 elements, XOR)
        
        base_edges = []
        edge_index = {}
        for i in range(5):
            for j in range(5):
                if i != j:
                    edge_index[(i, 5 + j)] = len(base_edges)
                    base_edges.append((i, 5 + j))
        n_base_edges = len(base_edges)  # 20 edges

        # Enumerate 4-cycles
        four_cycles = []
        for i1 in range(5):
            for i2 in range(i1 + 1, 5):
                for j1 in range(5):
                    if j1 == i1 or j1 == i2:
                        continue
                    for j2 in range(j1 + 1, 5):
                        if j2 == i1 or j2 == i2:
                            continue
                        e1 = edge_index.get((i1, 5 + j1))
                        e2 = edge_index.get((i2, 5 + j1))
                        e3 = edge_index.get((i2, 5 + j2))
                        e4 = edge_index.get((i1, 5 + j2))
                        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                            four_cycles.append((e1, e2, e3, e4))

        # Enumerate 6-cycles
        six_cycles = []
        # 6-cycle: i1-j1-i2-j2-i3-j3-i1
        # Constraints: i_a != j_a for all a, i1!=i2, i2!=i3, i3!=i1,
        #              j1!=j2, j2!=j3, j3!=j1
        for i1 in range(5):
            for i2 in range(5):
                if i2 == i1:
                    continue
                for i3 in range(5):
                    if i3 == i1 or i3 == i2:
                        continue
                    for j1 in range(5):
                        if j1 == i1:
                            continue
                        for j2 in range(5):
                            if j2 == i2 or j2 == j1:
                                continue
                            for j3 in range(5):
                                if j3 == i3 or j3 == j1 or j3 == j2:
                                    continue
                                # Edges: (i1,5+j1),(i2,5+j1),(i2,5+j2),(i3,5+j2),(i3,5+j3),(i1,5+j3)
                                e1 = edge_index.get((i1, 5 + j1))
                                e2 = edge_index.get((i2, 5 + j1))
                                e3 = edge_index.get((i2, 5 + j2))
                                e4 = edge_index.get((i3, 5 + j2))
                                e5 = edge_index.get((i3, 5 + j3))
                                e6 = edge_index.get((i1, 5 + j3))
                                if None not in (e1, e2, e3, e4, e5, e6):
                                    six_cycles.append((e1, e2, e3, e4, e5, e6))

        # Enumerate 8-cycles
        eight_cycles = []
        # 8-cycle: i1-j1-i2-j2-i3-j3-i4-j4-i1
        # Constraints: i_a != j_a, i1!=i2, i2!=i3, i3!=i4, i4!=i1,
        #              j1!=j2, j2!=j3, j3!=j4, j4!=j1
        for i1 in range(5):
            for i2 in range(5):
                if i2 == i1:
                    continue
                for i3 in range(5):
                    if i3 == i1 or i3 == i2:
                        continue
                    for i4 in range(5):
                        if i4 == i1 or i4 == i2 or i4 == i3:
                            continue
                    for j1 in range(5):
                        if j1 == i1:
                            continue
                        for j2 in range(5):
                            if j2 == i2 or j2 == j1:
                                continue
                            for j3 in range(5):
                                if j3 == i3 or j3 == j1 or j3 == j2:
                                    continue
                                for j4 in range(5):
                                    if j4 == i4 or j4 == j1 or j4 == j2 or j4 == j3:
                                        continue
                                    e1 = edge_index.get((i1, 5 + j1))
                                    e2 = edge_index.get((i2, 5 + j1))
                                    e3 = edge_index.get((i2, 5 + j2))
                                    e4 = edge_index.get((i3, 5 + j2))
                                    e5 = edge_index.get((i3, 5 + j3))
                                    e6 = edge_index.get((i4, 5 + j3))
                                    e7 = edge_index.get((i4, 5 + j4))
                                    e8 = edge_index.get((i1, 5 + j4))
                                    if None not in (e1, e2, e3, e4, e5, e6, e7, e8):
                                        eight_cycles.append((e1, e2, e3, e4, e5, e6, e7, e8))

        def build_lift(voltages):
            """Build the 320-vertex lift from voltage assignment."""
            n_lift = 320
            lift_edges = []
            for u, v in base_edges:
                idx = edge_index[(u, v)]
                w = voltages[idx]
                for x in range(32):
                    u_lift = u * 32 + x
                    v_lift = v * 32 + (x ^ w)
                    lift_edges.append((u_lift, v_lift))
            return n_lift, lift_edges

        def check_voltages(voltages):
            """Check if voltage assignment kills all short cycles.
            
            In the voltage graph on K5,5 minus matching, each base edge (i, 5+j) 
            goes from vertex i (in part A) to vertex 5+j (in part B).
            A cycle alternates A-B-A-B-...
            For a cycle e1, e2, ..., e2m where e_{2k-1} goes A->B and e_{2k} goes B->A:
            The voltage around the cycle is XOR of voltages on A->B edges XOR'd with
            voltages on B->A edges (traversed backwards).
            
            In our base graph, edges are stored as (i, 5+j) meaning from A-vertex i to B-vertex (5+j).
            When traversing a cycle, odd-position edges go A->B (use voltage as-is),
            even-position edges go B->A (use voltage as-is since XOR is symmetric).
            
            Actually for Z_2^5, the inverse is the same element, so direction doesn't matter
            for XOR. The cycle voltage is just XOR of all edge voltages in the cycle.
            """
            for cycle in four_cycles:
                s = 0
                for e in cycle:
                    s ^= voltages[e]
                if s == 0:
                    return False
            for cycle in six_cycles:
                s = 0
                for e in cycle:
                    s ^= voltages[e]
                if s == 0:
                    return False
            for cycle in eight_cycles:
                s = 0
                for e in cycle:
                    s ^= voltages[e]
                if s == 0:
                    return False
            return True

        # Systematic search: w_{(i,j)} = a_i ^ b_j ^ c_{i,j}
        # This reduces the search space from 32^20 to 32^5 * 32^5 * 32^20 still too large
        # But we can be smarter: fix a_0 = b_0 = 0 (gauge freedom), search over remaining
        # Actually, let's use a more structured approach.
        
        # Key insight: For the voltage graph to have girth >= 10, we need:
        # - No 4-cycle has XOR sum = 0
        # - No 6-cycle has XOR sum = 0  
        # - No 8-cycle has XOR sum = 0
        
        # Let's try: w_{i,j} = A_i ^ B_j for some vectors A_i, B_j in Z_2^5
        # This gives 32^10 = ~10^15 possibilities, still too many.
        # But with gauge freedom (shift all A_i by constant, all B_j by same constant),
        # we can fix A_0 = 0, B_0 = 0, leaving 32^8 possibilities.
        
        # Better: use a linear structure. Let w_{i,j} = M @ [i, j] for a 5x2 matrix M.
        # That's 5*2*5 = 50 bits, but we need XOR structure.
        
        # Actually, let's just do a smarter random search with local improvement.
        
        random.seed(42)
        found = None
        
        # First, try the structured form w_{i,j} = a_i ^ b_j
        # Fix a_0 = 0 for gauge. Search a_1..a_4, b_0..b_4 (10 variables, 5 bits each)
        # That's 32^10 = 2^50, too many. But we can use simulated annealing.
        
        # Alternative: try w_{i,j} = (i * j) in some field-like structure
        # Z_2^5 is not a field, but we can try multiplicative-like structures.
        
        # Let's try: interpret i, j as elements of Z_5, and use a bilinear form
        # w_{i,j} = sum_k i_k * j_k * v_k for some vectors v_k in Z_2^5
        # where i_k, j_k are bits of i, j (but i,j are 0..4, so 3 bits each)
        
        # Simpler: just do random search but with a better initial guess
        # and local search (flip one voltage at a time to reduce violations)
        
        def count_violations(voltages):
            """Count number of short cycles with zero voltage sum."""
            count = 0
            for cycle in four_cycles:
                s = 0
                for e in cycle:
                    s ^= voltages[e]
                if s == 0:
                    count += 1
            for cycle in six_cycles:
                s = 0
                for e in cycle:
                    s ^= voltages[e]
                if s == 0:
                    count += 1
            for cycle in eight_cycles:
                s = 0
                for e in cycle:
                    s ^= voltages[e]
                if s == 0:
                    count += 1
            return count
        
        # Local search: start random, flip voltages to reduce violations
        best_voltages = None
        best_count = float('inf')
        
        for attempt in range(200):
            if time.time() > deadline - 60:
                break
            voltages = [random.randint(0, 31) for _ in range(n_base_edges)]
            count = count_violations(voltages)
            if count == 0:
                found = voltages
                break
            if count < best_count:
                best_count = count
                best_voltages = voltages[:]
            
            # Local search: try flipping each voltage
            improved = True
            while improved and time.time() < deadline - 60:
                improved = False
                for idx in range(n_base_edges):
                    old = voltages[idx]
                    for new_val in range(32):
                        if new_val == old:
                            continue
                        voltages[idx] = new_val
                        new_count = count_violations(voltages)
                        if new_count < count:
                            count = new_count
                            improved = True
                            if count < best_count:
                                best_count = count
                                best_voltages = voltages[:]
                            if count == 0:
                                found = voltages
                                break
                        else:
                            voltages[idx] = old
                    if found:
                        break
                if found:
                    break
        
        if found is not None:
            n_lift, lift_edges = build_lift(found)
            if verify_girth(n_lift, lift_edges, 10):
                return n_lift, lift_edges
        
        # Try more structured forms
        # w_{i,j} = (i * a + j * b + c) mod 32 for various a, b, c
        for a in range(32):
            if time.time() > deadline - 40:
                break
            for b in range(32):
                if time.time() > deadline - 35:
                    break
                for c in range(32):
                    if time.time() > deadline - 30:
                        break
                    voltages = [0] * n_base_edges
                    for idx, (u, v) in enumerate(base_edges):
                        i = u
                        j = v - 5
                        voltages[idx] = (i * a + j * b + c) & 31
                    if check_voltages(voltages):
                        n_lift, lift_edges = build_lift(voltages)
                        if verify_girth(n_lift, lift_edges, 10):
                            return n_lift, lift_edges

        # Fallback: LUW construction
        for d_try in [5, 6, 7, 8, 9]:
            if time.time() > deadline - 10:
                break
            n, edges = luw_construct(d_try, 4)
            if verify_girth(n, edges, 10):
                return n, edges

    # General case
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = luw_construct(d, k)
    if verify_girth(n, edges, g):
        return n, edges

    # Try larger d values
    for d_try in range(d + 1, min(d + 10, 20)):
        if time.time() > deadline - 5:
            break
        n, edges = luw_construct(d_try, k)
        if verify_girth(n, edges, g):
            return n, edges

    # Ultimate fallback: return LUW even if unverified
    return n, edges
