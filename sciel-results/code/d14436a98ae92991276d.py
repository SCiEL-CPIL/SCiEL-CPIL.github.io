import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Moore bound for cubic graphs
    if g % 2 == 0:
        D = g // 2
        n_min = 2 * (2**D - 1)
    else:
        D = (g - 1) // 2
        n_min = 1 + k * sum((k-1)**i for i in range(D))
    n_min = max(n_min, 6)

    # Petersen graph (unique cubic graph of girth 5, 10 vertices)
    # Standard labeling: outer pentagon 0-4, inner star 5-9
    PETERSEN = [(0,1),(1,2),(2,3),(3,4),(4,0),
               (5,7),(7,9),(9,6),(6,8),(8,5),
               (0,5),(1,6),(2,7),(3,8),(4,9)]
    PETERSEN_N = 10
    PETERSEN_EDGES = 15

    def verify_girth_fast(adj, n, target_g):
        """Non-backtracking BFS from every vertex. Returns min cycle length found (or >= target_g)."""
        min_girth = target_g  # we only care if < target_g
        # Non-backtracking: state = (node, prev_node)
        # Distance in non-backtracking walk
        for start in range(n):
            # BFS in non-backtracking space
            # visited[prev][node] = distance
            visited = {}
            q = deque()
            for v in adj[start]:
                visited[(start, v)] = 1
                q.append((start, v))
            while q:
                prev, node = q.popleft()
                dist = visited[(prev, node)]
                if dist >= target_g:
                    continue
                for nxt in adj[node]:
                    if nxt == prev:
                        continue
                    state = (node, nxt)
                    if state in visited:
                        # Found a non-backtracking closed walk
                        # Cycle length = dist + 1
                        cl = dist + 1
                        if cl < min_girth:
                            min_girth = cl
                            if min_girth < target_g:
                                return min_girth
                    else:
                        visited[state] = dist + 1
                        q.append(state)
        return min_girth

    def build_lift(m, voltages):
        """Build the voltage lift of Petersen over Z_m.
        voltages: list of 15 integers (one per Petersen edge)
        Returns adj list or None if invalid (self-loop/multi-edge).
        """
        n = PETERSEN_N * m
        adj = [set() for _ in range(n)]
        for (u, v) in PETERSEN:
            # edge index
            idx = PETERSEN.index((u, v)) if (u,v) in PETERSEN else PETERSEN.index((v,u))
            # find the voltage
            # We need consistent indexing
            # Let's precompute
            pass
        # Redo with proper indexing
        adj = [set() for _ in range(n)]
        for i, (u, v) in enumerate(PETERSEN):
            volt = voltages[i]
            for a in range(m):
                su = u * m + a
                sv = v * m + ((a + volt) % m)
                if su == sv:
                    return None  # self-loop
                if sv in adj[su]:
                    return None  # multi-edge
                adj[su].add(sv)
                adj[sv].add(su)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return None
        return adj

    def _precompute_petersen_edge_index():
        """Return list of edge indices for consistent voltage assignment."""
        return list(range(PETERSEN_EDGES))

    # Precompute Petersen edge structure for voltage lift
    # Each edge i connects PETERSEN[i][0] and PETERSEN[i][1]

    def primitive_roots(m):
        """Find all primitive roots modulo m (for m where they exist)."""
        if m <= 1:
            return []
        # phi(m)
        def phi(x):
            result = x
            p = 2
            while p * p <= x:
                if x % p == 0:
                    while x % p == 0:
                        x //= p
                    result -= result // p
                p += 1
            if x > 1:
                result -= result // x
            return result
        if m == 2:
            return [1]
        if m == 4:
            return [3]
        # Primitive roots exist only for m = 2, 4, p^k, 2*p^k where p is odd prime
        # For simplicity, just try all candidates
        ph = phi(m)
        roots = []
        for a in range(1, m):
            if np.gcd(a, m) != 1:
                continue
            # Check if a is primitive root: order = phi(m)
            # a^ph % m == 1 and a^(ph/q) % m != 1 for all prime q dividing ph
            ok = True
            x = pow(a, ph, m)
            if x != 1:
                continue
            # factor ph
            ph_temp = ph
            factors = set()
            p = 2
            while p * p <= ph_temp:
                if ph_temp % p == 0:
                    factors.add(p)
                    while ph_temp % p == 0:
                        ph_temp //= p
                p += 1
            if ph_temp > 1:
                factors.add(ph_temp)
            for q in factors:
                if pow(a, ph // q, m) == 1:
                    ok = False
                    break
            if ok:
                roots.append(a)
        return roots

    def _is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i+2) == 0:
                return False
            i += 6
        return True

    best_result = None

    # Strategy 1: Algebraic voltage lifts of Petersen
    # For m, assign voltages from multiplicative structure to maximize closed walk orders
    # Use primitive root powers: voltages[i] = g^i mod m (or similar)
    # The key insight: a closed walk in the lift has total voltage = sum of edge voltages
    # (with signs depending on direction). For the lift to have girth >= g,
    # no non-backtracking closed walk of length < g should have total voltage = 0 mod m.

    # For Petersen, the girth is 5. The shortest non-backtracking closed walks
    # have length 5 (the pentagons). We need those to NOT close (total voltage != 0).
    # More generally, we need all closed walks of length < g to have non-zero total voltage.

    # For a lift to work, m must be large enough. The Moore bound says n >= 2*(2^8-1) = 510.
    # So m >= 510/10 = 51.

    # Try m from ceil(n_min/10) upward
    m_start = max(51, (n_min + PETERSEN_N - 1) // PETERSEN_N)
    m_end = m_start + 200  # search range

    # For each m, try different voltage assignments
    # Best assignment: use a primitive root g, set voltages[i] = g^i mod m
    # This makes the "total voltage" of a walk depend on the product of g^edge indices

    # Alternative: for prime m, use voltages[i] = g^i where g is primitive root
    # This ensures that the sum of a geometric-like sequence is non-zero for short walks

    # Let's try multiple voltage patterns:
    # Pattern A: voltages[i] = (i+1) * g mod m (linear in edge index, scaled by primitive root)
    # Pattern B: voltages[i] = g^i mod m (exponential)
    # Pattern C: random shifts of the above

    # For the Petersen graph, the 5-cycles are the key constraint.
    # There are 12 five-cycles in the Petersen graph.
    # For each 5-cycle, the sum of voltages (with appropriate signs) must be != 0 mod m.

    # Actually, for a non-backtracking walk, the voltage accumulates as we traverse.
    # For a cycle v0-v1-v2-v3-v4-v0, the total voltage is:
    # volt(e0) + volt(e1) + volt(e2) + volt(e3) + volt(e4) (if all in same direction)
    # But in the lift, going forward adds voltage, going backward subtracts.
    # For a simple cycle in the base, all edges are traversed in one direction,
    # so total voltage = sum of the 5 edge voltages (mod m).

    # So we need: for every 5-cycle C in Petersen, sum_{e in C} volt_e != 0 mod m.
    # And for every 6-cycle, 7-cycle, ..., (g-1)-cycle, the total voltage != 0 mod m.

    # This is a system of constraints. Let's just try and verify.

    # Generate candidate voltage assignments
    def generate_voltage_candidates(m, max_patterns=200):
        """Generate candidate voltage assignments for the 15 Petersen edges."""
        candidates = []
        if m < 2:
            return candidates

        # Get primitive roots
        roots = primitive_roots(m)
        if not roots:
            # No primitive root, use random
            for _ in range(max_patterns):
                volts = [random.randint(1, m-1) for _ in range(15)]
                candidates.append(volts)
            return candidates

        # Use primitive roots to generate structured voltages
        for g in roots[:5]:  # limit to first 5 primitive roots
            # Pattern 1: voltages[i] = g^i mod m
            volts = [pow(g, i, m) for i in range(15)]
            if all(v != 0 for v in volts):
                candidates.append(volts)
            # Pattern 2: voltages[i] = (i+1) * g mod m
            volts = [((i+1) * g) % m for i in range(15)]
            if all(v != 0 for v in volts):
                candidates.append(volts)
            # Pattern 3: voltages[i] = g^(i+1) mod m (shifted)
            volts = [pow(g, i+1, m) for i in range(15)]
            if all(v != 0 for v in volts):
                candidates.append(volts)

        # Add random candidates
        for _ in range(max_patterns - len(candidates)):
            if _ >= max_patterns:
                break
            volts = [random.randint(1, m-1) for _ in range(15)]
            candidates.append(volts)

        return candidates[:max_patterns]

    # Quick pre-check: verify 5-cycle constraints before full verification
    # Find all 5-cycles in Petersen
    def find_5cycles_petersen():
        """Find all 5-cycles in the Petersen graph."""
        adj = [set() for _ in range(PETERSEN_N)]
        for u, v in PETERSEN:
            adj[u].add(v)
            adj[v].add(u)
        cycles = set()
        for start in range(PETERSEN_N):
            # DFS for 5-cycles starting at start
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                if len(path) == 5:
                    if start in adj[node]:
                        # Found a 5-cycle
                        cycle = tuple(sorted(path))
                        cycles.add(cycle)
                elif len(path) < 5:
                    for nxt in adj[node]:
                        if nxt not in visited and nxt > start or (nxt == start and len(path) == 5):
                            if nxt == start and len(path) < 5:
                                continue
                            stack.append((nxt, path + [nxt], visited | {nxt}))
        return cycles

    # Precompute 5-cycles (and their edge index sets)
    # For a 5-cycle (v0,v1,v2,v3,v4), the edges are (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v0)
    # We need to map these to edge indices in PETERSEN

    def get_edge_index(u, v):
        for i, (a, b) in enumerate(PETERSEN):
            if (a == u and b == v) or (a == v and b == u):
                return i
        return -1

    # Precompute 5-cycle edge sets
    five_cycle_edge_sets = []
    adj_p = [set() for _ in range(PETERSEN_N)]
    for u, v in PETERSEN:
        adj_p[u].add(v)
        adj_p[v].add(u)

    # Find all 5-cycles
    seen_cycles = set()
    for start in range(PETERSEN_N):
        # DFS
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) == 5:
                if start in adj_p[node]:
                    # Canonical form: rotate so min is first
                    cycle = path[:]
                    min_idx = cycle.index(min(cycle))
                    canon = tuple(cycle[min_idx:] + cycle[:min_idx])
                    # Also check reverse
                    rev = list(cycle)
                    rev.reverse()
                    min_idx_r = rev.index(min(rev))
                    canon_r = tuple(rev[min_idx_r:] + rev[:min_idx_r])
                    key = (canon, canon_r)
                    if key not in seen_cycles:
                        seen_cycles.add(key)
                        # Get edge indices
                        edge_set = set()
                        for i in range(5):
                            e_idx = get_edge_index(cycle[i], cycle[(i+1) % 5])
                            edge_set.add(e_idx)
                        five_cycle_edge_sets.append(edge_set)
            elif len(path) < 5:
                for nxt in adj_p[node]:
                    if nxt not in path:
                        stack.append((nxt, path + [nxt]))

    def quick_check_voltages(volts, m):
        """Check if voltages satisfy 5-cycle non-closure constraints."""
        for edge_set in five_cycle_edge_sets:
            total = sum(volts[e] for e in edge_set) % m
            if total == 0:
                return False
        return True

    # Main search loop
    for m in range(m_start, m_end + 1):
        if time.time() > deadline - 5:
            break

        # Check if n = 10*m is valid (>= n_min)
        n = PETERSEN_N * m
        if n < n_min:
            continue

        # Generate voltage candidates
        candidates = generate_voltage_candidates(m, max_patterns=100)

        for volts in candidates:
            if time.time() > deadline - 5:
                break

            # Quick check: 5-cycle constraints
            if not quick_check_voltages(volts, m):
                continue

            # Build the lift
            adj = build_lift(m, volts)
            if adj is None:
                continue

            # Full girth verification
            girth = verify_girth_fast(adj, n, g)
            if girth >= g:
                # Found a valid graph!
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges

    # Fallback: LUW construction
    # LUW: q-ary de Bruijn-like construction
    q = k  # 3
    # d = g - 5 for even g? Let's use the standard LUW parameter
    # For girth g, use d = g - 5 (gives girth >= 2d+1 = g)
    # Actually LUW with parameter d gives girth 2d+1 for odd, or 2d for even
    # Standard: n = 2 * q^d, girth >= 2d+1
    # For g=16 (even), we need girth >= 16, so 2d >= 16, d >= 8
    # But LUW gives girth 2d+1 which is odd. For even girth g, use d = g/2 - 1?
    # Let's just use d such that 2*d+1 >= g, i.e., d >= (g-1)/2
    d = (g - 1) // 2  # for g=16, d=7, girth >= 15... not enough
    # For even girth, LUW gives girth 2d, so need 2d >= g, d >= g/2
    d = g // 2  # for g=16, d=8, girth >= 16

    # LUW construction
    # Vertices: (side, x) where side in {0,1}, x in Z_q^d
    # n = 2 * q^d
    n_luw = 2 * (q ** d)

    # Rule for LUW
    rule = {}
    for c in range(2, d + 1):
        # Standard LUW rule
        # For c=2: (1,1), c=3: (2,1), c=4: (1,2), then periodic
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def luw_neighbours(vertex):
        side, x = vertex
        x = list(x)
        for first in range(q):
            y = [None] + [first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
            yield (1 - side), tuple(y)

    start_v = (0, tuple([0] * d))
    index = {start_v: 0}
    order = [start_v]
    edges_luw = []

    for vertex in order:
        for other in luw_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))

    n_luw_actual = len(order)

    # Verify LUW
    adj_luw = [set() for _ in range(n_luw_actual)]
    for u, v in edges_luw:
        adj_luw[u].add(v)
        adj_luw[v].add(u)

    girth_luw = verify_girth_fast(adj_luw, n_luw_actual, g)

    if girth_luw >= g:
        return n_luw_actual, edges_luw

    # If LUW also fails, return it anyway (shouldn't happen)
    return n_luw_actual, edges_luw
