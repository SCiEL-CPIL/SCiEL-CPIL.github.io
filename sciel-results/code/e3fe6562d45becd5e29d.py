import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def check_c4_fast(adj_bits, n, u, v):
        """Check if adding edge (u,v) creates a C4 using bitmasks.
        A C4 u-x-y-v-u exists iff there exist x in N(u), y in N(v) with x!=y and x~y."""
        nu = adj_bits[u]
        nv = adj_bits[v]
        
        # Iterate over smaller neighbor set
        if nu.bit_count() < nv.bit_count():
            x_bits = nu
            while x_bits:
                lsb = x_bits & (-x_bits)
                x = lsb.bit_length() - 1
                x_bits ^= lsb
                # Check if N(x) intersects (N(v) \ {x})
                if adj_bits[x] & (nv & ~(1 << x)):
                    return True
        else:
            y_bits = nv
            while y_bits:
                lsb = y_bits & (-y_bits)
                y = lsb.bit_length() - 1
                y_bits ^= lsb
                # Check if N(y) intersects (N(u) \ {y})
                if adj_bits[y] & (nu & ~(1 << y)):
                    return True
        return False

    def is_c4_free(adj_bits, n):
        """Verify no C4 exists."""
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj_bits[i] & (1 << j)):
                    common = adj_bits[i] & adj_bits[j]
                    if common.bit_count() >= 2:
                        return False
        return True

    def to_edge_list(adj_bits, n):
        edges = []
        for a in range(n):
            for b in range(a + 1, n):
                if adj_bits[a] & (1 << b):
                    edges.append([a, b])
        return edges

    def build_polarity_seed(n, rng):
        """Build a C4-free seed using polarity graph ideas.
        Uses a structured approach: partition vertices into groups and create
        a bipartite-like structure that avoids C4s."""
        adj_bits = [0] * n
        
        # Strategy: Create a graph based on quadratic residues or similar structure
        # For small n, use a greedy approach with strong structural bias
        
        # Divide into two halves for a bipartite-like structure
        mid = n // 2
        
        # Create edges between vertices in a structured way
        # Use modular arithmetic to avoid C4s
        for i in range(n):
            for j in range(i + 1, n):
                # Use a hash-like function to determine edge inclusion
                # This creates a pseudo-random but structured pattern
                h = (i * 31 + j * 17 + i * j * 7) % 100
                if h < 35:  # ~35% edge probability
                    if not check_c4_fast(adj_bits, n, i, j):
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
        
        return adj_bits

    def greedy_build(n, min_deg_req, time_limit, seed):
        """Build a C4-free graph with min degree >= min_deg_req."""
        t0 = time.time()
        rng = random.Random(seed)

        # Start with polarity-inspired seed
        adj_bits = build_polarity_seed(n, rng)

        def add_edge(u, v):
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)

        def remove_edge(u, v):
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)

        def can_add_edge(u, v):
            if u == v:
                return False
            if adj_bits[u] & (1 << v):
                return False
            return not check_c4_fast(adj_bits, n, u, v)

        def min_degree():
            return min(adj_bits[i].bit_count() for i in range(n))

        def get_deg(i):
            return adj_bits[i].bit_count()

        stuck_count = 0
        repair_budget = 100

        while time.time() - t0 < time_limit:
            md = min_degree()
            if md >= min_deg_req:
                if is_c4_free(adj_bits, n):
                    return {'n': n, 'edges': to_edge_list(adj_bits, n)}
                return None

            # Find lowest-degree vertex
            low_v = min(range(n), key=lambda i: adj_bits[i].bit_count())
            low_deg = adj_bits[low_v].bit_count()

            if low_deg < min_deg_req:
                candidates = [v for v in range(n) if v != low_v and not (adj_bits[low_v] & (1 << v))]
                rng.shuffle(candidates)

                added = False
                for v in candidates:
                    if can_add_edge(low_v, v):
                        add_edge(low_v, v)
                        added = True
                        stuck_count = 0
                        break

                if not added:
                    stuck_count += 1
                    if stuck_count >= 3 and repair_budget > 0:
                        repair_budget -= 1

                        # Find edges to remove to unblock
                        low_verts = [i for i in range(n) if adj_bits[i].bit_count() < min_deg_req]
                        if low_verts and any(adj_bits[i].bit_count() > 0 for i in range(n)):
                            # Try removing edges not incident to low vertices
                            safe_edges = []
                            for a in range(n):
                                if a in low_verts:
                                    continue
                                for b in range(a + 1, n):
                                    if b in low_verts:
                                        continue
                                    if adj_bits[a] & (1 << b):
                                        safe_edges.append((a, b))
                            
                            if safe_edges:
                                u, w = rng.choice(safe_edges)
                                du = adj_bits[u].bit_count() - 1
                                dw = adj_bits[w].bit_count() - 1
                                if du >= min_deg_req and dw >= min_deg_req:
                                    remove_edge(u, w)
                                    stuck_count = 0
                            else:
                                # Try removing any safe edge
                                all_safe = [(a, b) for a in range(n) for b in range(a + 1, n)
                                          if adj_bits[a] & (1 << b) and 
                                          adj_bits[a].bit_count() - 1 >= min_deg_req and
                                          adj_bits[b].bit_count() - 1 >= min_deg_req]
                                if all_safe:
                                    u, w = rng.choice(all_safe)
                                    remove_edge(u, w)
                                    stuck_count = 0
                    elif stuck_count > 50:
                        break
        return None

    best_result = None

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline - 3:
            break

        min_deg_req = max(1, n - 39)
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break

        # Allocate time: more attempts for larger n
        num_attempts = min(10, max(3, int(remaining / 4.0)))
        time_per_attempt = (remaining - 2) / max(1, num_attempts)

        for attempt in range(num_attempts):
            if time.time() - start_time > deadline - 2:
                break

            seed = 42 + attempt * 104729 + n * 7919
            result = greedy_build(n, min_deg_req, time_per_attempt, seed)
            if result is not None:
                return result

    # Fallback: small valid graph
    return {'n': 3, 'edges': [[0, 1], [1, 2], [0, 2]]}
