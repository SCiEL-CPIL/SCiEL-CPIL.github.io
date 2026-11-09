import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    # Fast K4 counter using bit masks
    def count_k4(n, masks):
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            k4_count += 1
        return k4_count

    # Fast independent 6-set counter
    def count_ind6(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [full_mask & ~(masks[u] | (1 << u)) for u in range(n)]
        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        ind6_count += 1
        return ind6_count

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def edges_from_adj(n, adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    def is_valid(n, adj):
        masks = build_masks(n, adj)
        return count_k4(n, masks) == 0 and count_ind6(n, masks) == 0

    # Finite field GF(q) arithmetic helpers
    def gf_q_add(a, b, q):
        return (a + b) % q

    def gf_q_sub(a, b, q):
        return (a - b) % q

    def gf_q_mul(a, b, q):
        return (a * b) % q

    def gf_q_inv(a, q):
        # For prime q, use Fermat's little theorem
        return pow(a, q - 2, q)

    # Quadratic form: Q(x, y) = x^2 + a*x*y + y^2 over GF(q)
    def quad_form(x, y, a, q):
        return (x * x + a * x * y + y * y) % q

    # Multiplicative character: 1 if x is a quadratic residue mod p (and x != 0), -1 otherwise
    def mult_char(x, p):
        if x == 0:
            return 0
        return 1 if pow(x, (p - 1) // 2, p) == 1 else -1

    # Paley graph on prime p vertices
    def paley_graph(p):
        adj = np.zeros((p, p), dtype=bool)
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if mult_char(diff, p) == 1:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    # Generalized Paley-like graph using quadratic form
    def quad_form_graph(n, q, a_val):
        """
        Map vertices 0..n-1 to pairs (x, y) in GF(q) x GF(q) where q*q >= n.
        Edge between (x1,y1) and (x2,y2) if quad_form(x1-x2, y1-y2, a_val, q) is a QR.
        """
        # Map vertex i to (i // q, i % q)
        def to_point(i):
            return (i // q, i % q)

        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            x1, y1 = to_point(i)
            for j in range(i + 1, n):
                x2, y2 = to_point(j)
                dx = (x1 - x2) % q
                dy = (y1 - y2) % q
                val = quad_form(dx, dy, a_val, q)
                if mult_char(val, q) == 1:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    # Search for valid finite-field-based constructions
    def search_finite_field(n, max_time=20.0):
        start = time.time()
        deadline_local = start + max_time

        # Try Paley graphs for prime n
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        if is_prime(n):
            adj = paley_graph(n)
            if is_valid(n, adj):
                return edges_from_adj(n, adj)

        # Try quadratic form graphs with various q and a values
        # We need q*q >= n, so q >= ceil(sqrt(n))
        q_min = int(np.ceil(np.sqrt(n)))
        for q in range(q_min, min(q_min + 10, n)):
            if not is_prime(q):
                continue
            if time.time() > deadline_local:
                break
            for a_val in range(1, q):
                if time.time() > deadline_local:
                    break
                adj = quad_form_graph(n, q, a_val)
                if is_valid(n, adj):
                    return edges_from_adj(n, adj)

        return None

    # Incremental construction: add vertices one by one
    def incremental_construct(n, max_time=30.0):
        """
        Build graph incrementally: start with a small valid core, then add
        vertices one at a time, choosing edges to maintain K4-free and ind6-free.
        """
        start = time.time()
        deadline_local = start + max_time

        # Start with Clebsch graph (16 vertices) as seed
        clebsch_edges = [
            (0, 6), (0, 7), (0, 10), (0, 11),
            (1, 5), (1, 8), (1, 10), (1, 13),
            (2, 4), (2, 9), (2, 11), (2, 13),
            (3, 4), (3, 7), (3, 9), (3, 12),
            (5, 6), (5, 8), (5, 12), (5, 15),
            (6, 7), (6, 9), (6, 14), (6, 15),
            (7, 8), (7, 11), (7, 14), (7, 15),
            (8, 9), (8, 12), (8, 15),
            (9, 10), (9, 14),
            (10, 11), (10, 13), (10, 14),
            (11, 12), (11, 14), (11, 15),
            (12, 13), (12, 15),
            (13, 14), (13, 15)
        ]

        if n <= 16:
            adj = np.zeros((n, n), dtype=bool)
            for u, v in clebsch_edges:
                if u < n and v < n:
                    adj[u, v] = True
                    adj[v, u] = True
            if is_valid(n, adj):
                return edges_from_adj(n, adj)
            return None

        adj = np.zeros((n, n), dtype=bool)
        for u, v in clebsch_edges:
            adj[u, v] = True
            adj[v, u] = True

        # Add vertices 16 to n-1 one by one
        for new_v in range(16, n):
            if time.time() > deadline_local:
                break

            # Try multiple random edge patterns for the new vertex
            best_pattern = None
            best_score = float('inf')

            for attempt in range(200):
                if time.time() > deadline_local:
                    break

                # Choose random neighbors for new_v among existing vertices
                pattern = np.zeros(n, dtype=bool)
                for u in range(new_v):
                    # Probability of connecting based on some heuristic
                    # Higher probability for vertices with lower degree
                    deg_u = np.sum(adj[u, :new_v])
                    p_connect = 0.3 + 0.2 * (new_v - deg_u) / new_v
                    if random.random() < p_connect:
                        pattern[u] = True

                # Tentatively add edges
                for u in range(new_v):
                    if pattern[u]:
                        adj[new_v, u] = True
                        adj[u, new_v] = True

                # Check validity
                masks = build_masks(n, adj)
                k4c = count_k4(n, masks)
                ind6c = count_ind6(n, masks)
                score = k4c * 100 + ind6c

                if score < best_score:
                    best_score = score
                    best_pattern = pattern.copy()

                if k4c == 0 and ind6c == 0:
                    return edges_from_adj(n, adj)

                # Remove tentative edges
                for u in range(new_v):
                    adj[new_v, u] = False
                    adj[u, new_v] = False

            # Apply best pattern
            if best_pattern is not None:
                for u in range(new_v):
                    if best_pattern[u]:
                        adj[new_v, u] = True
                        adj[u, new_v] = True

        # Final check
        if is_valid(n, adj):
            return edges_from_adj(n, adj)
        return None

    # Simulated annealing with better move selection
    def simulated_annealing(n, max_time=40.0):
        start = time.time()
        deadline_local = start + max_time

        # Initialize with random graph
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    adj[i, j] = True
                    adj[j, i] = True

        masks = build_masks(n, adj)
        current_cost = count_k4(n, masks) * 100 + count_ind6(n, masks)

        if current_cost == 0:
            return edges_from_adj(n, adj)

        T = 5.0
        best_cost = current_cost
        best_adj = adj.copy()
        no_improve = 0
        max_no_improve = 2000

        while time.time() < deadline_local and no_improve < max_no_improve:
            # Choose a move: add or remove a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue

            old = adj[u, v]
            adj[u, v] = not old
            adj[v, u] = not old

            # Update masks
            if adj[u, v]:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            else:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)

            new_cost = count_k4(n, masks) * 100 + count_ind6(n, masks)

            if new_cost < current_cost:
                current_cost = new_cost
                no_improve = 0
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj.copy()
                    if new_cost == 0:
                        return edges_from_adj(n, adj)
            else:
                delta = new_cost - current_cost
                T = max(0.1, T * 0.9995)
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    # Revert
                    adj[u, v] = old
                    adj[v, u] = old
                    if old:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                    else:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                no_improve += 1

        # Return best found
        if best_cost == 0:
            return edges_from_adj(n, best_adj)
        return None

    # Baseline: Clebsch graph (16 vertices)
    clebsch_edges = [
        (0, 6), (0, 7), (0, 10), (0, 11),
        (1, 5), (1, 8), (1, 10), (1, 13),
        (2, 4), (2, 9), (2, 11), (2, 13),
        (3, 4), (3, 7), (3, 9), (3, 12),
        (5, 6), (5, 8), (5, 12), (5, 15),
        (6, 7), (6, 9), (6, 14), (6, 15),
        (7, 8), (7, 11), (7, 14), (7, 15),
        (8, 9), (8, 12), (8, 15),
        (9, 10), (9, 14),
        (10, 11), (10, 13), (10, 14),
        (11, 12), (11, 14), (11, 15),
        (12, 13), (12, 15),
        (13, 14), (13, 15)
    ]
    best_n = 16
    best_edges = clebsch_edges

    # Verify baseline
    adj_16 = np.zeros((16, 16), dtype=bool)
    for u, v in clebsch_edges:
        adj_16[u, v] = True
        adj_16[v, u] = True
    if not is_valid(16, adj_16):
        best_n = 5
        best_edges = []

    # Try to find larger graphs
    # Strategy: try finite field constructions first (deterministic, fast),
    # then incremental, then simulated annealing

    # Try finite field constructions for n = 17 to 35
    for target_n in range(17, 36):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 3.0:
            break

        result = search_finite_field(target_n, max_time=min(5.0, remaining))
        if result is not None:
            best_n = target_n
            best_edges = result
            if best_n >= 35:
                break

    # If we didn't find large enough, try incremental construction
    if best_n < 30:
        for target_n in range(best_n + 1, 36):
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 5.0:
                break

            result = incremental_construct(target_n, max_time=min(15.0, remaining))
            if result is not None:
                best_n = target_n
                best_edges = result
                if best_n >= 35:
                    break

    # If still not large enough, try simulated annealing
    if best_n < 30 and time.time() < deadline - 10:
        for target_n in range(best_n + 1, min(best_n + 6, 36)):
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 10.0:
                break

            # Multiple restarts
            found = False
            for restart in range(3):
                if time.time() > deadline or found:
                    break
                result = simulated_annealing(target_n, max_time=min(10.0, deadline - time.time()))
                if result is not None:
                    found = True
                    best_n = target_n
                    best_edges = result
                    break

            if found and best_n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}
