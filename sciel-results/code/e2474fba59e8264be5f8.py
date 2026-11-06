"""Construct a C5-free graph with no independent set of 9, maximizing n."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Build adjacency bitmasks from edge list
    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # Check for C5 using bitsets
    def has_c5(adj, n):
        for u in range(n):
            nu = adj[u]
            v_mask = nu
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit

                a_mask = nu & ~(1 << v)
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit

                    b_mask = adj[a] & ~(1 << u) & ~(1 << v)
                    while b_mask:
                        b_bit = b_mask & (-b_mask)
                        b = b_bit.bit_length() - 1
                        b_mask ^= b_bit

                        c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                        if c_mask & adj[v]:
                            return True
        return False

    # Check for independent set of size 9
    def has_independent_set_9(adj, n):
        def can_find(available, count):
            if count == 0:
                return True
            if available == 0:
                return False
            if bin(available).count('1') < count:
                return False

            # Pick vertex with minimum degree in available
            best_v = -1
            best_excluded = 0
            avail = available
            while avail:
                v_bit = avail & (-avail)
                v = v_bit.bit_length() - 1
                avail ^= v_bit
                adj_in_avail = available & adj[v]
                excluded = bin(adj_in_avail).count('1')
                if best_v == -1 or excluded < best_excluded:
                    best_excluded = excluded
                    best_v = v
                if excluded == 0:
                    break

            if best_v == -1:
                return False

            v = best_v
            v_bit = 1 << v

            # Option 1: include v
            new_avail = available & ~adj[v] & ~v_bit
            if can_find(new_avail, count - 1):
                return True

            # Option 2: exclude v
            if can_find(available & ~v_bit, count):
                return True

            return False

        return can_find((1 << n) - 1, 9)

    def is_valid(n, adj):
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges

    # Known valid graph: 8 disjoint K4's -> n=32
    def make_8k4():
        n = 32
        adj = [0] * n
        for b in range(8):
            for i in range(4):
                for j in range(i + 1, 4):
                    u = 4 * b + i
                    v = 4 * b + j
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # Try to construct n=35 graph: 8 K4s + 3 extra vertices
    # Each extra vertex connects to all 4 of a distinct K4 block, and 1 vertex in each other block
    def try_construct_35():
        n = 35
        adj = make_8k4()

        # Extra vertices: 32, 33, 34
        # v32 connects to all of B0 (vertices 0-3) and 1 vertex from each other block
        # v33 connects to all of B1 (vertices 4-7) and 1 vertex from each other block
        # v34 connects to all of B2 (vertices 8-11) and 1 vertex from each other block

        for ex_idx, ex in enumerate([32, 33, 34]):
            # Connect to all 4 vertices of block ex_idx
            base = 4 * ex_idx
            for i in range(4):
                adj[ex] |= (1 << (base + i))
                adj[base + i] |= (1 << ex)

            # Connect to 1 vertex from each other block
            for b in range(8):
                if b == ex_idx:
                    continue
                base2 = 4 * b
                # Pick vertex 0 of that block
                v = base2
                adj[ex] |= (1 << v)
                adj[v] |= (1 << ex)

        return adj

    def try_construct_34():
        n = 34
        adj = make_8k4()

        # Extra vertices: 32, 33
        for ex_idx, ex in enumerate([32, 33]):
            base = 4 * ex_idx
            for i in range(4):
                adj[ex] |= (1 << (base + i))
                adj[base + i] |= (1 << ex)

            for b in range(8):
                if b == ex_idx:
                    continue
                base2 = 4 * b
                v = base2
                adj[ex] |= (1 << v)
                adj[v] |= (1 << ex)

        return adj

    def try_construct_33():
        n = 33
        adj = make_8k4()

        # Extra vertex: 32
        ex = 32
        base = 0
        for i in range(4):
            adj[ex] |= (1 << (base + i))
            adj[base + i] |= (1 << ex)

        for b in range(1, 8):
            base2 = 4 * b
            v = base2
            adj[ex] |= (1 << v)
            adj[v] |= (1 << ex)

        return adj

    # First try deterministic constructions
    best_result = None

    for target_n, builder in [(35, try_construct_35), (34, try_construct_34), (33, try_construct_33)]:
        if time.time() > deadline:
            break
        adj = builder()
        if is_valid(target_n, adj):
            edges = adj_to_edges(adj, target_n)
            best_result = {'n': target_n, 'edges': edges}
            break

    # If deterministic constructions fail, try randomized search
    if best_result is None or best_result['n'] < 35:
        for target_n in [35, 34, 33]:
            if time.time() > deadline:
                break
            if best_result and best_result['n'] >= target_n:
                break

            found = False
            num_attempts = 5000
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    break

                # Start with 8 K4s
                adj = make_8k4()
                n = target_n

                # Add extra vertices with random connections
                for ex in range(32, n):
                    # Choose 1-2 blocks to connect densely
                    num_dense_blocks = random.randint(1, 2)
                    dense_blocks = random.sample(range(8), num_dense_blocks)

                    for b in dense_blocks:
                        base = 4 * b
                        num_conn = random.randint(3, 4)
                        targets = random.sample(range(4), num_conn)
                        for t in targets:
                            v = base + t
                            adj[ex] |= (1 << v)
                            adj[v] |= (1 << ex)

                    # Connect to 1 vertex from each other block
                    for b in range(8):
                        if b in dense_blocks:
                            continue
                        base2 = 4 * b
                        if random.random() < 0.8:  # 80% chance to connect
                            v = 4 * b + random.randint(0, 3)
                            adj[ex] |= (1 << v)
                            adj[v] |= (1 << ex)

                # Also add edges between extra vertices if multiple
                for i in range(32, n):
                    for j in range(i + 1, n):
                        if random.random() < 0.3:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)

                if has_c5(adj, n):
                    continue
                if has_independent_set_9(adj, n):
                    continue

                edges = adj_to_edges(adj, n)
                best_result = {'n': n, 'edges': edges}
                found = True
                break

            if found:
                break

    # Fallback: return known n=32 graph
    if best_result is None:
        adj = make_8k4()
        edges = adj_to_edges(adj, 32)
        best_result = {'n': 32, 'edges': edges}

    return best_result

# EVOLVE-BLOCK-END
