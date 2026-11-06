"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Known valid graph: 8 disjoint K4's -> n=32, alpha=8 < 9, no C5
    known_32 = {
        'n': 32,
        'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    }

    def build_adj(n, edges):
        """Build adjacency bitmasks from edge list."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5_fast(adj, n):
        """Fast C5 detection using the characterization:
        A C5 exists iff there exist distinct vertices a,b,c such that
        b ∈ N(a) ∩ N(c), and there exist u ∈ N(a)\\{b,c} and v ∈ N(c)\\{a,b}
        with u ~ v and {u,v} ∩ {a,b,c} = ∅.
        
        For each pair (a,c) sharing at least one common neighbor b,
        check if (N(a) \\ {b,c}) has an edge to (N(c) \\ {a,b} \\ {a,b,c}).
        """
        # Precompute: for each vertex, its neighbor mask
        for a in range(n):
            na = adj[a]
            # For each c that shares a common neighbor with a
            c_mask = na  # candidates for c: neighbors of a
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                if c == a:
                    continue
                nc = adj[c]
                # Common neighbors of a and c
                common = na & nc & ~(1 << a) & ~(1 << c)
                if common == 0:
                    continue
                # For each b in common, check the condition
                # We need: ∃ u ∈ N(a)\{b,c}, ∃ v ∈ N(c)\{a,b} with u~v, u,v ∉ {a,b,c}
                # u_mask = N(a) \ {a,b,c} = na & ~(1<<a) & ~(1<<b) & ~(1<<c)
                # v_mask = N(c) \ {a,b,c} = nc & ~(1<<a) & ~(1<<b) & ~(1<<c)
                # Check if any u in u_mask is adjacent to any v in v_mask
                b_mask = common
                while b_mask:
                    b_bit = b_mask & (-b_mask)
                    b = b_bit.bit_length() - 1
                    b_mask ^= b_bit
                    u_mask = na & ~(1 << a) & ~(1 << b) & ~(1 << c)
                    v_mask = nc & ~(1 << a) & ~(1 << b) & ~(1 << c)
                    if u_mask == 0 or v_mask == 0:
                        continue
                    # Check if any vertex in u_mask is adjacent to any vertex in v_mask
                    # This is: does the subgraph induced by u_mask ∪ v_mask have an edge crossing?
                    # Equivalent: for some u in u_mask, adj[u] & v_mask != 0
                    # Optimize: check if union of adj[u] for u in u_mask intersects v_mask
                    # But that's expensive. Instead: iterate over the smaller set.
                    # For small graphs (n≤35), this is fine.
                    um = u_mask
                    while um:
                        u_bit = um & (-um)
                        u = u_bit.bit_length() - 1
                        um ^= u_bit
                        if adj[u] & v_mask:
                            return True
        return False

    def has_c5(adj, n):
        """Check if graph contains a 5-cycle. Returns True if found."""
        # Use the fast method
        return has_c5_fast(adj, n)

    def has_independent_set_9(adj, n):
        """Check if graph has an independent set of size 9 using bitset backtracking
        with branch-and-bound."""

        # Precompute popcount for speed (n <= 35)
        popcount_cache = {}
        def popcount(x):
            if x in popcount_cache:
                return popcount_cache[x]
            c = bin(x).count('1')
            popcount_cache[x] = c
            return c

        def can_find(available, count):
            if count == 0:
                return True
            if available == 0:
                return False
            avail_count = popcount(available)
            if avail_count < count:
                return False

            # Pick vertex with minimum degree in available (excludes fewest)
            best_v = -1
            best_excluded = n + 1
            avail = available
            while avail:
                v_bit = avail & (-avail)
                v = v_bit.bit_length() - 1
                avail ^= v_bit
                adj_in_avail = available & adj[v]
                excluded = popcount(adj_in_avail)
                if excluded < best_excluded:
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
        """Check if graph is C5-free and has no independent set of 9."""
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True

    def adj_to_edges(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges

    best_result = known_32

    # Strategy 0: Deterministic algebraic construction
    # Try to build a C5-free graph on 33-35 vertices using structured patterns
    if time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break
            found = False
            # Pattern: 8 blocks of K4 + extra vertices connected to specific blocks
            # Key: extra vertices must connect to at most 2 vertices per block to avoid C5
            for block_pattern in range(20):
                if time.time() > deadline:
                    break
                adj = [0] * target_n
                # 8 blocks of K4
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

                extra = target_n - 32
                # Connect extra vertices: 1 vertex in each block for alpha < 9
                for ex_idx in range(extra):
                    ex = 32 + ex_idx
                    for b in range(8):
                        t = (ex_idx + b + block_pattern) % 4
                        v = 4 * b + t
                        adj[ex] |= (1 << v)
                        adj[v] |= (1 << ex)

                # Also connect extra vertices to each other if multiple
                if extra > 1:
                    for i in range(extra):
                        for j in range(i + 1, extra):
                            adj[32 + i] |= (1 << (32 + j))
                            adj[32 + j] |= (1 << (32 + i))

                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue

                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break
            if found:
                break

    # Strategy 1: Extend 8xK4 by adding vertices with careful connections
    if best_result['n'] == 32 and time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break

            extra = target_n - 32
            num_attempts = 20000
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    break

                # Initialize: 8xK4 structure
                adj = [0] * target_n
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

                # Add edges from extra vertices to existing vertices
                # Key: connect to exactly 1 vertex in each of 8 blocks to ensure
                # no independent set of 9. Verify C5-freeness after construction.
                for ex in range(32, target_n):
                    for b in range(8):
                        t = random.randint(0, 3)
                        v = 4 * b + t
                        adj[ex] |= (1 << v)
                        adj[v] |= (1 << ex)

                # Connect extra vertices to each other
                if extra > 1:
                    for i in range(extra):
                        for j in range(i + 1, extra):
                            if random.random() < 0.5:
                                adj[32 + i] |= (1 << (32 + j))
                                adj[32 + j] |= (1 << (32 + i))

                # Check validity
                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue

                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                break

    # Strategy 2: Local search - start from valid n=32 graph and modify
    if best_result['n'] == 32 and time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break

            found = False
            for attempt in range(5000):
                if time.time() > deadline:
                    break

                # Start with 8xK4 + extra vertices with limited connections
                adj = [0] * target_n
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

                # Add random edges from extra vertices (conservative)
                # Connect to 1 vertex in each block to ensure alpha < 9
                for ex in range(32, target_n):
                    for b in range(8):
                        t = random.randint(0, 3)
                        v = 4 * b + t
                        adj[ex] |= (1 << v)
                        adj[v] |= (1 << ex)

                # Check if valid directly
                if has_c5(adj, target_n):
                    # Try removing edges to fix C5
                    adj_copy = adj[:]
                    for _ in range(30):
                        if time.time() > deadline:
                            break
                        if not has_c5(adj_copy, target_n):
                            break
                        # Remove a random edge
                        all_edges = []
                        for i in range(target_n):
                            for j in range(i + 1, target_n):
                                if adj_copy[i] & (1 << j):
                                    all_edges.append((i, j))
                        if not all_edges:
                            break
                        u, v = random.choice(all_edges)
                        adj_copy[u] &= ~(1 << v)
                        adj_copy[v] &= ~(1 << u)

                    adj = adj_copy

                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue

                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break

            if found:
                break

    # Strategy 3: Pure random search with low density
    if best_result['n'] == 32 and time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break

            found = False
            for attempt in range(8000):
                if time.time() > deadline:
                    break

                # Random graph with low density
                p = random.uniform(0.06, 0.18)

                adj = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)

                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue

                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break

            if found:
                break

    return best_result

# EVOLVE-BLOCK-END
