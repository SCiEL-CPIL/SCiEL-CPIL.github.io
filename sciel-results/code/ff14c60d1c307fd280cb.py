import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # Keep 5s buffer for 120s limit

    def elapsed():
        return time.time() - start_time

    def deadline_hit():
        return elapsed() >= deadline

    # 1. Construct Polarity Graph of PG(2,5) - 31 vertices, C4-free
    q = 5
    # Points of PG(2,q): (x,y,z) up to scalar
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n_init = len(points)  # 31

    def is_orthogonal(p, qpt):
        x1, y1, z1 = p
        x2, y2, z2 = qpt
        return (x1 * y2 + y1 * x2 + z1 * z2) % q == 0

    adj = [set() for _ in range(n_init)]
    for i in range(n_init):
        for j in range(i + 1, n_init):
            if is_orthogonal(points[i], points[j]):
                adj[i].add(j)
                adj[j].add(i)

    # Build conflict structure using bitmasks for speed
    # conflict[i] = bitmask of vertices j such that adj[i] & adj[j] != empty
    # This is key: two vertices can't both be neighbors of a new vertex if they conflict
    
    n = n_init
    # Use bitmask representation for adj for fast alpha checking
    # adj_mask[i] = bitmask of neighbors of vertex i
    
    def build_adj_mask(adj):
        n_v = len(adj)
        masks = [0] * n_v
        for i in range(n_v):
            m = 0
            for j in adj[i]:
                m |= (1 << j)
            masks[i] = m
        return masks

    def build_conflict_mask(adj):
        """conflict_mask[i] = bitmask of j where adj[i] & adj[j] != empty"""
        n_v = len(adj)
        adj_mask = build_adj_mask(adj)
        conflict_mask = [0] * n_v
        for i in range(n_v):
            for j in range(i + 1, n_v):
                if adj_mask[i] & adj_mask[j]:
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
        return conflict_mask

    adj_mask = build_adj_mask(adj)
    conflict_mask = build_conflict_mask(adj)

    # Fast alpha checker using bitmasks
    def has_independent_set_size_k(adj_mask, k, n_v):
        """Returns True if there is an independent set of size k in graph with adj_mask."""
        if k > n_v:
            return False
        if k <= 0:
            return True
        if k == 1:
            return n_v >= 1

        sys.setrecursionlimit(10000)
        
        # Greedy lower bound first - if we can find k greedily, return True immediately
        # This is a quick positive check
        def greedy_find_k(adj_mask, k, n_v):
            remaining = (1 << n_v) - 1
            count = 0
            while remaining and count < k:
                # Pick vertex with minimum degree in remaining
                best_v = -1
                best_deg = 10**9
                rem = remaining
                while rem:
                    lsb = rem & (-rem)
                    v = lsb.bit_length() - 1
                    rem ^= lsb
                    deg = bin(adj_mask[v] & remaining).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                count += 1
                remaining &= ~(adj_mask[best_v] | (1 << best_v))
            return count >= k

        if greedy_find_k(adj_mask, k, n_v):
            return True

        # Exact branch and bound
        # Upper bound: maximum possible independent set in remaining
        # Simple bound: number of vertices in remaining
        # Better: use greedy coloring on complement? Too complex.
        # Use: if popcount(remaining) + current < k, prune
        
        def popcount(x):
            return bin(x).count('1')

        def branch(remaining_mask, current_size):
            if current_size >= k:
                return True
            if remaining_mask == 0:
                return False
            if current_size + popcount(remaining_mask) < k:
                return False

            # Pick vertex with highest degree in remaining (to maximize pruning)
            rem = remaining_mask
            best_v = -1
            best_deg = -1
            while rem:
                lsb = rem & (-rem)
                v = lsb.bit_length() - 1
                rem ^= lsb
                deg = bin(adj_mask[v] & remaining_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            # Option 1: Include best_v
            new_rem = remaining_mask & ~(adj_mask[best_v] | (1 << best_v))
            if branch(new_rem, current_size + 1):
                return True

            # Option 2: Exclude best_v
            new_rem2 = remaining_mask & ~(1 << best_v)
            if branch(new_rem2, current_size):
                return True

            return False

        full = (1 << n_v) - 1
        return branch(full, 0)

    # Helper: check if a set S (bitmask) is independent in conflict graph
    def is_conflict_independent(S_mask, conflict_mask, n_v):
        """Check if no two vertices in S_mask have a conflict."""
        rem = S_mask
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            # Check if any other vertex in S_mask conflicts with v
            if conflict_mask[v] & S_mask:
                return False
        return True

    # Main expansion loop
    best_n = n
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])

    while not deadline_hit():
        added = False
        attempts = 0
        max_attempts = 200

        while attempts < max_attempts and not deadline_hit():
            attempts += 1

            # Generate random independent set in conflict graph
            # Greedy with randomization
            S_mask = 0
            available = (1 << n) - 1

            # Shuffle vertices for randomness
            vertices = list(range(n))
            random.shuffle(vertices)

            for u in vertices:
                if not (available & (1 << u)):
                    continue
                # Check if u conflicts with any vertex already in S
                if conflict_mask[u] & S_mask:
                    continue
                S_mask |= (1 << u)
                # Remove u and its conflicts from available
                available &= ~(1 << u)
                available &= ~conflict_mask[u]

            if S_mask == 0:
                continue

            # Check if S is conflict-independent (should be by construction, but verify)
            if not is_conflict_independent(S_mask, conflict_mask, n):
                continue

            # Try adding new vertex with neighbors S
            new_n = n + 1
            new_v = n

            # Build new adj_mask
            new_adj_mask = list(adj_mask)
            new_adj_mask.append(0)  # new vertex

            # Add edges from new_v to vertices in S
            for u in range(n):
                if S_mask & (1 << u):
                    new_adj_mask[u] |= (1 << new_v)
                    new_adj_mask[new_v] |= (1 << u)

            # Check alpha: need no independent set of size 11
            # alpha(new_graph) = max(alpha(old_graph), 1 + alpha(old_graph - S))
            # We know alpha(old) <= 10 from previous steps
            # So we need 1 + alpha(old_graph restricted to V\S) <= 10
            # i.e., alpha(old_graph restricted to V\S) <= 9

            # Build restricted adj_mask for V\S
            # Vertices not in S
            not_S_mask = ((1 << n) - 1) & ~S_mask
            num_not_S = bin(not_S_mask).count('1')

            if num_not_S >= 10:
                # Need to check if there's an independent set of size 10 in old graph restricted to V\S
                # Build adj_mask for the subgraph
                # Map old indices to new indices
                old_to_new = {}
                new_idx = 0
                sub_adj_mask = []
                for i in range(n):
                    if not_S_mask & (1 << i):
                        old_to_new[i] = new_idx
                        new_idx += 1

                sub_n = new_idx
                sub_adj_mask = [0] * sub_n
                for i in range(n):
                    if not (not_S_mask & (1 << i)):
                        continue
                    ni = old_to_new[i]
                    for j in range(n):
                        if (not_S_mask & (1 << j)) and (adj_mask[i] & (1 << j)):
                            sub_adj_mask[ni] |= (1 << old_to_new[j])

                # Check if subgraph has independent set of size 10
                if has_independent_set_size_k(sub_adj_mask, 10, sub_n):
                    # Invalid: would create independent set of size 11 in new graph
                    continue
            # If num_not_S < 10, then 1 + alpha(V\S) <= 1 + 9 = 10 < 11, so OK

            # Valid! Update state
            adj_mask = new_adj_mask
            n = new_n

            # Update conflict_mask
            new_conflict_mask = list(conflict_mask)
            new_conflict_mask.append(0)

            # Pairs in S become conflicts (they now share new_v as common neighbor)
            S_list = []
            for u in range(n - 1):
                if S_mask & (1 << u):
                    S_list.append(u)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    u, w = S_list[i], S_list[j]
                    new_conflict_mask[u] |= (1 << w)
                    new_conflict_mask[w] |= (1 << u)

            # Conflicts involving new vertex new_v:
            # new_v and u conflict if adj_mask[u] & adj_mask[new_v] != empty
            # adj_mask[new_v] = S_mask
            # So new_v and u conflict if adj_mask[u] & S_mask != empty
            for u in range(n - 1):
                if adj_mask[u] & S_mask:
                    new_conflict_mask[u] |= (1 << new_v)
                    new_conflict_mask[new_v] |= (1 << u)

            conflict_mask = new_conflict_mask

            # Update best
            if n > best_n:
                best_n = n
                best_edges = []
                # Reconstruct edges from adj_mask
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_mask[u] & (1 << v):
                            best_edges.append([u, v])

            added = True
            break  # Break inner loop, try adding another vertex

        if not added:
            # Try single-vertex additions as fallback
            found_single = False
            for _ in range(20):
                if deadline_hit():
                    break
                u = random.randint(0, n - 1)
                S_mask = 1 << u

                # Single vertex is always conflict-independent
                new_n = n + 1
                new_v = n

                new_adj_mask = list(adj_mask)
                new_adj_mask.append(0)
                new_adj_mask[u] |= (1 << new_v)
                new_adj_mask[new_v] |= (1 << u)

                # Check alpha
                not_S_mask = ((1 << n) - 1) & ~S_mask
                num_not_S = bin(not_S_mask).count('1')

                if num_not_S >= 10:
                    old_to_new = {}
                    new_idx = 0
                    for i in range(n):
                        if not_S_mask & (1 << i):
                            old_to_new[i] = new_idx
                            new_idx += 1
                    sub_n = new_idx
                    sub_adj_mask = [0] * sub_n
                    for i in range(n):
                        if not (not_S_mask & (1 << i)):
                            continue
                        ni = old_to_new[i]
                        for j in range(n):
                            if (not_S_mask & (1 << j)) and (adj_mask[i] & (1 << j)):
                                sub_adj_mask[ni] |= (1 << old_to_new[j])
                    if has_independent_set_size_k(sub_adj_mask, 10, sub_n):
                        continue

                # Valid
                adj_mask = new_adj_mask
                n = new_n
                new_conflict_mask = list(conflict_mask)
                new_conflict_mask.append(0)
                for x in range(n - 1):
                    if adj_mask[x] & (1 << new_v):
                        new_conflict_mask[x] |= (1 << new_v)
                        new_conflict_mask[new_v] |= (1 << x)
                conflict_mask = new_conflict_mask

                if n > best_n:
                    best_n = n
                    best_edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj_mask[u] & (1 << v):
                                best_edges.append([u, v])
                found_single = True
                break

            if not found_single:
                break

    return {'n': best_n, 'edges': best_edges}
