"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct():
    """Construct a graph with no K5 and no independent set of size 5."""
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 300s limit

    # Bitmask-based violation finder
    def find_violation(adj, n):
        """Find K5 or independent set of size 5. Returns (type, vertices) or None."""
        # Check for K5
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    w = 0
                    while common:
                        w_bit = common & -common
                        w = w_bit.bit_length() - 1
                        common ^= w_bit
                        if w <= v:
                            continue
                        common2 = common & adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                        x = 0
                        while common2:
                            x_bit = common2 & -common2
                            x = x_bit.bit_length() - 1
                            common2 ^= x_bit
                            if x <= w:
                                continue
                            common3 = common2 & adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                            if common3:
                                y_bit = common3 & -common3
                                y = y_bit.bit_length() - 1
                                return ('clique', [u, v, w, x, y])

        # Check for Independent Set of size 5
        non_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            non_adj[i] = all_mask & ~(1 << i) & ~adj[i]

        for u in range(n):
            for v in range(u + 1, n):
                if non_adj[u] & (1 << v):
                    common = non_adj[u] & non_adj[v]
                    w = 0
                    while common:
                        w_bit = common & -common
                        w = w_bit.bit_length() - 1
                        common ^= w_bit
                        if w <= v:
                            continue
                        common2 = common & non_adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                        x = 0
                        while common2:
                            x_bit = common2 & -common2
                            x = x_bit.bit_length() - 1
                            common2 ^= x_bit
                            if x <= w:
                                continue
                            common3 = common2 & non_adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                            if common3:
                                y_bit = common3 & -common3
                                y = y_bit.bit_length() - 1
                                return ('independent', [u, v, w, x, y])

        return None

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    current_n = q
    best_n = q
    best_adj = adj[:]

    # Try to grow the graph vertex by vertex
    for target_n in range(current_n + 1, 47):
        if time.time() > deadline - 30:
            break

        success = False
        # Try multiple random connection patterns for the new vertex
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break

            # Create new graph with new vertex at index target_n - 1
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = adj[i]

            new_idx = target_n - 1
            # Randomly connect new vertex to existing vertices
            # Use probability around 0.5 for balanced approach
            for i in range(current_n):
                if random.random() < 0.5:
                    new_adj[i] |= (1 << new_idx)
                    new_adj[new_idx] |= (1 << i)

            # Local search: repair violations by flipping edges
            local_deadline = time.time() + 1.5
            valid = False
            max_repair_steps = 2000
            for step in range(max_repair_steps):
                if time.time() > local_deadline:
                    break

                viol = find_violation(new_adj, target_n)
                if viol is None:
                    valid = True
                    break

                vtype, verts = viol
                if vtype == 'clique':
                    # Remove an edge from the clique
                    # Choose randomly from the 10 pairs in the K5
                    u, v = random.sample(verts, 2)
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                else:
                    # Add an edge to break the independent set
                    u, v = random.sample(verts, 2)
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)

            if valid:
                current_n = target_n
                adj = new_adj
                best_n = current_n
                best_adj = adj[:]
                success = True
                break

        if not success:
            # If we can't grow to this N, try a few more times with different seeds
            # but don't spend too much time
            extra_attempts = 50
            for attempt in range(extra_attempts):
                if time.time() > deadline - 10:
                    break

                new_adj = [0] * target_n
                for i in range(current_n):
                    new_adj[i] = adj[i]

                new_idx = target_n - 1
                for i in range(current_n):
                    if random.random() < 0.5:
                        new_adj[i] |= (1 << new_idx)
                        new_adj[new_idx] |= (1 << i)

                local_deadline = time.time() + 0.8
                valid = False
                max_repair_steps = 1000
                for step in range(max_repair_steps):
                    if time.time() > local_deadline:
                        break

                    viol = find_violation(new_adj, target_n)
                    if viol is None:
                        valid = True
                        break

                    vtype, verts = viol
                    if vtype == 'clique':
                        u, v = random.sample(verts, 2)
                        new_adj[u] &= ~(1 << v)
                        new_adj[v] &= ~(1 << u)
                    else:
                        u, v = random.sample(verts, 2)
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)

                if valid:
                    current_n = target_n
                    adj = new_adj
                    best_n = current_n
                    best_adj = adj[:]
                    success = True
                    break

            if not success:
                # Try one more time with a slightly different probability
                for p in [0.4, 0.6]:
                    if time.time() > deadline - 10:
                        break
                    for attempt in range(30):
                        if time.time() > deadline - 10:
                            break
                        new_adj = [0] * target_n
                        for i in range(current_n):
                            new_adj[i] = adj[i]

                        new_idx = target_n - 1
                        for i in range(current_n):
                            if random.random() < p:
                                new_adj[i] |= (1 << new_idx)
                                new_adj[new_idx] |= (1 << i)

                        local_deadline = time.time() + 0.5
                        valid = False
                        max_repair_steps = 800
                        for step in range(max_repair_steps):
                            if time.time() > local_deadline:
                                break
                            viol = find_violation(new_adj, target_n)
                            if viol is None:
                                valid = True
                                break
                            vtype, verts = viol
                            if vtype == 'clique':
                                u, v = random.sample(verts, 2)
                                new_adj[u] &= ~(1 << v)
                                new_adj[v] &= ~(1 << u)
                            else:
                                u, v = random.sample(verts, 2)
                                new_adj[u] |= (1 << v)
                                new_adj[v] |= (1 << u)

                        if valid:
                            current_n = target_n
                            adj = new_adj
                            best_n = current_n
                            best_adj = adj[:]
                            success = True
                            break
                    if success:
                        break

            if not success:
                # Give up on this target size, but continue trying larger?
                # Usually if we can't get N, we can't get N+1 easily.
                # But let's try one more time before giving up entirely.
                pass

    # Final verification of best_adj
    if find_violation(best_adj, best_n) is not None:
        # Should not happen, but fallback to Paley
        best_n = 37
        best_adj = [0] * 37
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    best_adj[i] |= (1 << j)
                    best_adj[j] |= (1 << i)

    return best_n, get_edges(best_adj, best_n)

# EVOLVE-BLOCK-END
