import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 20

    def count_bits(x):
        return bin(x).count('1')

    def can_add_edge_fast(adj, u, v):
        """Check if adding edge (u,v) maintains K3,3-freedom."""
        old_u = adj[u]
        old_v = adj[v]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        valid = True
        # Check all triples containing u or v
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if i == u or j == u or k == u or i == v or j == v or k == v:
                        if count_bits(adj[i] & adj[j] & adj[k]) >= 3:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break

        if not valid:
            adj[u] = old_u
            adj[v] = old_v

        return valid

    def is_valid_adj(adj):
        """Full validity check."""
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if count_bits(adj[i] & adj[j] & adj[k]) >= 3:
                        return False
        return True

    def purge(adj):
        """Remove edges until graph is valid."""
        while not is_valid_adj(adj):
            found = False
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        common = adj[i] & adj[j] & adj[k]
                        if count_bits(common) >= 3:
                            neighbors = []
                            c = common
                            while len(neighbors) < 3 and c:
                                lsb = c & -c
                                neighbors.append(lsb.bit_length() - 1)
                                c -= lsb
                            u = random.choice([i, j, k])
                            v = random.choice(neighbors)
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break

    def greedy_fill(adj):
        """Try to add all possible missing edges."""
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        if can_add_edge_fast(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True

    def local_search(adj):
        """Local search with swaps and perturbations."""
        while time.time() < deadline - 5:
            added = False
            missing = [(u, v) for u in range(n) for v in range(u + 1, n)
                      if not (adj[u] >> v) & 1]
            if not missing:
                break

            random.shuffle(missing)
            for u, v in missing:
                if can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break

            if added:
                continue

            existing = [(u, v) for u in range(n) for v in range(u + 1, n)
                       if (adj[u] >> v) & 1]
            if not existing:
                break

            random.shuffle(existing)
            swap_found = False
            for u1, v1 in existing[:20]:
                if time.time() > deadline - 5:
                    break
                old_u1 = adj[u1]
                old_v1 = adj[v1]
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)

                for u2, v2 in missing[:50]:
                    if can_add_edge_fast(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        swap_found = True
                        break
                    if time.time() > deadline - 5:
                        break

                if not swap_found:
                    adj[u1] = old_u1
                    adj[v1] = old_v1
                else:
                    break

            if not swap_found:
                if not existing:
                    break
                u, v = random.choice(existing)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)

    best_edges = 0
    best_adj = None

    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1

        if restarts == 1:
            adj = [0] * n
        elif restarts <= 5:
            p = 0.4
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            p = 0.2 + 0.4 * random.random()
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

        purge(adj)
        greedy_fill(adj)
        local_search(adj)

        edge_count = sum(count_bits(x) for x in adj) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj[:]

    edges = []
    if best_adj:
        for u in range(n):
            for v in range(u + 1, n):
                if (best_adj[u] >> v) & 1:
                    edges.append([u, v])
    else:
        edges = []

    return edges
