import itertools
import time
import random

def construct():
    start = time.time()
    deadline = start + 135  # 15s safety margin before 150s kill

    def edges_from(adj, n):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if (adj[i] >> j) & 1]

    def is_regular(adj, S):
        """Check if induced subgraph on vertex tuple S is regular."""
        mask = 0
        for v in S:
            mask |= (1 << v)
        first_deg = -1
        for v in S:
            d = (adj[v] & mask).bit_count()
            if first_deg == -1:
                first_deg = d
            elif d != first_deg:
                return False
        return True

    def count_6_violations(adj, n, subsets6):
        """Count regular induced subgraphs of size exactly 6."""
        count = 0
        for S in subsets6:
            if is_regular(adj, S):
                count += 1
        return count

    def count_all_violations(adj, n):
        """Count all regular induced subgraphs of size >= 6."""
        count = 0
        for k in range(6, n + 1):
            for S in itertools.combinations(range(n), k):
                if is_regular(adj, S):
                    count += 1
        return count

    def search(n, time_budget):
        """Try to find a valid graph on n vertices using random restarts + hill climbing."""
        if time.time() > deadline:
            return None

        # Precompute all 6-subsets
        subsets6 = list(itertools.combinations(range(n), 6))

        for restart in range(100):
            if time.time() > deadline - 3:
                break

            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # Count 6-violations
            v6 = count_6_violations(adj, n, subsets6)

            # Quick check: if no 6-violations, verify full validity
            if v6 == 0:
                if count_all_violations(adj, n) == 0:
                    return {'n': n, 'edges': edges_from(adj, n)}

            # Hill climbing to reduce 6-violations
            hc_start = time.time()
            while v6 > 0 and time.time() - hc_start < time_budget * 0.4 and time.time() < deadline - 3:
                # Find a violating 6-set
                bad = None
                for S in subsets6:
                    if is_regular(adj, S):
                        bad = S
                        break
                if bad is None:
                    break

                # Try flipping each edge within the violating set
                best_flip = None
                best_v6 = v6
                for i in range(6):
                    for j in range(i + 1, 6):
                        u, v = bad[i], bad[j]
                        has_edge = (adj[u] >> v) & 1
                        # Flip
                        if has_edge:
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

                        new_v6 = count_6_violations(adj, n, subsets6)

                        # Revert
                        if has_edge:
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                        else:
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)

                        if new_v6 < best_v6:
                            best_v6 = new_v6
                            best_flip = (u, v, has_edge)

                if best_flip:
                    u, v, has_edge = best_flip
                    if has_edge:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    v6 = best_v6
                else:
                    # Stuck, break
                    break

            # After hill climbing, check if valid
            if v6 == 0:
                if count_all_violations(adj, n) == 0:
                    return {'n': n, 'edges': edges_from(adj, n)}

        return None

    # Try n from 22 down to 6
    for n in [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        remaining = deadline - time.time()
        if remaining < 8:
            break
        budget = min(remaining - 3, 50)
        result = search(n, budget)
        if result:
            return result

    # Ultimate fallback
    return {'n': 5, 'edges': []}
