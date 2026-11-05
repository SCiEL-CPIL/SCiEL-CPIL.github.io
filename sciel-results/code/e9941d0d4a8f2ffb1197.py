"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def build_adj_list(adj, n):
        adj_list = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    adj_list[i].add(j)
                    adj_list[j].add(i)
        return adj_list

    def has_c5(adj_list, n):
        """Check for a 5-cycle in the graph. O(n^5) but with early exits."""
        for a in range(n):
            for b in adj_list[a]:
                if b <= a:
                    continue
                for c in adj_list[b]:
                    if c <= b:
                        continue
                    for d in adj_list[c]:
                        if d <= c:
                            continue
                        if d in adj_list[a]:
                            # a-b-c-d-a is a 4-cycle, not 5
                            pass
                        for e in adj_list[d]:
                            if e <= d:
                                continue
                            if e in adj_list[a] and e not in adj_list[b] and e not in adj_list[c]:
                                # a-b-c-d-e-a is a 5-cycle
                                return True
        return False

    def has_indep_set(adj_list, n, k):
        """Check if there is an independent set of size k using backtracking."""
        def backtrack(remaining, current_size):
            if current_size == k:
                return True
            if len(remaining) < k - current_size:
                return False
            # Sort by degree (greedy heuristic: lower degree vertices are more likely in indep set)
            remaining_sorted = sorted(remaining, key=lambda v: len(adj_list[v]))
            for i, v in enumerate(remaining_sorted):
                new_remaining = [u for u in remaining_sorted[i + 1:] if u not in adj_list[v]]
                if backtrack(new_remaining, current_size + 1):
                    return True
            return False
        return backtrack(list(range(n)), 0)

    def validate(adj, n):
        """Full validation: no C5 and no independent set of size 9."""
        adj_list = build_adj_list(adj, n)
        if has_c5(adj_list, n):
            return False
        if has_indep_set(adj_list, n, 9):
            return False
        return True

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    # First, ensure the fallback (8 K4s, n=32) is ready
    fallback_adj = np.zeros((32, 32), dtype=bool)
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                fallback_adj[base + i, base + j] = True
                fallback_adj[base + j, base + i] = True
    fallback_edges = edges_from_adj(fallback_adj, 32)

    # Try to find a valid graph on n=33, 34, or 35
    for n_target in [35, 34, 33]:
        if time.time() > deadline:
            break

        # Strategy: start with 8 K4s (32 vertices) and add 1-3 more vertices
        # with random edges, then try to add sparse edges between K4s
        # to break independent sets while avoiding C5s

        for attempt in range(500):
            if time.time() > deadline:
                break

            random.seed(attempt * 1000 + n_target)  # reproducible per attempt

            adj = np.zeros((n_target, n_target), dtype=bool)

            # Place 8 K4s in vertices 0..31
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        adj[base + i, base + j] = True
                        adj[base + j, base + i] = True

            # Add extra vertices 32..n_target-1 with random edges to K4 vertices
            for v in range(32, n_target):
                for b in range(8):
                    base = 4 * b
                    # Connect to 1-3 vertices in each K4 block
                    num_conn = random.randint(1, 3)
                    targets = random.sample(range(4), num_conn)
                    for t in targets:
                        adj[v, base + t] = True
                        adj[base + t, v] = True

            # Add sparse edges between K4s to reduce independence number
            num_cross_edges = random.randint(5, 20)
            for _ in range(num_cross_edges):
                i = random.randint(0, 31)
                j = random.randint(0, 31)
                if i != j and i // 4 != j // 4:
                    adj[i, j] = True
                    adj[j, i] = True

            # Validate
            if validate(adj, n_target):
                return {'n': n_target, 'edges': edges_from_adj(adj, n_target)}

    # Fallback: 8 disjoint K4s (n=32)
    return {'n': 32, 'edges': fallback_edges}

# EVOLVE-BLOCK-END
