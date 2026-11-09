"""Program for R_5(C4) maximization: greedy construction minimizing forbidden-pair creation."""
import time
import random
import numpy as np

def construct():
    """
    Returns {'n': n, 'colors': [...]} where colors is a list of length n*(n-1)//2
    giving one colour in {0,...,4} for each pair (i,j), i<j, in lexicographic order.
    No colour class may contain a C4 (no two vertices with >=2 common neighbours in one colour).
    """
    t0 = time.time()
    deadline = t0 + 115.0

    def edge_idx(i, j, n):
        return i * n - i * (i + 1) // 2 - 1 + (j - i - 1)

    def idx_to_edge(idx, n):
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    def greedy_forbidden_pairs(n, seed, time_limit):
        """
        Greedy: assign edges one by one. For each edge (u,v), for each color c,
        compute how many new 'forbidden pairs' would be created. A forbidden pair
        (a,b) in color c is one where common_c[a][b] == 1. Creating a C4 means
        pushing a forbidden pair to common >= 2. We minimize the number of new
        forbidden pairs created, with secondary objective of minimizing total
        common-neighbor increases.
        """
        random.seed(seed)
        m = n * (n - 1) // 2
        colors = [-1] * m

        # neighbors[c][u] = set of neighbors of u in color c
        neighbors = [[set() for _ in range(n)] for _ in range(5)]
        # common[c] is a flat array: common[c][u*n+v] = |N_c(u) ∩ N_c(v)|
        common = [np.zeros(n * n, dtype=np.int32) for _ in range(5)]
        # forbidden[c] is a flat array: 1 if common[c][u*n+v] == 1
        forbidden = [np.zeros(n * n, dtype=np.int32) for _ in range(5)]
        # total_forbidden = sum of forbidden counts
        total_forbidden = 0

        edges = list(range(m))
        random.shuffle(edges)
        t_start = time.time()

        for eidx in edges:
            if time.time() > t_start + time_limit:
                break
            i, j = idx_to_edge(eidx, n)

            best_c = -1
            best_key = (float('inf'), float('inf'))

            for c in range(5):
                # If we add edge (i,j) in color c:
                # For each w in N_c(i) (w != j): common[c][j][w] += 1
                #   If common[c][j][w] was 1, it becomes 2 -> creates a C4 (bad!)
                #   If common[c][j][w] was 0, it becomes 1 -> creates a new forbidden pair
                # For each w in N_c(j) (w != i): common[c][i][w] += 1
                #   Similarly
                # Also: the edge (i,j) itself doesn't change common[c][i][j]
                #   (as shown in the analysis, adding edge (i,j) doesn't add a common neighbor to i and j)

                c4_created = 0
                new_forbidden = 0

                for w in neighbors[c][i]:
                    if w == j:
                        continue
                    idx_jw = j * n + w
                    if idx_jw > j * n + j:  # ensure j < w for canonical form? No, just use flat
                        pass
                    # common[c][j][w]
                    val = common[c][j * n + w]
                    if val == 1:
                        c4_created += 1
                    elif val == 0:
                        new_forbidden += 1
                    # if val >= 2, already has C4, one more doesn't change forbidden count

                for w in neighbors[c][j]:
                    if w == i:
                        continue
                    val = common[c][i * n + w]
                    if val == 1:
                        c4_created += 1
                    elif val == 0:
                        new_forbidden += 1

                # Key: first minimize C4s created, then minimize new forbidden pairs
                key = (c4_created, new_forbidden)
                if key < best_key:
                    best_key = key
                    best_c = c

            # Assign best_c
            colors[eidx] = best_c
            c = best_c
            # Update data structures
            for w in neighbors[c][i]:
                if w == j:
                    continue
                idx_jw = j * n + w
                old_val = common[c][idx_jw]
                common[c][idx_jw] = old_val + 1
                if old_val == 0:
                    forbidden[c][idx_jw] = 1
                    total_forbidden += 1
                elif old_val == 1:
                    forbidden[c][idx_jw] = 0
                    total_forbidden -= 1
                    # C4 created - we don't track total C4s separately, the key already accounts

            for w in neighbors[c][j]:
                if w == i:
                    continue
                idx_iw = i * n + w
                old_val = common[c][idx_iw]
                common[c][idx_iw] = old_val + 1
                if old_val == 0:
                    forbidden[c][idx_iw] = 1
                    total_forbidden += 1
                elif old_val == 1:
                    forbidden[c][idx_iw] = 0
                    total_forbidden -= 1

            neighbors[c][i].add(j)
            neighbors[c][j].add(i)

        # Verify: check if any C4 exists
        has_c4 = False
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if common[c][u * n + v] >= 2:
                        has_c4 = True
                        break
                if has_c4:
                    break
            if has_c4:
                break

        return colors, not has_c4

    # Try increasing n from 26 down, with multiple seeds
    for target_n in [26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 3:
            break

        # Multiple seeds for diversity
        seeds = [42, 137, 256, 777, 1234, 5555, 9999, 31415, 27182, 16180]
        found = False
        for seed in seeds:
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 2:
                break
            tl = min(remaining * 0.4, 8.0)
            colors, valid = greedy_forbidden_pairs(target_n, seed, tl)
            if valid:
                return {'n': target_n, 'colors': colors}
            found = True

    # Fallback: small cyclic constructions
    for fallback_n in [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline:
            break
        m = fallback_n * (fallback_n - 1) // 2
        colors = [0] * m
        for idx in range(m):
            i, j = idx_to_edge(idx, fallback_n)
            colors[idx] = (j - i) % 5
        # Quick verification
        common_check = [np.zeros(fallback_n * fallback_n, dtype=np.int32) for _ in range(5)]
        for idx in range(m):
            i, j = idx_to_edge(idx, fallback_n)
            c = colors[idx]
            for w in range(fallback_n):
                if w == i or w == j:
                    continue
            # Just check via direct count
        # Direct verification
        valid = True
        for c in range(5):
            adj = [set() for _ in range(fallback_n)]
            for idx in range(m):
                i, j = idx_to_edge(idx, fallback_n)
                if colors[idx] == c:
                    adj[i].add(j)
                    adj[j].add(i)
            for u in range(fallback_n):
                for v in range(u + 1, fallback_n):
                    common_cnt = len(adj[u] & adj[v])
                    if common_cnt >= 2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return {'n': fallback_n, 'colors': colors}

    return {'n': 1, 'colors': []}
