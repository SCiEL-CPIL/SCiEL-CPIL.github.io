import numpy as np
import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = 110.0  # Safety margin

    # For small n, standard construction is optimal and fast
    if n <= 4:
        # Standard construction
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    P = np.array(perms)  # Shape (N, n)

    # Compute overlap matrix O
    # O[i, j] = max k such that P[i][-k:] == P[j][:k]
    O = np.zeros((N, N), dtype=np.int8)

    for k in range(n - 1, 0, -1):
        if time.time() > deadline:
            break

        suffixes = P[:, n-k:]  # (N, k)
        prefixes = P[:, :k]    # (N, k)

        # Use chunked comparison to avoid huge intermediate arrays
        chunk_size = 512
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            s_chunk = suffixes[i_start:i_end]  # (chunk, k)
            # Compare each row of s_chunk against all prefixes
            # (chunk, 1, k) vs (1, N, k) -> (chunk, N, k)
            matches_chunk = (s_chunk[:, None, :] == prefixes[None, :, :]).all(axis=-1)
            # Update O for this chunk
            mask = matches_chunk & (O[i_start:i_end] == 0)
            O[i_start:i_end][mask] = k

    # Cost matrix C = n - O
    C = n - O
    C = C.astype(np.int32)

    # TSP Heuristic: Nearest Neighbor
    unvisited = np.ones(N, dtype=bool)
    path = [0]
    unvisited[0] = False
    current = 0

    while np.any(unvisited):
        if time.time() > deadline:
            break

        # Find min cost to unvisited nodes
        costs = C[current, :].copy()
        costs[~unvisited] = np.inf

        next_node = np.argmin(costs)

        if costs[next_node] == np.inf:
            break

        path.append(next_node)
        unvisited[next_node] = False
        current = next_node

    # If we didn't finish NN, fallback
    if len(path) < N:
        word_std = _standard_construction(n)
        return "".join(map(str, word_std))

    # 2-opt improvement (Lin-Kernighan style edge swaps)
    # Convert path to numpy array for speed
    path_arr = np.array(path, dtype=np.int32)

    # Compute current total cost
    def path_cost(p):
        # p is a numpy array of indices
        # Sum of C[p[i], p[i+1]] for all consecutive pairs
        return np.sum(C[p[:-1], p[1:]])

    current_cost = path_cost(path_arr)
    improved = True
    max_passes = 5

    for _pass in range(max_passes):
        if time.time() > deadline - 10:
            break
        improved = False

        # Randomized 2-opt: try random pairs of positions
        n_attempts = min(20000, N * N // 10)
        for _ in range(n_attempts):
            if time.time() > deadline - 10:
                break

            # Pick two random edges (i, i+1) and (j, j+1) with i < j
            i = np.random.randint(0, N - 1)
            j = np.random.randint(i + 1, N)

            # Current edges: (path[i], path[i+1]) and (path[j], path[j+1])
            # After reversal of segment [i+1, j]:
            # New edges: (path[i], path[j]) and (path[i+1], path[j+1])
            # (The internal edges of the reversed segment have their costs swapped but sum is same
            #  only if C is symmetric. C is NOT symmetric in general, so we need to be careful.)

            # Actually for asymmetric TSP, reversing a segment changes costs.
            # Let's compute the delta properly.
            # Old segment: path[i] -> path[i+1] -> ... -> path[j] -> path[j+1]
            # New segment: path[i] -> path[j] -> ... -> path[i+1] -> path[j+1]

            # The edges that change:
            # Remove: C[path[i], path[i+1]] and C[path[j], path[j+1]]
            # Add: C[path[i], path[j]] and C[path[i+1], path[j+1]]
            # All internal edges within [i+1, j] are reversed, so their costs change from
            # C[path[k], path[k+1]] to C[path[k+1], path[k]] for k in [i+1, j-1]

            # This is complex. For simplicity, let's just try 2-opt on the symmetric version
            # or use a simpler improvement: 1-opt (swap two adjacent elements).

            # 1-opt: swap path[j] and path[j+1] for some j
            # Actually, let's try a simpler approach: try swapping adjacent elements
            # and see if it reduces cost.

            # Better: try reversing a small random segment
            seg_len = np.random.randint(2, min(10, N // 4))
            start = np.random.randint(0, N - seg_len)
            end = start + seg_len - 1

            # Compute cost delta for reversing path[start:end+1]
            # Edges affected:
            # Before: ... -> path[start-1] -> path[start] -> ... -> path[end] -> path[end+1] -> ...
            # After:  ... -> path[start-1] -> path[end] -> ... -> path[start] -> path[end+1] -> ...

            # Removed edges: C[path[start-1], path[start]] (if start>0) and C[path[end], path[end+1]] (if end<N-1)
            # Added edges: C[path[start-1], path[end]] (if start>0) and C[path[start], path[end+1]] (if end<N-1)
            # Internal edges reversed: for k in [start, end-1], C[path[k], path[k+1]] becomes C[path[k+1], path[k]]

            delta = 0
            if start > 0:
                delta += C[path_arr[start], path_arr[end]] - C[path_arr[start-1], path_arr[start]]
            if end < N - 1:
                delta += C[path_arr[start], path_arr[end+1]] - C[path_arr[end], path_arr[end+1]]
            # Internal
            for k in range(start, end):
                delta += C[path_arr[k+1], path_arr[k]] - C[path_arr[k], path_arr[k+1]]

            if delta < 0:
                # Perform reversal
                path_arr[start:end+1] = path_arr[start:end+1][::-1]
                current_cost += delta
                improved = True

        if not improved:
            break

    # Convert back to list
    path = list(path_arr)

    # Construct the word
    word = list(P[path[0]])
    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        ov = O[prev, curr]
        word.extend(P[curr][ov:])

    # Verify validity
    found = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        found.add(w)

    if len(found) == N:
        return "".join(map(str, word))
    else:
        # Fallback to standard construction
        word_std = _standard_construction(n)
        return "".join(map(str, word_std))

def _standard_construction(n):
    word_std = [1]
    for m in range(2, n + 1):
        perms_std, seen = [], set()
        for i in range(len(word_std) - m + 2):
            window = tuple(word_std[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms_std.append(window)
        new = []
        for p in perms_std:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_std = new
    return word_std
