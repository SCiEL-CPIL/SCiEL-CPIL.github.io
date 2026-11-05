import random
import time
import sys
sys.setrecursionlimit(10000)

def construct():
    n = 24
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for safety

    # Generate the pinned edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    r = random.Random(614)
    r.shuffle(edges)

    # Map edge to index
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx

    # Precompute adjacency lists sorted by edge arrival index
    adj_raw = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj_raw[u].append((v, idx))
        adj_raw[v].append((u, idx))

    for i in range(n):
        adj_raw[i].sort(key=lambda x: x[1])

    full_mask = (1 << n) - 1

    # Helper: Check if G_K has a Hamilton cycle and return it.
    def find_hamiltonian_cycle(K):
        # Build adjacency bitmasks
        adj_mask = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        # Quick check: min degree
        for i in range(n):
            if adj_mask[i].bit_count() < 2:
                return None

        # Connectivity check
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            u = stack.pop()
            m = adj_mask[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    stack.append(v)
        if count < n:
            return None

        # Precompute for each vertex its neighbors in order of fewest remaining options
        # We'll use Warnsdorff's rule with dynamic degree calculation

        path = [0]
        visited_mask = 1  # vertex 0 is visited

        # Iterative DFS with explicit stack to avoid recursion depth issues
        # and to allow better pruning
        stack = [(0, 0, 0)]  # (vertex, next_candidate_idx, visited_mask_at_entry)
        # Actually, let's use a cleaner recursive approach with better pruning

        def dfs(u, vmask, depth):
            nonlocal deadline
            if time.time() > deadline:
                return False

            if depth == n:
                return (adj_mask[u] & 1) != 0

            candidates = adj_mask[u] & (~vmask)
            if candidates == 0:
                return False

            # Check if any unvisited vertex has degree 0 in remaining graph
            # (it would be stranded)
            # This is expensive, so only do it periodically or use a heuristic

            # Warnsdorff's rule: try neighbors with fewest options first
            cand_list = []
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                # Count unvisited neighbors of v
                deg = (adj_mask[v] & ~vmask).bit_count()
                # If v would become isolated (degree 0 after we visit it, unless it's the last)
                if depth + 1 < n and deg == 0:
                    return False  # v would be stranded
                cand_list.append((deg, v))

            cand_list.sort()

            for deg, v in cand_list:
                path.append(v)
                if dfs(v, vmask | (1 << v), depth + 1):
                    return True
                path.pop()

            return False

        if dfs(0, 1, 1):
            return path
        return None

    # Binary search for minimum K
    low = n
    high = len(edges)
    k_min = len(edges)

    # Find fallback cycle for full graph
    fallback_cycle = find_hamiltonian_cycle(len(edges))
    if fallback_cycle is None:
        return list(range(n))

    best_cycle = fallback_cycle
    best_K = len(edges)

    # Binary search
    while low <= high:
        if time.time() > deadline:
            break
        mid = (low + high) // 2
        cycle = find_hamiltonian_cycle(mid)
        if cycle:
            k_min = mid
            best_cycle = cycle
            best_K = mid
            high = mid - 1
        else:
            low = mid + 1

    if best_cycle:
        return best_cycle
    return fallback_cycle

if __name__ == "__main__":
    pass
