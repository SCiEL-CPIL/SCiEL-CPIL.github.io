import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    timeout = 140.0

    # Moore bound for k=5, g=9
    # n_min = 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i = 1 + 5 * (1 + 4 + 16) = 1 + 5*21 = 106
    # But actual minimum is much larger. Known: n(5,9) <= 1116 (Exoo et al. 2026)
    # Lower bound: 428

    # For k=5, g=9, we need a 5-regular graph with no cycle shorter than 9.
    # Circulant graphs: C_n(S) where S is a set of 5 steps symmetric about 0.
    # For n even, S = {s1, n-s1, s2, n-s2, n/2} where s1, s2 are in 1..n/2-1 distinct.

    def check_girth_circulant(n, S, k, g):
        """Check if circulant graph C_n(S) has girth at least g."""
        adj = [[] for _ in range(n)]
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:  # no loops
                    adj[i].append(j)

        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False

        # Check no repeated edges (should be fine for circulant with distinct S)
        # Check girth using BFS from each vertex
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        q.append(y)
                    else:
                        # If y is already visited and not the parent, we have a cycle
                        if parent[x] != y:
                            cycle_len = dist[x] + dist[y] + 1
                            if cycle_len < g:
                                return False
        return True

    # Strategy 1: Try circulant graphs with systematic search
    # For n even, S = {s1, n-s1, s2, n-s2, n/2}
    # We need s1 != s2, s1 != n-s2, etc. (i.e., all 5 steps distinct mod n)
    # Also s1, s2 should not be 0 or n/2

    # First, try smaller n values (closer to Moore bound)
    # n must be even
    min_n = 106  # Moore bound
    max_n = 1200

    # Try circulant with systematic s1, s2
    found = None
    best_n = None

    # Sort candidate n values: try smaller n first
    candidate_ns = []
    for n in range(min_n + 1, max_n + 1, 2):  # n must be even
        if n % 2 == 0:
            candidate_ns.append(n)

    # But 1116 is known to work, so try around there too
    # Add some targeted values
    for n in [1100, 1102, 1104, 1106, 1108, 1110, 1112, 1114, 1116, 1118, 1120, 1122, 1124, 1126, 1128, 1130]:
        if n not in candidate_ns:
            candidate_ns.append(n)
        if n % 2 != 0:
            candidate_ns.append(n + 1)

    candidate_ns = sorted(set(n for n in candidate_ns if n % 2 == 0))

    random.seed(42)

    for n in candidate_ns:
        if time.time() - start_time > timeout:
            break

        # For this n, try random pairs (s1, s2)
        half = n // 2
        valid_steps = list(range(1, half))  # 1..n/2-1

        # Number of trials per n
        max_trials = 200

        for trial in range(max_trials):
            if time.time() - start_time > timeout:
                break

            s1, s2 = random.sample(valid_steps, 2)
            S = [s1, n - s1, s2, n - s2, half]

            # Check distinctness
            if len(set(S)) != 5:
                continue

            if check_girth_circulant(n, S, k, g):
                edges = []
                for i in range(n):
                    for s in S:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Strategy 2: Try non-circulant constructions
    # Use a product or lexicographic product approach
    # For k=5, g=9, we can try to construct using known small graphs

    # Fallback: try a larger search space with different step patterns
    # S = {a, n-a, b, n-b, c, n-c} but we need exactly 5 steps for degree 5
    # So one step must be n/2 (when n is even)

    # Try more systematic search for smaller n
    for n in range(200, 600, 2):
        if time.time() - start_time > timeout:
            break

        half = n // 2
        valid_steps = list(range(1, half))

        # Try all pairs for smaller n (feasible if n < 400)
        if n < 400:
            for i, s1 in enumerate(valid_steps):
                for s2 in valid_steps[i+1:]:
                    S = [s1, n - s1, s2, n - s2, half]
                    if len(set(S)) != 5:
                        continue
                    if check_girth_circulant(n, S, k, g):
                        edges = []
                        for j in range(n):
                            for s in S:
                                jj = (j + s) % n
                                if j < jj:
                                    edges.append((j, jj))
                        return n, edges
        else:
            # Random sampling for larger n
            for _ in range(50):
                if time.time() - start_time > timeout:
                    break
                s1, s2 = random.sample(valid_steps, 2)
                S = [s1, n - s1, s2, n - s2, half]
                if len(set(S)) != 5:
                    continue
                if check_girth_circulant(n, S, k, g):
                    edges = []
                    for j in range(n):
                        for s in S:
                            jj = (j + s) % n
                            if j < jj:
                                edges.append((j, jj))
                    return n, edges

    # Final fallback: return a known valid construction if available
    # Use n=1116 with a known working circulant if found during search
    # Otherwise, try a dense random search at n=1116
    n = 1116
    half = n // 2
    valid_steps = list(range(1, half))
    for _ in range(500):
        if time.time() - start_time > timeout - 5:
            break
        s1, s2 = random.sample(valid_steps, 2)
        S = [s1, n - s1, s2, n - s2, half]
        if len(set(S)) != 5:
            continue
        if check_girth_circulant(n, S, k, g):
            edges = []
            for i in range(n):
                for s in S:
                    j = (i + s) % n
                    if i < j:
                        edges.append((i, j))
            return n, edges

    # Absolute fallback: return an error-like result (should not happen)
    # Return n=0 with no edges to signal failure
    return 0, []
