import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    def is_regular(adj, mask, n):
        """Check if induced subgraph on mask is regular. Returns True if regular."""
        verts = []
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            m &= m - 1
            verts.append(v)
        if len(verts) < 6:
            return False
        # Compute degrees within the induced subgraph
        first_deg = bin(adj[verts[0]] & mask).count('1')
        for v in verts[1:]:
            if bin(adj[v] & mask).count('1') != first_deg:
                return False
        return True

    def full_verify(adj, n):
        """Verify no subset of size >= 6 induces a regular subgraph."""
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_regular(adj, mask, n):
                    return False
        return True

    def count_violations(adj, n, max_k=10):
        """Count regular induced subgraphs of size 6 to max_k."""
        count = 0
        for k in range(6, min(max_k + 1, n + 1)):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_regular(adj, mask, n):
                    count += 1
        return count

    # Try n=20 first with multiple strategies
    for n in [20, 19, 18, 15, 10]:
        if time.time() > deadline - 10:
            break

        # Strategy 1: Deterministic circulant-like construction
        # Use a known good seed pattern
        import random
        rng = random.Random(n * 1000 + 42)

        # Try several random graphs with local improvement
        for attempt in range(30):
            if time.time() > deadline - 5:
                break

            # Initialize with random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # Local search: repeatedly find violations and fix them
            improved = True
            steps = 0
            max_steps = 2000
            while improved and steps < max_steps:
                improved = False
                steps += 1
                if time.time() > deadline - 5:
                    break

                # Find a violation
                found = False
                for k in [6, 7, 8]:
                    if k > n:
                        continue
                    for combo in itertools.combinations(range(n), k):
                        mask = 0
                        for v in combo:
                            mask |= (1 << v)
                        if is_regular(adj, mask, n):
                            # Found violation; flip a random edge within
                            u, v = rng.sample(combo, 2)
                            if adj[u] & (1 << v):
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                            else:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break

            # Check if we have a valid graph
            if steps < max_steps and not improved:
                # No violations of size 6,7,8 found; do full check
                if full_verify(adj, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}

        # Strategy 2: Greedy edge addition starting from empty graph
        if time.time() > deadline - 10:
            break

        for attempt in range(5):
            if time.time() > deadline - 5:
                break

            adj = [0] * n
            # Add edges greedily, checking for violations
            edge_order = []
            for i in range(n):
                for j in range(i + 1, n):
                    edge_order.append((i, j))
            rng.shuffle(edge_order)

            for (u, v) in edge_order:
                if time.time() > deadline - 5:
                    break
                # Try adding edge (u,v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                # Quick check: does this create a regular induced subgraph of size 6?
                # Only check subsets containing both u and v
                violation = False
                for k in [6]:
                    if k > n:
                        continue
                    # Check all 6-sets containing u and v
                    for combo in itertools.combinations(range(n), k):
                        if u in combo and v in combo:
                            mask = 0
                            for x in combo:
                                mask |= (1 << x)
                            if is_regular(adj, mask, n):
                                violation = True
                                break
                    if violation:
                        break
                if violation:
                    # Remove edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

            # Full verify
            if full_verify(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}

    # Fallback: small verified graph
    n = 10
    adj = [0] * n
    import random
    rng = random.Random(42)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    if full_verify(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    return {'n': 5, 'edges': []}
