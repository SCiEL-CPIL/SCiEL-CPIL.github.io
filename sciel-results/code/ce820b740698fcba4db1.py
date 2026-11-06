"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        """Check if graph contains a K4."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        return True
        return False

    def max_triangle_free_size(adj, time_limit=3.0):
        """Find the largest induced triangle-free subgraph using branch and bound."""
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        t0 = time.time()
        timeout = [False]

        # Precompute for each vertex, the set of its neighbors that form triangles with other neighbors
        # For faster triangle check when adding vertex v to set S:
        # v can be added if no two neighbors of v in S are adjacent to each other
        
        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            # Pruning
            if current_size + (n - idx) <= best[0]:
                return

            v = order[idx]

            # Option 1: Don't include v
            backtrack(idx + 1, current_mask, current_size)

            # Option 2: Include v
            if not timeout[0]:
                neighbors_in_S = current_mask & adj[v]
                has_edge = False
                tmp = neighbors_in_S
                while tmp:
                    bit = tmp & (-tmp)
                    u = bit.bit_length() - 1
                    rest = tmp & ~bit
                    if (adj[u] & rest) != 0:
                        has_edge = True
                        break
                    tmp = rest
                if not has_edge:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)

        backtrack(0, 0, 0)
        return best[0]

    def remove_k4(adj):
        """Remove edges to make graph K4-free."""
        for _ in range(200):
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if not ((common >> c) & 1):
                            continue
                        remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                        if remaining:
                            d_bit = remaining & (-remaining)
                            d = d_bit.bit_length() - 1
                            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                            x, y = pair
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return not has_k4(adj)

    def make_k4_free_by_partition(parts, cross_density, within_density):
        """Create a K4-free graph from a partition with controlled densities."""
        adj = [0] * n
        num_parts = len(parts)
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                for a in parts[i]:
                    for b in parts[j]:
                        if random.random() < cross_density:
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
        for i in range(num_parts):
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < within_density:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    random.seed(42)
    candidates = []

    # Strategy 1: Complete tripartite K_{6,6,6} - triangle-free induced subgraph is max 2 parts = 12
    adj1 = [0] * n
    parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts3[i]:
                for b in parts3[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)

    # Strategy 2: Turan graph T(18, 3) with some internal edges to break large triangle-free sets
    # Add edges within each part to reduce triangle-free induced subgraphs
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        adj = [0] * n
        # Full cross-part edges
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        # Add random edges within parts (density 0.3-0.7)
        for i in range(3):
            wd = 0.3 + random.random() * 0.4
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < wd:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 3: Random 4-partition with cross edges
    for trial in range(40):
        if time.time() > deadline - 30:
            break
        vertices = list(range(n))
        random.shuffle(vertices)
        # Partition into 4 parts of sizes ~4-5
        sizes = [5, 5, 4, 4]
        parts = []
        idx = 0
        for size in sizes:
            parts.append(vertices[idx:idx + size])
            idx += size
        adj = make_k4_free_by_partition(parts, 0.6 + random.random() * 0.4, random.random() * 0.3)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 4: 5-partition
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        vertices = list(range(n))
        random.shuffle(vertices)
        sizes = [4, 4, 4, 3, 3]
        parts = []
        idx = 0
        for size in sizes:
            parts.append(vertices[idx:idx + size])
            idx += size
        adj = make_k4_free_by_partition(parts, 0.5 + random.random() * 0.5, random.random() * 0.2)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 5: Complement of sparse graph (dense K4-free)
    for trial in range(40):
        if time.time() > deadline - 30:
            break
        density = 0.3 + random.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Evaluate candidates
    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 15:
            break
        score = max_triangle_free_size(adj, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=2.0)

    # Local search to improve
    adj = best_adj[:]
    current_score = best_score
    best_adj = adj[:]
    best_score = current_score

    temperature = 1.0
    cooling = 0.998

    for iteration in range(8000):
        if time.time() > deadline - 3:
            break
        if best_score <= 6:
            break

        # Pick a random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i

        old_adj = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

        if has_k4(adj):
            adj = old_adj
            continue

        score = max_triangle_free_size(adj, time_limit=1.2)
        delta = score - current_score

        if delta <= 0 or random.random() < min(1.0, math.exp(-delta / max(temperature, 0.01))):
            current_score = score
            if score < best_score:
                best_score = score
                best_adj = adj[:]
        else:
            adj = old_adj

        temperature *= cooling

    return edges_from_adj(best_adj)
# EVOLVE-BLOCK-END
