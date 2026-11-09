"""Triangle-free graph with alpha < 10, maximizing n.
Uses algebraic constructions (Cayley graphs on Z_n with sum-free sets)
combined with local search to find valid graphs on up to 40 vertices.
"""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_triangle_free_bitmask(n, adj):
    """Check triangle-freeness using bitmasks. adj[i] is a bitmask of neighbors of i."""
    for i in range(n):
        # For each pair of neighbors of i, check if they're connected
        a = adj[i]
        while a:
            lsb = a & (-a)
            a ^= lsb
            j = lsb.bit_length() - 1
            # Check if any neighbor of j is also a neighbor of i (excluding i itself)
            common = adj[j] & adj[i]
            # Remove i from common (since i is adjacent to j, i is in adj[j])
            common &= ~(1 << i)
            # If common has any bit set, there's a triangle
            if common:
                return False
    return True


def _has_independent_set_of_size(n, adj, target=10):
    """Branch and bound to check if there's an independent set of size >= target.
    adj[i] is a bitmask of neighbors of i.
    Returns True if such a set exists."""
    # Greedy upper bound: repeatedly pick vertex with fewest neighbors in candidates
    def upper_bound(candidates_mask):
        count = 0
        mask = candidates_mask
        while mask:
            # Find vertex with minimum degree within candidates
            best_v = -1
            best_deg = n + 1
            m = mask
            while m:
                lsb = m & (-m)
                m ^= lsb
                v = lsb.bit_length() - 1
                deg = bin(adj[v] & mask).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            if best_v == -1:
                break
            count += 1
            # Remove best_v and its neighbors from candidates
            mask &= ~(1 << best_v)
            mask &= ~adj[best_v]
        return count

    def bnb(candidates_mask, size):
        if size >= target:
            return True
        if not candidates_mask:
            return False
        if size + upper_bound(candidates_mask) < target:
            return False
        # Pick a vertex to branch on - use the one with fewest neighbors in candidates
        m = candidates_mask
        best_v = -1
        best_deg = n + 1
        while m:
            lsb = m & (-m)
            m ^= lsb
            v = lsb.bit_length() - 1
            deg = bin(adj[v] & candidates_mask).count('1')
            if deg < best_deg:
                best_deg = deg
                best_v = v
        # Include best_v: remove it and its neighbors
        new_mask = candidates_mask & ~(1 << best_v) & ~adj[best_v]
        if bnb(new_mask, size + 1):
            return True
        # Exclude best_v
        new_mask = candidates_mask & ~(1 << best_v)
        if bnb(new_mask, size):
            return True
        return False

    return bnb((1 << n) - 1, 0)


def _edges_from_adj_bitmask(n, adj):
    """Convert adjacency bitmasks to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges


def _build_cayley_graph(n, S):
    """Build Cayley graph on Z_n with connection set S. Returns adj bitmasks."""
    S_set = set(S)
    adj = [0] * n
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j != i:
                adj[i] |= (1 << j)
    return adj


def _try_cayley_constructions():
    """Try various Cayley graph constructions on Z_n for n up to 40."""
    deadline = time.time() + 120
    best_result = None
    best_n = 0

    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
        if n <= best_n:
            continue

        # Try various sum-free sets
        # A set S is sum-free if (S+S) ∩ S = ∅
        # For triangle-free Cayley graph: need S ∩ (S+S) = ∅
        # Also need S = -S for symmetry (undirected graph)

        candidates = []

        # Interval-based sum-free sets
        # S = {a, a+1, ..., b} is sum-free if 2a > b (mod n considerations)
        # For Z_n: S = {a, ..., b} with 2a >= n or 2b < n and 2a > b
        for a in range(1, n):
            for b in range(a, n):
                S = list(range(a, b + 1))
                # Check symmetry: need -S = S mod n, i.e., for each s in S, (n-s) % n in S
                S_set = set(S)
                symmetric = True
                for s in S:
                    if (n - s) % n not in S_set and s != 0:
                        symmetric = False
                        break
                if not symmetric:
                    continue
                # Check sum-free: for all s1, s2 in S, (s1+s2) % n not in S
                sum_free = True
                for s1 in S:
                    for s2 in S:
                        if (s1 + s2) % n in S_set:
                            sum_free = False
                            break
                    if not sum_free:
                        break
                if sum_free and len(S) >= 2:
                    candidates.append(S)

        # Also try non-interval symmetric sum-free sets
        # Try random symmetric sets
        random.seed(n * 1000 + 42)
        for _ in range(200):
            k = random.randint(n // 4, n // 2)
            # Pick k elements from {1, ..., n//2} and add their negatives
            half = list(range(1, n // 2 + 1))
            chosen = random.sample(half, min(k, len(half)))
            S = set()
            for s in chosen:
                S.add(s)
                S.add(n - s)
            S.discard(0)
            if len(S) < 2:
                continue
            # Check sum-free
            sum_free = True
            for s1 in S:
                for s2 in S:
                    if (s1 + s2) % n in S:
                        sum_free = False
                        break
                if not sum_free:
                    break
            if sum_free:
                candidates.append(list(S))

        # Sort by size (larger S = more edges = smaller alpha)
        candidates.sort(key=len, reverse=True)

        for S in candidates[:50]:  # Try top 50
            if time.time() > deadline:
                break
            adj = _build_cayley_graph(n, S)
            if not _is_triangle_free_bitmask(n, adj):
                continue
            if not _has_independent_set_of_size(n, adj, 10):
                edges = _edges_from_adj_bitmask(n, adj)
                result = {'n': n, 'edges': edges}
                if n > best_n:
                    best_n = n
                    best_result = result
                break  # Found valid for this n, move to next n

    return best_result


def _local_search(n, target_edges, deadline):
    """Local search: start with random graph, fix triangles, minimize alpha."""
    random.seed(int(time.time() * 1000) % 100000)

    def random_initial_graph():
        """Generate a random graph with approximately target_edges edges."""
        adj = [0] * n
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < target_edges / (n * (n - 1) / 2):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edge_count += 1
        return adj

    def count_triangles(adj):
        count = 0
        for i in range(n):
            a = adj[i]
            while a:
                lsb = a & (-a)
                a ^= lsb
                j = lsb.bit_length() - 1
                common = adj[j] & adj[i] & ~(1 << i)
                count += bin(common).count('1')
        return count // 3

    def remove_triangle_edge(adj, i, j, k):
        """Remove one edge from triangle (i,j,k) to break it."""
        # Remove edge (i,j)
        adj[i] &= ~(1 << j)
        adj[j] &= ~(1 << i)
        return adj

    def greedy_fix_triangles(adj, max_iter=10000):
        """Iteratively remove edges to eliminate all triangles."""
        for _ in range(max_iter):
            # Find a triangle
            found = False
            for i in range(n):
                a = adj[i]
                while a:
                    lsb = a & (-a)
                    a ^= lsb
                    j = lsb.bit_length() - 1
                    common = adj[j] & adj[i] & ~(1 << i)
                    if common:
                        lsb2 = common & (-common)
                        k = lsb2.bit_length() - 1
                        # Remove edge (j, k) to break triangle
                        adj[j] &= ~(1 << k)
                        adj[k] &= ~(1 << j)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return adj

    best_adj = None
    best_score = float('inf')

    max_attempts = 500
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break

        adj = random_initial_graph()
        adj = greedy_fix_triangles(adj)

        if not _is_triangle_free_bitmask(n, adj):
            continue

        # Check independence
        if _has_independent_set_of_size(n, adj, 10):
            # Try to fix by adding edges
            # Find a large independent set and add edges within it
            # For now, just score it
            score = 1000 + count_triangles(adj)
        else:
            score = 0
            edges = _edges_from_adj_bitmask(n, adj)
            if best_score > 0:
                best_adj = adj
                best_score = 0
                return {'n': n, 'edges': edges}

    return None


def _deterministic_construction():
    """Try specific known constructions."""
    deadline = time.time() + 120

    # Try n=40 with various structured sets
    for n in [40, 39, 38, 37, 36, 35]:
        if time.time() > deadline:
            break

        # Try the "middle" interval construction
        # S = {floor(n/4)+1, ..., floor(3n/4)}
        lo = n // 4 + 1
        hi = (3 * n) // 4
        S = list(range(lo, hi + 1))

        # Make symmetric
        S_set = set(S)
        for s in list(S_set):
            S_set.add(n - s)
        S_set.discard(0)

        # Check sum-free
        sum_free = True
        for s1 in S_set:
            for s2 in S_set:
                if (s1 + s2) % n in S_set:
                    sum_free = False
                    break
            if not sum_free:
                break

        if sum_free:
            adj = _build_cayley_graph(n, list(S_set))
            if _is_triangle_free_bitmask(n, adj):
                if not _has_independent_set_of_size(n, adj, 10):
                    edges = _edges_from_adj_bitmask(n, adj)
                    return {'n': n, 'edges': edges}

        # Try complementary construction: connect if distance is NOT in a certain set
        # i.e., complement of a triangle-free graph might not be triangle-free
        # Skip this

    return None


def construct():
    """Construct a triangle-free graph with no independent set of size 10, maximizing n."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # First try deterministic constructions
    result = _deterministic_construction()
    if result and result['n'] >= 39:
        return result

    # Try Cayley graph constructions
    result = _try_cayley_constructions()
    if result and result['n'] >= 39:
        return result

    # Try local search for various n
    for n in range(40, 30, -1):
        if time.time() > deadline:
            break
        if result and result['n'] >= n:
            continue
        target_edges = n * 7 // 2  # Average degree ~7
        res = _local_search(n, target_edges, deadline)
        if res:
            if not result or res['n'] > result['n']:
                result = res
            if res['n'] >= 39:
                return result

    # Fallback: try a few more random attempts
    if time.time() < deadline:
        for n in [40, 39, 38]:
            if time.time() > deadline:
                break
            for _ in range(100):
                if time.time() > deadline:
                    break
                random.seed(random.getrandbits(64))
                adj = [0] * n
                # Generate random edges
                p = 7.0 / n  # Expected degree ~7
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                # Fix triangles
                adj = greedy_fix_triangles_local(adj, n)
                if not _is_triangle_free_bitmask(n, adj):
                    continue
                if not _has_independent_set_of_size(n, adj, 10):
                    edges = _edges_from_adj_bitmask(n, adj)
                    if not result or n > result['n']:
                        result = {'n': n, 'edges': edges}
                    if n >= 39:
                        return result

    # Final fallback: small known valid graph
    # Clebsch graph on 16 vertices: triangle-free, alpha = 5
    # Or just return a simple valid graph
    if result is None:
        # Return a simple valid graph: complete bipartite K_{9,9} is triangle-free
        # but alpha = 9 (take one partition). That's exactly 9 < 10. n=18.
        # Actually K_{9,9}: independent sets are subsets of one partition, max size 9.
        # Triangle-free: yes, bipartite.
        n = 18
        edges = []
        for i in range(9):
            for j in range(9, 18):
                edges.append([i, j])
        result = {'n': n, 'edges': edges}

    return result


def greedy_fix_triangles_local(adj, n, max_iter=5000):
    """Local version of triangle fixing."""
    for _ in range(max_iter):
        found = False
        for i in range(n):
            a = adj[i]
            while a:
                lsb = a & (-a)
                a ^= lsb
                j = lsb.bit_length() - 1
                common = adj[j] & adj[i] & ~(1 << i)
                if common:
                    lsb2 = common & (-common)
                    k = lsb2.bit_length() - 1
                    adj[j] &= ~(1 << k)
                    adj[k] &= ~(1 << j)
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return adj

# EVOLVE-BLOCK-END
