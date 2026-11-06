"""Stronger heuristic: start from a known (5,5)-free graph (Paley(37) or Clebsch complement)
and grow it toward 42 vertices using targeted vertex addition with degree-matching
and local edge-flip repair, validated each step with a fast bitmask clique/ind-set check."""
import random
import time


def _has_k5_or_ind5(adj_matrix, n):
    """Return True if graph has a K5 or an independent 5-set (i.e., NOT (5,5)-free)."""
    # Build bitmasks
    adj_bits = [0] * n
    for i in range(n):
        bits = 0
        row = adj_matrix[i]
        for j in range(n):
            if i != j and row[j]:
                bits |= (1 << j)
        adj_bits[i] = bits

    def _has_clique(bits_list, n):
        """Check for clique of size 5 using backtracking with bitmasks."""
        # Order vertices by degree (fewer candidates first for pruning)
        order = sorted(range(n), key=lambda v: bin(bits_list[v]).count('1'))
        
        # Pre-compute for speed: candidate lists
        def backtrack(candidates_mask, depth):
            if depth == 5:
                return True
            # Count bits in candidates
            if candidates_mask == 0:
                return False
            # Need (5 - depth) more vertices
            # Quick prune: if popcount < needed, fail
            remaining = 5 - depth
            if bin(candidates_mask).count('1') < remaining:
                return False
            # Try each candidate
            cm = candidates_mask
            while cm:
                lb = cm & (-cm)
                v = lb.bit_length() - 1
                cm ^= lb
                # New candidates: intersection of candidates_mask (without v and above) with adj_bits[v]
                # We need candidates that are connected to v and come after v in some order
                # Use a simple approach: new candidates = candidates_mask & adj_bits[v] & ~(mask of vertices <= v)
                # Actually for clique detection, order doesn't matter much, but we need to avoid duplicates
                # Standard: new_candidates = candidates_mask & adj_bits[v], then remove v
                # To avoid re-exploring, we remove v and all previously tried
                new_cand = candidates_mask & adj_bits[v]
                new_cand &= ~(lb - 1)  # only vertices with higher index to maintain ordering
                if bin(new_cand).count('1') >= remaining - 1:
                    if backtrack(new_cand, depth + 1):
                        return True
            return False
        
        return backtrack((1 << n) - 1, 0)

    if _has_clique(adj_bits, n):
        return True

    # Complement for independent set
    comp_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in range(n):
            if i != j and not adj_matrix[i][j]:
                bits |= (1 << j)
        comp_bits[i] = bits

    return _has_clique(comp_bits, n)


def _paley_graph(q):
    """Return (n, adj) for Paley graph on q vertices (q prime, q ≡ 1 mod 4)."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[False] * q for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            d = (j - i) % q
            if d in residues:
                adj[i][j] = True
                adj[j][i] = True
    return q, adj


def _clebsch_complement():
    """Complement of Clebsch graph: 16 vertices, adjacent if Hamming distance is 1, 2, or 4."""
    n = 16
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = bin(i ^ j).count('1')
            if dist in (1, 2, 4):
                adj[i][j] = True
                adj[j][i] = True
    return n, adj


def _get_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    return edges


def _try_add_vertex(adj, n, time_limit, start_time):
    """Try to add one vertex to adj. Returns (success, new_adj, new_n)."""
    new_n = n + 1
    # Try adding vertex with various connection patterns
    # Strategy: pick a random base vertex v, copy its adjacency, then flip some edges
    for _ in range(20):  # 20 random patterns per attempt
        if time.time() > time_limit:
            return False, adj, n

        v = random.randint(0, n - 1)
        new_adj = [row[:] for row in adj]
        new_adj.append([False] * new_n)
        for i in range(n):
            new_adj[i].append(False)

        # Base: same connections as v
        for i in range(n):
            new_adj[v][i] = adj[v][i]
            new_adj[i][v] = adj[v][i]

        # Flip random subset of edges (keep it sparse)
        flip_count = random.randint(0, max(1, n // 5))
        flipped = set()
        for _ in range(flip_count):
            i = random.randint(0, n - 1)
            if i not in flipped:
                flipped.add(i)
                new_adj[v][i] = not new_adj[v][i]
                new_adj[i][v] = not new_adj[v][i]

        # Validate
        if not _has_k5_or_ind5(new_adj, new_n):
            return True, new_adj, new_n

    return False, adj, n


def _try_flip_edge(adj, n):
    """Try flipping one random edge. Returns (success, new_adj, new_n)."""
    a = random.randint(0, n - 1)
    b = random.randint(0, n - 1)
    if a == b:
        return False, adj, n
    if a > b:
        a, b = b, a

    new_adj = [row[:] for row in adj]
    new_adj[a][b] = not new_adj[a][b]
    new_adj[b][a] = not new_adj[b][a]

    if not _has_k5_or_ind5(new_adj, n):
        return True, new_adj, n
    return False, adj, n


def construct():
    """Construct a graph with no K5 and no independent 5-set, maximizing n."""
    start_time = time.time()
    time_limit = 280

    random.seed(12345)

    # Start with Paley(37) — known (5,5)-free
    n, adj = _paley_graph(37)

    # Verify
    if _has_k5_or_ind5(adj, n):
        # Fallback to Clebsch complement
        n, adj = _clebsch_complement()

    best_n = n
    best_edges = _get_edges(adj, n)

    # Growth phase: try to add vertices
    target_n = 42
    while n < target_n:
        if time.time() > time_limit:
            break
        success, adj, n = _try_add_vertex(adj, n, time_limit, start_time)
        if success:
            if n > best_n:
                best_n = n
                best_edges = _get_edges(adj, n)
        else:
            # If we can't add, try some edge flips to create space, then retry
            flipped = False
            for _ in range(10):
                if time.time() > time_limit:
                    break
                s2, adj2, n2 = _try_flip_edge(adj, n)
                if s2:
                    adj = adj2
                    flipped = True
                    break
            if not flipped and n < target_n:
                # Stuck; break out
                break

    # After growth, try to squeeze out more via edge flips (might enable further growth)
    for _ in range(500):
        if time.time() > time_limit:
            break
        if n >= target_n:
            break
        # Try adding again
        success, adj, n = _try_add_vertex(adj, n, time_limit, start_time)
        if success:
            if n > best_n:
                best_n = n
                best_edges = _get_edges(adj, n)
        else:
            # Try a flip
            s2, adj, n2 = _try_flip_edge(adj, n)
            if s2:
                adj = adj

    # Final: ensure best is valid
    if best_n < 37:
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

    # Validate best_edges
    final_adj = [[False] * best_n for _ in range(best_n)]
    for (a, b) in best_edges:
        final_adj[a][b] = True
        final_adj[b][a] = True
    if _has_k5_or_ind5(final_adj, best_n):
        # Fallback
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

    return best_n, best_edges
