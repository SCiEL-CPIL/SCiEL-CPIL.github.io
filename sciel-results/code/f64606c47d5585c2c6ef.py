"""
Search for a large graph with no regular induced subgraph on 6+ vertices.
Uses Paley graph initialization + local search for fast convergence.
"""
import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 130  # 20s safety margin before 150s kill

    best_n = 0
    best_edges = []

    # Try n=20 first (known to exist), then n=21 if time permits
    for target_n in [20, 19, 18]:
        if time.time() > deadline - 20:
            break

        found = _solve(target_n, deadline)
        if found is not None:
            best_n = target_n
            best_edges = found
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}

    # Fallback: trivially valid small graph
    return {'n': 5, 'edges': []}


def _solve(n, deadline):
    """
    Try to find a valid graph on n vertices using Paley init + local search.
    Returns list of edges if found, else None.
    """
    # Try multiple Paley-like initializations and random perturbations
    for attempt in range(20):
        if time.time() > deadline - 15:
            return None

        if attempt == 0:
            # Pure Paley graph
            adj = _paley_graph(n)
        elif attempt == 1:
            # Paley with small random perturbation
            adj = _paley_graph(n)
            rng = random.Random(42 + attempt)
            for _ in range(n):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i != j:
                    if adj[i] & (1 << j):
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        else:
            # More random: start from Paley but flip more edges
            adj = _paley_graph(n)
            rng = random.Random(100 + attempt * 7)
            flips = min(20, n * 2)
            for _ in range(flips):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i != j:
                    if adj[i] & (1 << j):
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

        # Local search: repeatedly find and fix violations
        adj = _local_search(adj, n, deadline)

        if adj is None:
            continue

        if _is_valid_fast(adj, n):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return edges

    return None


def _paley_graph(n):
    """
    Construct a Paley-like graph on n vertices.
    For prime p, Paley graph connects i,j if (j-i) is a quadratic residue mod p.
    For general n, we use a modular approach with a large prime.
    """
    adj = [0] * n
    # Use a prime >= n for the modular arithmetic
    p = _next_prime(n)

    # Precompute quadratic residues mod p
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)

    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    return adj


def _next_prime(n):
    """Return the smallest prime >= n."""
    if n <= 2:
        return 2
    candidate = n if n % 2 == 1 else n + 1
    while True:
        if _is_prime(candidate):
            return candidate
        candidate += 2


def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True


def _local_search(adj, n, deadline):
    """
    Local search: repeatedly find a violating subset and flip an edge to fix it.
    Returns modified adj if successful within time, else None.
    """
    max_iters = 50000
    for it in range(max_iters):
        if time.time() > deadline - 10:
            return None

        # Find a violation
        violation = _find_violation(adj, n)
        if violation is None:
            return adj

        subset, v0, v1 = violation

        # Try flipping edges involving vertices in the subset to break regularity
        # Strategy: flip an edge to change degree of one vertex in the subset
        rng = random.Random(it)
        improved = False

        # Try flipping each edge within the subset
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                a, b = subset[i], subset[j]
                # Flip edge (a, b)
                if adj[a] & (1 << b):
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)

                # Check if this fixed the violation (and didn't create worse ones)
                new_violation = _find_violation(adj, n)
                if new_violation is None:
                    return adj

                # If still violating, undo the flip (simple hill climbing)
                if adj[a] & (1 << b):
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)

        # If no simple flip helped, try a random edge flip
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        if a != b:
            if adj[a] & (1 << b):
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
            else:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

            # Check if this helped
            if _find_violation(adj, n) is None:
                return adj

            # Undo if it made things worse (check violation count)
            # For simplicity, just undo
            if adj[a] & (1 << b):
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
            else:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

    return None


def _find_violation(adj, n):
    """
    Find a subset of size >= 6 that induces a regular graph.
    Returns (subset, v0, v1) or None if no violation.
    Checks size 6 first (most common violation), then larger sizes.
    """
    # Check size 6 subsets first (most likely to find a violation quickly)
    for subset in itertools.combinations(range(n), 6):
        mask = 0
        for v in subset:
            mask |= (1 << v)

        # Check if regular
        v0 = subset[0]
        d0 = (adj[v0] & mask).bit_count()

        is_regular = True
        for v in subset[1:]:
            dv = (adj[v] & mask).bit_count()
            if dv != d0:
                is_regular = False
                break

        if is_regular:
            return (subset, v0, subset[1])

    # Check size 7+ if needed (less common)
    for size in range(7, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)

            v0 = subset[0]
            d0 = (adj[v0] & mask).bit_count()

            is_regular = True
            for v in subset[1:]:
                dv = (adj[v] & mask).bit_count()
                if dv != d0:
                    is_regular = False
                    break

            if is_regular:
                return (subset, v0, subset[1])

    return None


def _is_valid_fast(adj, n):
    """
    Check if no subset of size >= 6 induces a regular graph.
    """
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)

            v0 = subset[0]
            d0 = (adj[v0] & mask).bit_count()

            is_regular = True
            for v in subset[1:]:
                dv = (adj[v] & mask).bit_count()
                if dv != d0:
                    is_regular = False
                    break

            if is_regular:
                return False

    return True
