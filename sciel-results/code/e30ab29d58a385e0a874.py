"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import random
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    rng = random.Random(42)

    best = None
    best_n = 0

    # Try deterministic algebraic constructions first
    for target_n in [21, 20]:
        if time.time() > deadline - 60:
            break
        result = _algebraic_search(target_n, deadline - 55, rng)
        if result is not None and result['n'] > best_n:
            best = result
            best_n = result['n']
            if best_n >= 21:
                return best

    # Fallback to local search from random init
    for target_n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 30:
            break
        if target_n <= best_n:
            continue
        result = _local_search(target_n, deadline - 25, rng)
        if result is not None and result['n'] > best_n:
            best = result
            best_n = result['n']
            if best_n >= 21:
                return best

    # If we didn't find anything, use fallback
    if best is None or best_n < 10:
        fallback = _make_fallback(rng)
        if fallback['n'] > best_n:
            best = fallback
            best_n = fallback['n']

    return best


def _make_fallback(rng):
    """Generate a verified valid graph on 10 vertices."""
    n = 10
    for _ in range(5000):
        adj = [[0]*n for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i][j] = adj[j][i] = 1
                    edges.append([i, j])
        if _is_valid_fast(adj, n):
            return {'n': n, 'edges': edges}
    return {'n': 5, 'edges': []}


def _paley_graph(n):
    """
    Construct a Paley-type graph on n vertices.
    For prime p, vertex i is connected to j iff (j-i) mod p is a quadratic residue.
    For composite n, use a pseudorandom linear construction.
    """
    adj = np.zeros((n, n), dtype=np.uint8)
    
    # Check if n is prime for true Paley
    if _is_prime(n):
        p = n
        # Quadratic residues mod p
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                if diff in residues or (p - diff) in residues:
                    adj[i, j] = 1
                    adj[j, i] = 1
    else:
        # For non-prime n, use a pseudo-Paley construction
        # Based on a linear congruential generator with good mixing
        seed_val = 12345
        for i in range(n):
            for j in range(i + 1, n):
                # Use a hash-like function to determine edge
                val = (i * 31 + j * 17 + i * j * 7) % (n * 2)
                if val < n:
                    adj[i, j] = 1
                    adj[j, i] = 1
    
    return adj


def _is_prime(n):
    """Simple primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _is_valid_fast(adj, n):
    """Check if no subset of size >= 6 induces a regular graph using bitmasks."""
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)

    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset_fast(adj_mask, subset):
                return False
    return True


def _is_regular_subset_fast(adj_mask, subset):
    """Check if induced subgraph on subset is regular using bitmask operations."""
    s = len(subset)
    if s < 2:
        return True

    subset_mask = 0
    for v in subset:
        subset_mask |= (1 << v)

    d0 = None
    for v in subset:
        d = bin(adj_mask[v] & subset_mask).count('1')
        if d0 is None:
            d0 = d
        elif d != d0:
            return False
    return True


def _find_violation_fast(adj_mask, n):
    """Find a subset of size >= 6 that induces a regular graph. Returns subset or None."""
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset_fast(adj_mask, subset):
                return subset
    return None


def _algebraic_search(target_n, deadline, rng):
    """
    Use algebraic constructions (Paley-type) as starting points,
    then refine with local search.
    """
    n = target_n
    
    # Generate several algebraic starting points
    candidates = []
    
    # 1. True Paley if n is prime
    if _is_prime(n):
        adj = _paley_graph(n)
        candidates.append(adj.copy())
    
    # 2. Pseudo-Paley for any n
    adj = _paley_graph(n)
    candidates.append(adj.copy())
    
    # 3. Random perturbation of Paley
    for _ in range(10):
        adj_pert = adj.copy()
        # Flip random edges
        for _ in range(n * n // 4):
            i, j = rng.sample(range(n), 2)
            adj_pert[i, j] = 1 - adj_pert[i, j]
            adj_pert[j, i] = adj_pert[i, j]
        candidates.append(adj_pert)
    
    # 4. Pure random graphs
    for _ in range(5):
        adj_rand = np.random.randint(0, 2, (n, n), dtype=np.uint8)
        # Make symmetric
        for i in range(n):
            for j in range(i + 1, n):
                val = adj_rand[i, j]
                adj_rand[i, j] = val
                adj_rand[j, i] = val
        # Zero diagonal
        np.fill_diagonal(adj_rand, 0)
        candidates.append(adj_rand)
    
    # Try each candidate with local search
    for adj in candidates:
        if time.time() > deadline:
            break
        
        # Check if already valid
        if _is_valid_fast(adj, n):
            return _adj_to_result(adj, n)
        
        # Run local search from this starting point
        result = _local_search_from_adj(adj, n, deadline, rng)
        if result is not None:
            return result
    
    return None


def _local_search_from_adj(adj, n, deadline, rng):
    """Local search starting from a given adjacency matrix."""
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)
    
    best_viol_count = _count_violations(adj_mask, n)
    
    for iteration in range(300):
        if time.time() > deadline:
            break
        
        # Find a violation
        viol = _find_violation_fast(adj_mask, n)
        if viol is None:
            return _adj_to_result(adj, n)
        
        subset = viol
        size = len(subset)
        
        # Try flipping edges within the violation subset to break it
        improved = False
        internal_edges = []
        for i in range(size):
            for j in range(i + 1, size):
                u, v = subset[i], subset[j]
                internal_edges.append((u, v))
        
        rng.shuffle(internal_edges)
        
        for u, v in internal_edges:
            old_val = adj[u, v]
            adj[u, v] = adj[v, u] = 1 - old_val
            if old_val:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
            
            new_viol_count = _count_violations(adj_mask, n)
            if new_viol_count < best_viol_count:
                best_viol_count = new_viol_count
                improved = True
                break
            else:
                # Revert
                adj[u, v] = adj[v, u] = old_val
                if old_val:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                else:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
        
        if not improved:
            # Try flipping a random edge
            a, b = rng.sample(range(n), 2)
            old_val2 = adj[a, b]
            adj[a, b] = adj[b, a] = 1 - old_val2
            if old_val2:
                adj_mask[a] &= ~(1 << b)
                adj_mask[b] &= ~(1 << a)
            else:
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
            
            new_viol_count = _count_violations(adj_mask, n)
            if new_viol_count < best_viol_count:
                best_viol_count = new_viol_count
            else:
                # Revert
                adj[a, b] = adj[b, a] = old_val2
                if old_val2:
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
                else:
                    adj_mask[a] &= ~(1 << b)
                    adj_mask[b] &= ~(1 << a)
        
        # Occasionally add random noise
        if rng.random() < 0.1:
            for _ in range(3):
                a, b = rng.sample(range(n), 2)
                adj[a, b] = adj[b, a] = 1 - adj[a, b]
                if adj[a, b]:
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
                else:
                    adj_mask[a] &= ~(1 << b)
                    adj_mask[b] &= ~(1 << a)
    
    return None


def _local_search(target_n, deadline, rng):
    """Local search to find a valid graph on target_n vertices from random init."""
    n = target_n
    
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with random graph
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Check if already valid
        if _is_valid_fast(adj, n):
            return _adj_to_result(adj, n)
        
        # Run local search
        result = _local_search_from_adj(adj, n, deadline, rng)
        if result is not None:
            return result
    
    return None


def _adj_to_result(adj, n):
    """Convert adjacency matrix to result dict."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


def _count_violations(adj_mask, n):
    """Count the number of subsets of size >= 6 that induce a regular graph."""
    count = 0
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset_fast(adj_mask, subset):
                count += 1
    return count
# EVOLVE-BLOCK-END
