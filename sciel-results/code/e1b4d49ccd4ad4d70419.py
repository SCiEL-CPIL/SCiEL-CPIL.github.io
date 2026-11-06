"""Improved approach: use bit-parallel clique/independent-set detection and a smarter
vertex-addition strategy. Start from Paley(37), then try to add vertices using structured
patterns derived from known constructions, with fallback to random search."""
import time
import random
import sys

DEADLINE = 295.0  # Safety margin before 300s limit

_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0, DEADLINE - _elapsed())

def _build_bitsets(adj, n):
    """Build bitset representation of neighbors for fast clique detection."""
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and adj[i][j]:
                mask |= (1 << j)
        neigh[i] = mask
    return neigh

def _has_k_clique_bitset(neigh, n, k):
    """Check for clique of size k using bitset backtracking."""
    if n < k:
        return False
    if k <= 1:
        return n >= 1
    
    def backtrack(candidates, depth):
        if depth == k:
            return True
        if candidates == 0:
            return False
        # Count bits quickly
        c = candidates
        bit_count = 0
        while c:
            c &= c - 1
            bit_count += 1
        if bit_count < k - depth:
            return False
        
        # Try vertices in candidates
        c = candidates
        while c:
            v_bit = c & (-c)
            c ^= v_bit
            v = v_bit.bit_length() - 1
            new_cands = candidates & neigh[v] & (v_bit - 1)  # only lower bits for ordering
            if new_cands:
                if backtrack(new_cands, depth + 1):
                    return True
            elif depth + 1 == k:
                return True
        return False
    
    all_bits = (1 << n) - 1
    return backtrack(all_bits, 0)

def _has_k_independent_bitset(neigh, n, k):
    """Independent set in G = clique in complement."""
    if n < k:
        return False
    comp_neigh = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_neigh[i] = all_mask ^ neigh[i] ^ (1 << i)
    return _has_k_clique_bitset(comp_neigh, n, k)

def _is_valid_55_free(adj, n):
    """Check no K5 and no independent 5-set."""
    neigh = _build_bitsets(adj, n)
    if _has_k_clique_bitset(neigh, n, 5):
        return False
    if _has_k_independent_bitset(neigh, n, 5):
        return False
    return True

def _paley_37():
    """Construct the Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[0] * q for _ in range(q)]
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append((i, j))
    return q, adj, edges

def _try_add_vertex(adj, n, pattern):
    """Try to add a new vertex with the given adjacency pattern to existing vertices.
    pattern is a list of 0s and 1s of length n.
    Returns (new_adj, new_n) if valid, else (None, n)."""
    new_n = n + 1
    new_adj = [row[:] for row in adj]
    new_adj.append([0] * new_n)
    for i in range(n):
        new_adj[i].append(pattern[i])
        new_adj[n][i] = pattern[i]
    
    # Quick check: no K5 and no independent 5
    if not _is_valid_55_free(new_adj, new_n):
        return None, n
    return new_adj, new_n

def _structured_patterns(n, seed=42):
    """Generate structured adjacency patterns that are more likely to be valid."""
    rng = random.Random(seed)
    patterns = []
    
    # Pattern 1: Connect to a random subset with density ~0.5
    for _ in range(5):
        p = [1 if rng.random() < 0.5 else 0 for _ in range(n)]
        patterns.append(p)
    
    # Pattern 2: Connect to vertices with even/odd indices
    patterns.append([1 if i % 2 == 0 else 0 for i in range(n)])
    patterns.append([1 if i % 2 == 1 else 0 for i in range(n)])
    
    # Pattern 3: Connect to a block of consecutive vertices
    block_size = max(1, n // 2)
    start = rng.randint(0, n - block_size)
    p = [0] * n
    for i in range(start, min(start + block_size, n)):
        p[i] = 1
    patterns.append(p)
    
    # Pattern 4: Use the Paley structure extended
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    p = [0] * n
    for i in range(n):
        if i < q:
            # For vertices in the Paley part, use a shifted Paley pattern
            p[i] = 1 if (n - i) % q in residues else 0
        else:
            p[i] = 1 if rng.random() < 0.5 else 0
    patterns.append(p)
    
    # Pattern 5: Sparse connection (fewer edges)
    p = [0] * n
    k = max(1, n // 4)
    chosen = rng.sample(range(n), min(k, n))
    for i in chosen:
        p[i] = 1
    patterns.append(p)
    
    # Pattern 6: Dense connection (more edges)
    p = [0] * n
    k = max(1, 3 * n // 4)
    chosen = rng.sample(range(n), min(k, n))
    for i in chosen:
        p[i] = 1
    patterns.append(p)
    
    return patterns

def _greedy_add_vertices(adj, n, max_add=15):
    """Try to add vertices one by one using structured and random patterns."""
    current_adj = adj
    current_n = n
    added_count = 0
    
    for add_round in range(max_add):
        if _elapsed() > DEADLINE - 20:
            break
        
        # Generate a batch of candidate patterns
        rng = random.Random(int(time.time() * 1000) % (2**31))
        candidates = _structured_patterns(current_n, seed=rng.randint(0, 2**31))
        
        # Also add purely random patterns
        for _ in range(10):
            p = [1 if rng.random() < 0.5 else 0 for _ in range(current_n)]
            candidates.append(p)
        
        found = False
        for pattern in candidates:
            if _elapsed() > DEADLINE - 20:
                break
            
            new_adj, new_n = _try_add_vertex(current_adj, current_n, pattern)
            if new_adj is not None:
                current_adj = new_adj
                current_n = new_n
                added_count += 1
                found = True
                break
        
        if not found:
            # Try more random patterns with varying densities
            for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
                if _elapsed() > DEADLINE - 20:
                    break
                for trial in range(20):
                    if _elapsed() > DEADLINE - 20:
                        break
                    p = [1 if rng.random() < density else 0 for _ in range(current_n)]
                    new_adj, new_n = _try_add_vertex(current_adj, current_n, p)
                    if new_adj is not None:
                        current_adj = new_adj
                        current_n = new_n
                        added_count += 1
                        found = True
                        break
                if found:
                    break
            
            if not found:
                # Give up adding more vertices
                break
    
    return current_adj, current_n

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with Paley(37)
    n, adj, edges = _paley_37()
    
    # Verify the seed is valid
    if not _is_valid_55_free(adj, n):
        # This shouldn't happen, but handle gracefully
        pass
    
    best_n = n
    best_adj = [row[:] for row in adj]
    best_edges = edges[:]
    
    # Try to add vertices
    if _elapsed() < DEADLINE - 30:
        new_adj, new_n = _greedy_add_vertices(adj, n, max_add=10)
        if new_n > best_n:
            best_n = new_n
            best_adj = new_adj
    
    # Build the edge list from the best adjacency matrix
    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i][j]:
                final_edges.append((i, j))
    
    return best_n, final_edges

# Entry point
# The construct() function above is the entry point
