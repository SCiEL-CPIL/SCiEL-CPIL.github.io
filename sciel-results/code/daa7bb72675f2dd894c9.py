"""Combined approach: Paley(37) seed + structured vertex addition with
efficient clique/independent-set checking via bitsets."""

import time
import random
import sys

# Deadline guard: evaluator kills at 330s, use 290s internal for safety
_DEADLINE = 290.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _paley_37():
    """Return adjacency matrix (list of lists of int) and edge list for Paley(37)."""
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

def _to_bitsets(adj, n):
    """Convert adjacency matrix to list of Python ints (bitsets)."""
    bits = [0] * n
    for i in range(n):
        b = 0
        for j in range(n):
            if adj[i][j]:
                b |= (1 << j)
        bits[i] = b
    return bits

def _has_k_clique(bits, n, k):
    """Check if graph (bitsets) has a clique of size k. Returns True/False."""
    if n < k:
        return False
    
    # Backtracking with bitsets
    def backtrack(candidates_mask, size_needed):
        if size_needed == 0:
            return True
        if size_needed == 1:
            return candidates_mask != 0
        # Pick a vertex with highest degree within candidates for pruning
        # Simple: iterate candidates
        cm = candidates_mask
        while cm:
            # Get lowest set bit
            lb = cm & (-cm)
            v = lb.bit_length() - 1
            cm ^= lb
            new_cands = candidates_mask & bits[v]
            if new_cands.bit_count() < size_needed - 1:
                continue
            if backtrack(new_cands, size_needed - 1):
                return True
        return False
    
    all_mask = (1 << n) - 1
    # Try starting from each vertex
    for v in range(n):
        cands = all_mask & bits[v]
        if cands.bit_count() < k - 1:
            continue
        if backtrack(cands, k - 1):
            return True
    return False

def _has_k_independent(bits, n, k):
    """Check if graph (bitsets) has an independent set of size k."""
    if n < k:
        return False
    # Independent set in G = clique in complement
    comp_bits = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_bits[i] = all_mask & ~bits[i]
        comp_bits[i] &= ~(1 << i)  # remove self
    return _has_k_clique(comp_bits, n, k)

def _is_valid(bits, n):
    """Check (5,5)-free: no K5, no independent 5."""
    if _has_k_clique(bits, n, 5):
        return False
    if _has_k_independent(bits, n, 5):
        return False
    return True

def _try_add_vertex(adj, n, bits, density, rng):
    """Try to add one vertex with given edge density to existing vertices.
    Returns (new_n, new_adj, new_bits, added_edges) or None if invalid."""
    new_n = n + 1
    new_row = [0] * new_n
    added_edges = []
    
    for i in range(n):
        if rng.random() < density:
            new_row[i] = 1
            added_edges.append((i, n))
    
    # Build new adjacency
    new_adj = [row[:] for row in adj]
    for i in range(n):
        new_adj[i].append(new_row[i])
    new_adj.append(new_row[:])
    
    # Build new bitsets
    new_bits = bits[:] + [0]
    new_bits[n] = 0
    for i in range(n):
        if new_row[i]:
            new_bits[i] |= (1 << n)
            new_bits[n] |= (1 << i)
    
    if _is_valid(new_bits, new_n):
        return new_n, new_adj, new_bits, added_edges
    return None

def _try_remove_edge_and_add(adj, n, bits, edge_idx, density, rng):
    """Remove edge edge_idx from graph, then try to add a vertex.
    Returns (new_n, new_adj, new_bits, new_edges_list) or None."""
    i, j = edge_idx
    # Remove edge
    new_adj = [row[:] for row in adj]
    new_adj[i][j] = 0
    new_adj[j][i] = 0
    
    new_bits = bits[:]
    new_bits[i] &= ~(1 << j)
    new_bits[j] &= ~(1 << i)
    
    # Rebuild edge list
    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            if new_adj[a][b]:
                edges.append((a, b))
    
    result = _try_add_vertex(new_adj, n, new_bits, density, rng)
    if result is not None:
        new_n, new_adj2, new_bits2, added = result
        edges.extend(added)
        return new_n, new_adj2, new_bits2, edges
    return None

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with Paley(37)
    n, adj, edges = _paley_37()
    bits = _to_bitsets(adj, n)
    
    # Verify seed
    if not _is_valid(bits, n):
        # Shouldn't happen, but return as-is
        return n, edges
    
    best_n = n
    best_adj = [row[:] for row in adj]
    best_edges = edges[:]
    best_bits = bits[:]
    
    rng = random.Random(42)
    
    # Phase 1: Try to add vertices with various densities
    max_add_attempts = 50
    densities = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    added_count = 0
    for attempt in range(max_add_attempts):
        if _time_left() < 30:
            break
        
        # Try different densities
        found = False
        for d in densities:
            if _time_left() < 30:
                break
            for trial in range(20):
                if _time_left() < 30:
                    break
                result = _try_add_vertex(adj, n, bits, d, rng)
                if result is not None:
                    n, adj, bits, added_edges = result
                    edges.extend(added_edges)
                    best_n = n
                    best_adj = [row[:] for row in adj]
                    best_edges = edges[:]
                    best_bits = bits[:]
                    added_count += 1
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Phase 2: Try removing an edge to create room
            if _time_left() < 60:
                break
            
            removed = False
            # Try removing random edges
            for _ in range(100):
                if _time_left() < 60:
                    break
                if not edges:
                    break
                ei = rng.randrange(len(edges))
                d = rng.choice(densities)
                result = _try_remove_edge_and_add(adj, n, bits, edges[ei], d, rng)
                if result is not None:
                    n, adj, bits, edges = result
                    best_n = n
                    best_adj = [row[:] for row in adj]
                    best_edges = edges[:]
                    best_bits = bits[:]
                    removed = True
                    break
            
            if not removed:
                # Try a few more random addition attempts with fresh randomness
                for _ in range(50):
                    if _time_left() < 60:
                        break
                    d = rng.choice(densities)
                    result = _try_add_vertex(adj, n, bits, d, rng)
                    if result is not None:
                        n, adj, bits, added_edges = result
                        edges.extend(added_edges)
                        best_n = n
                        best_adj = [row[:] for row in adj]
                        best_edges = edges[:]
                        best_bits = bits[:]
                        removed = True
                        break
                
                if not removed:
                    break  # Give up adding more
    
    # Return best result
    return best_n, best_edges

# Keep the required entry point name
# The function is already named construct()
