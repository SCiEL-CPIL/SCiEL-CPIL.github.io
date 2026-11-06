"""Combined approach: Paley(37) seed + structured vertex addition with
efficient clique/independent-set checking via bitsets and smart patterns."""

import time
import random
import sys

_DEADLINE = 290.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _paley_37():
    """Return adjacency matrix and edge list for Paley(37)."""
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
    """Check if graph (bitsets) has a clique of size k."""
    if n < k:
        return False
    
    def backtrack(candidates_mask, size_needed):
        if size_needed == 0:
            return True
        if size_needed == 1:
            return candidates_mask != 0
        cm = candidates_mask
        while cm:
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
    comp_bits = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_bits[i] = all_mask & ~bits[i]
        comp_bits[i] &= ~(1 << i)
    return _has_k_clique(comp_bits, n, k)

def _is_valid(bits, n):
    """Check (5,5)-free: no K5, no independent 5."""
    if _has_k_clique(bits, n, 5):
        return False
    if _has_k_independent(bits, n, 5):
        return False
    return True

def _try_add_vertex_smart(adj, n, bits, rng):
    """Try to add one vertex using structured patterns.
    Returns (new_n, new_adj, new_bits, added_edges) or None if invalid."""
    new_n = n + 1
    all_mask_old = (1 << n) - 1
    
    # Strategy 1: Try connecting to vertices that are mutually non-adjacent
    # (to avoid creating K5) but not too many (to avoid independent 5)
    
    # Generate candidate patterns based on graph structure
    patterns = []
    
    # Pattern A: Connect to a random subset of moderate size
    for size in range(5, 30):
        if size > n:
            break
        patterns.append(('random_size', size))
    
    # Pattern B: Connect to neighbors of a specific vertex
    for v in range(min(n, 20)):
        patterns.append(('neighborhood', v))
    
    # Pattern C: Connect to non-neighbors of a specific vertex  
    for v in range(min(n, 20)):
        patterns.append(('non_neighbor', v))
    
    # Pattern D: Connect to vertices with low degree
    degs = [bits[i].bit_count() for i in range(n)]
    low_deg = sorted(range(n), key=lambda x: degs[x])[:15]
    patterns.append(('low_degree', low_deg))
    
    high_deg = sorted(range(n), key=lambda x: -degs[x])[:15]
    patterns.append(('high_degree', high_deg))
    
    # Shuffle patterns
    rng.shuffle(patterns)
    
    for pattern in patterns:
        if _time_left() < 15:
            break
        
        new_row = [0] * new_n
        added_edges = []
        
        if pattern[0] == 'random_size':
            size = pattern[1]
            if size > n:
                continue
            chosen = rng.sample(range(n), size)
            for i in chosen:
                new_row[i] = 1
                added_edges.append((i, n))
                
        elif pattern[0] == 'neighborhood':
            v = pattern[1]
            for i in range(n):
                if bits[v] & (1 << i):
                    new_row[i] = 1
                    added_edges.append((i, n))
                    
        elif pattern[0] == 'non_neighbor':
            v = pattern[1]
            for i in range(n):
                if i != v and not (bits[v] & (1 << i)):
                    new_row[i] = 1
                    added_edges.append((i, n))
                    
        elif pattern[0] == 'low_degree':
            for i in pattern[1]:
                new_row[i] = 1
                added_edges.append((i, n))
                
        elif pattern[0] == 'high_degree':
            for i in pattern[1]:
                new_row[i] = 1
                added_edges.append((i, n))
        
        # Build new bitsets
        new_bits = bits[:] + [0]
        new_bits[n] = 0
        for i in range(n):
            if new_row[i]:
                new_bits[i] |= (1 << n)
                new_bits[n] |= (1 << i)
        
        if _is_valid(new_bits, new_n):
            # Build new adjacency
            new_adj = [row[:] for row in adj]
            for i in range(n):
                new_adj[i].append(new_row[i])
            new_adj.append(new_row[:])
            return new_n, new_adj, new_bits, added_edges
    
    # Try random patterns with various densities
    for d in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        if _time_left() < 15:
            break
        for trial in range(10):
            if _time_left() < 15:
                break
            new_row = [0] * new_n
            added_edges = []
            for i in range(n):
                if rng.random() < d:
                    new_row[i] = 1
                    added_edges.append((i, n))
            
            new_bits = bits[:] + [0]
            new_bits[n] = 0
            for i in range(n):
                if new_row[i]:
                    new_bits[i] |= (1 << n)
                    new_bits[n] |= (1 << i)
            
            if _is_valid(new_bits, new_n):
                new_adj = [row[:] for row in adj]
                for i in range(n):
                    new_adj[i].append(new_row[i])
                new_adj.append(new_row[:])
                return new_n, new_adj, new_bits, added_edges
    
    return None

def _try_remove_and_add(adj, n, bits, rng):
    """Remove a random edge and try to add a vertex."""
    # Find all edges
    edges = []
    for i in range(n):
        b = bits[i]
        while b:
            lb = b & (-b)
            j = lb.bit_length() - 1
            if j > i:
                edges.append((i, j))
            b ^= lb
    
    if not edges:
        return None
    
    # Try removing a few random edges
    for _ in range(50):
        if _time_left() < 15:
            break
        ei = rng.randrange(len(edges))
        i, j = edges[ei]
        
        # Remove edge
        new_bits = bits[:]
        new_bits[i] &= ~(1 << j)
        new_bits[j] &= ~(1 << i)
        
        # Try to add vertex with various patterns
        new_n = n + 1
        for d in [0.3, 0.4, 0.5, 0.6]:
            if _time_left() < 15:
                break
            new_row = [0] * new_n
            added_edges = []
            for k in range(n):
                if rng.random() < d:
                    new_row[k] = 1
                    added_edges.append((k, n))
            
            final_bits = new_bits[:] + [0]
            final_bits[n] = 0
            for k in range(n):
                if new_row[k]:
                    final_bits[k] |= (1 << n)
                    final_bits[n] |= (1 << k)
            
            if _is_valid(final_bits, new_n):
                # Rebuild adjacency
                new_adj = [[0] * new_n for _ in range(new_n)]
                for a in range(n):
                    for b in range(a + 1, n):
                        if new_bits[a] & (1 << b):
                            new_adj[a][b] = 1
                            new_adj[b][a] = 1
                for k in range(n):
                    if new_row[k]:
                        new_adj[k][n] = 1
                        new_adj[n][k] = 1
                
                # Rebuild edge list
                new_edges = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if new_adj[a][b]:
                            new_edges.append((a, b))
                new_edges.extend(added_edges)
                
                return new_n, new_adj, final_bits, new_edges
    
    return None

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with Paley(37)
    n, adj, edges = _paley_37()
    bits = _to_bitsets(adj, n)
    
    # Verify seed
    if not _is_valid(bits, n):
        return n, edges
    
    best_n = n
    best_adj = [row[:] for row in adj]
    best_edges = edges[:]
    best_bits = bits[:]
    
    rng = random.Random(42)
    
    # Phase 1: Try to add vertices with smart patterns
    added_count = 0
    max_add_attempts = 30
    
    for attempt in range(max_add_attempts):
        if _time_left() < 30:
            break
        
        result = _try_add_vertex_smart(adj, n, bits, rng)
        if result is not None:
            n, adj, bits, added_edges = result
            edges.extend(added_edges)
            best_n = n
            best_adj = [row[:] for row in adj]
            best_edges = edges[:]
            best_bits = bits[:]
            added_count += 1
        else:
            # Phase 2: Try removing an edge to create room
            if _time_left() < 60:
                break
            
            result = _try_remove_and_add(adj, n, bits, rng)
            if result is not None:
                n, adj, bits, edges = result
                best_n = n
                best_adj = [row[:] for row in adj]
                best_edges = edges[:]
                best_bits = bits[:]
            else:
                break  # Give up adding more
    
    # Return best result
    return best_n, best_edges
