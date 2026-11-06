"""
Optimized search for Ramsey graphs R(5,5) > 37.
Uses Paley(37) as seed and adds vertices using constrained search
based on safe neighbor/non-neighbor sets to avoid K5 and Independent 5.
"""

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
    """Check if graph (bitsets) has a clique of size k using backtracking."""
    if n < k:
        return False
    
    # Pre-compute intersections? No, on the fly is better for sparse searches.
    
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

def _find_k_cliques(bits, n, k, limit=1000):
    """Find up to 'limit' cliques of size k. Returns list of masks."""
    cliques = []
    if n < k:
        return cliques
    
    def backtrack(candidates_mask, size_needed, current_clique):
        if len(cliques) >= limit:
            return
        if size_needed == 0:
            cliques.append(current_clique)
            return
        if size_needed == 1:
            if candidates_mask:
                cliques.append(current_clique | candidates_mask)
                return
            
        cm = candidates_mask
        while cm and len(cliques) < limit:
            lb = cm & (-cm)
            v = lb.bit_length() - 1
            cm ^= lb
            new_cands = candidates_mask & bits[v]
            if new_cands.bit_count() < size_needed - 1:
                continue
            backtrack(new_cands, size_needed - 1, current_clique | lb)
            
    all_mask = (1 << n) - 1
    for v in range(n):
        if len(cliques) >= limit:
            break
        cands = all_mask & bits[v]
        if cands.bit_count() < k - 1:
            continue
        backtrack(cands, k - 1, 1 << v)
        
    return cliques

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

def _find_k_independent(bits, n, k, limit=1000):
    """Find up to 'limit' independent sets of size k. Returns list of masks."""
    comp_bits = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_bits[i] = all_mask & ~bits[i]
        comp_bits[i] &= ~(1 << i)
    return _find_k_cliques(comp_bits, n, k, limit)

def _is_valid(bits, n):
    """Check (5,5)-free: no K5, no independent 5."""
    if _has_k_clique(bits, n, 5):
        return False
    if _has_k_independent(bits, n, 5):
        return False
    return True

def _get_safe_sets(bits, n):
    """
    Determine sets of vertices that are 'safe' to be neighbors or non-neighbors
    of a new vertex.
    
    SafeNeighbors: Vertices v such that adding a new vertex connected to v 
                   does not immediately create a K5.
                   This requires that v is not part of any K4.
                   
    SafeNonNeighbors: Vertices v such that adding a new vertex NOT connected to v
                      does not immediately create an Independent 5.
                      This requires that v is not part of any Independent 4.
    """
    # Find K4s
    k4s = _find_k_cliques(bits, n, 4, limit=5000)
    unsafe_neighbors = 0
    for mask in k4s:
        unsafe_neighbors |= mask
        
    # Find Ind-4s
    ind4s = _find_k_independent(bits, n, 4, limit=5000)
    unsafe_non_neighbors = 0
    for mask in ind4s:
        unsafe_non_neighbors |= mask
        
    all_mask = (1 << n) - 1
    safe_neighbors = all_mask & ~unsafe_neighbors
    safe_non_neighbors = all_mask & ~unsafe_non_neighbors
    
    return safe_neighbors, safe_non_neighbors

def _try_add_vertex_constrained(bits, n, rng):
    """
    Try to add a vertex using the safe set constraints.
    Returns (new_bits, new_adj_row) or None.
    """
    safe_neighbors, safe_non_neighbors = _get_safe_sets(bits, n)
    
    # Must connect to: Vertices that are safe neighbors but NOT safe non-neighbors.
    # If we don't connect to them, they become non-neighbors, but they are unsafe non-neighbors (part of Ind-4), 
    # so the new vertex + that Ind-4 would form an Ind-5.
    must_connect = safe_neighbors & ~safe_non_neighbors
    
    # Optional: Vertices that are safe neighbors AND safe non-neighbors.
    # We can choose to connect or not.
    optional = safe_neighbors & safe_non_neighbors
    
    # If a vertex is in neither safe_neighbors nor safe_non_neighbors, 
    # it is part of a K4 AND an Ind-4. 
    # If we connect to it, risk K5. If we don't, risk Ind-5.
    # Actually, if it's unsafe neighbor, it's in K4. If we connect, K4+new=K5. So MUST NOT connect.
    # If it's unsafe non-neighbor, it's in Ind-4. If we don't connect, Ind-4+new=Ind-5. So MUST connect.
    # Contradiction? If a vertex is in both, we can't connect (K5 risk) and can't not connect (Ind-5 risk).
    # So if such a vertex exists, no extension is possible with this logic?
    # Wait, the "risk" is only if the K4/Ind4 is fully connected/disconnected to the new vertex.
    # If v is in K4 {v,a,b,c}, and we connect new to v, we need to ensure we don't connect to a,b,c such that they form K4 with new?
    # No, if new connects to v, and v is in K4, then {new, v, a, b, c} is K5 ONLY IF new connects to a,b,c as well.
    # My previous logic was "v is part of K4" -> "unsafe neighbor". This is too strong.
    # Correct logic:
    # To avoid K5: The set of neighbors of new vertex, let's call it N_new, must NOT contain any K4 from the old graph.
    # So N_new must be a subset of the "K4-free" vertices? No, N_new can contain vertices from K4s, as long as it doesn't contain ALL of them.
    
    # Let's refine:
    # Forbidden Neighbor Sets: Any set S such that S contains a K4.
    # Forbidden Non-Neighbor Sets: Any set S such that S contains an Ind-4.
    
    # This is a hitting set problem or similar.
    # Let's stick to a simpler heuristic first:
    # If we pick N_new randomly from `optional` and force `must_connect`, we should check validity.
    # But `must_connect` logic above was flawed.
    
    # Let's restart the constraint logic properly.
    # 1. Identify all K4s. If N_new contains any K4, Invalid (K5).
    # 2. Identify all Ind-4s. If Non-N_new contains any Ind-4, Invalid (Ind-5).
    
    # Heuristic:
    # Start with N_new = empty.
    # We must avoid picking a set that covers any K4.
    # We must avoid NOT picking a set that covers any Ind-4.
    
    # Let's try a different approach: Randomized Local Search on the connection vector.
    # But the safe sets approach was promising if implemented correctly.
    
    # Correct Safe Sets:
    # A vertex v is "Critical for K5" if it is in a K4.
    # A vertex v is "Critical for Ind5" if it is in an Ind-4.
    
    # If we connect to a set S, S must not contain any K4.
    # If we don't connect to a set T, T must not contain any Ind-4.
    
    # Let's just use the full validity check but guide the search.
    # Generate random subsets of vertices to connect to.
    # Bias: 
    # - If a vertex is in many K4s, prefer NOT connecting to it? (To avoid K5)
    # - If a vertex is in many Ind-4s, prefer CONNECTING to it? (To avoid Ind5)
    
    k4s = _find_k_cliques(bits, n, 4, limit=2000)
    ind4s = _find_k_independent(bits, n, 4, limit=2000)
    
    # Count occurrences
    k4_count = [0] * n
    for mask in k4s:
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            k4_count[v] += 1
            mask ^= lb
            
    ind4_count = [0] * n
    for mask in ind4s:
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            ind4_count[v] += 1
            mask ^= lb
            
    # Probability of connecting to vertex i:
    # High ind4_count -> High prob connect (to break Ind-5)
    # High k4_count -> Low prob connect (to break K-5)
    
    probs = []
    for i in range(n):
        p = 0.5
        p += ind4_count[i] * 0.05
        p -= k4_count[i] * 0.05
        p = max(0.01, min(0.99, p))
        probs.append(p)
        
    # Try several random configurations based on these probabilities
    for _ in range(50):
        if _time_left() < 10:
            return None
            
        new_row = [0] * (n + 1)
        added_edges = []
        
        for i in range(n):
            if rng.random() < probs[i]:
                new_row[i] = 1
                added_edges.append((i, n))
                
        new_bits = bits[:] + [0]
        new_bits[n] = 0
        for i in range(n):
            if new_row[i]:
                new_bits[i] |= (1 << n)
                new_bits[n] |= (1 << i)
                
        if _is_valid(new_bits, n + 1):
            return new_bits, new_row
            
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
    
    # Phase 1: Try to add vertices
    max_add_attempts = 100
    for attempt in range(max_add_attempts):
        if _time_left() < 20:
            break
            
        result = _try_add_vertex_constrained(bits, n, rng)
        if result is not None:
            new_bits, new_row = result
            n += 1
            bits = new_bits
            # Update adj
            new_adj = [row[:] for row in adj]
            for i in range(n - 1):
                new_adj[i].append(new_row[i])
            new_adj.append(new_row[:])
            adj = new_adj
            
            # Update edges
            for i in range(n - 1):
                if new_row[i]:
                    edges.append((i, n - 1))
                    
            best_n = n
            best_adj = [row[:] for row in adj]
            best_edges = edges[:]
            best_bits = bits[:]
        else:
            # If we can't add a vertex, we might try to remove one and re-add?
            # Or just stop.
            # Let's try a few more random seeds if we stall
            if attempt > 10:
                break
                
    return best_n, best_edges
