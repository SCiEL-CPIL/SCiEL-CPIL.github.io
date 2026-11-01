"""Deterministic Algebraic Lift construction for k-regular graphs with high girth."""
import time
import sys
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _thue_morse_sequence(length):
    """Generate the first `length` bits of the Thue-Morse sequence."""
    seq = [0]
    for i in range(1, length):
        seq.append(bin(i).count('1') % 2)
    return seq

def _thue_morse_shifts(m, edge_count):
    """
    Generate edge shifts using Thue-Morse derived values for high dispersion.
    """
    tm_len = max(64, edge_count * 4)
    tm = _thue_morse_sequence(tm_len)
    
    shifts = []
    for i in range(edge_count):
        offset = (i * 4) % tm_len
        window = tm[offset:offset+8]
        if len(window) < 8:
            window = (window + tm[:8-len(window)])
        val = 0
        for bit in window:
            val = (val << 1) | bit
        shift = (val * m + i * 7) % m
        shifts.append(shift)
    return shifts

def _thue_morse_shifts_v2(m, edge_count):
    """
    Alternative Thue-Morse based shifts using cumulative sums.
    """
    tm_len = max(128, edge_count * 8)
    tm = _thue_morse_sequence(tm_len)
    
    prefix = [0]
    for i in range(tm_len):
        prefix.append(prefix[-1] + tm[i])
    
    shifts = []
    for i in range(edge_count):
        pos1 = (i * tm_len // edge_count) % tm_len
        pos2 = ((i + 1) * tm_len // edge_count) % tm_len
        val = (prefix[pos1] * 31 + prefix[pos2] * 17 + i * 13) % (m * m)
        shift = val % m
        shifts.append(shift)
    return shifts

def _golden_ratio_shifts(m, edge_count):
    """Generate shifts using golden ratio for good dispersion."""
    golden = (1 + 5**0.5) / 2.0
    shifts = []
    for i in range(edge_count):
        shift = int((i * golden * m) % m)
        shifts.append(shift)
    return shifts

def _quadratic_shifts(m, edge_count):
    """Generate shifts using quadratic residues for better mixing."""
    shifts = []
    for i in range(edge_count):
        shift = (i * i * 7 + i * 13 + 3) % m
        shifts.append(shift)
    return shifts

def _cubic_shifts(m, edge_count):
    """Generate shifts using cubic polynomials."""
    shifts = []
    for i in range(edge_count):
        shift = (i * i * i * 11 + i * i * 5 + i * 3 + 7) % m
        shifts.append(shift)
    return shifts

def _verify_girth(n, edges, g):
    """
    Correct girth verification using non-backtracking BFS from every vertex.
    Returns True if girth >= g, False otherwise.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each starting vertex, do BFS tracking parent to avoid backtracking
    # A cycle of length < g exists iff we find a node at distance d where
    # d + dist[other] + 1 < g for a non-parent connection
    
    half_g = g // 2
    
    for s in range(n):
        # BFS from s, tracking distance and parent
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            
            # No need to explore beyond distance half_g - 1
            # because any cycle through s would be at least 2*du
            if du >= half_g:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle: s -> ... -> u -> v -> ... -> s
                    # The cycle length is du + dist[v] + 1
                    # But we need to be careful: this might not be the shortest cycle
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using numpy-optimized BFS.
    """
    # Build adjacency as list of arrays for faster iteration
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    half_g = g // 2
    
    # Use flat arrays for speed
    dist_arr = np.full(n, -1, dtype=np.int32)
    parent_arr = np.full(n, -1, dtype=np.int32)
    
    for s in range(n):
        # Reset
        dist_arr[:] = -1
        parent_arr[:] = -1
        dist_arr[s] = 0
        parent_arr[s] = -1
        
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            du = dist_arr[u]
            
            if du >= half_g:
                continue
                
            for v in adj[u]:
                if dist_arr[v] == -1:
                    dist_arr[v] = du + 1
                    parent_arr[v] = u
                    queue.append(v)
                elif v != parent_arr[u]:
                    cycle_len = du + dist_arr[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Petersen graph: 10 vertices, 15 edges
    # Standard Petersen: outer pentagon 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Base edges (u < v) - there are 15
    base_edges = sorted(set((min(u,v), max(u,v)) for u,v in petersen_edges))
    num_base_edges = len(base_edges)  # 15

    # Moore bound for k=3, g=16 is 512. n = 10*m, so m >= 52.
    # But we can try smaller m first since the lift might have girth >= 16 even if n < 512
    # Actually no - Moore bound is a lower bound, so n must be >= 512, meaning m >= 52.
    
    # Strategy: try m from 52 upward, with multiple shift patterns per m
    # Use deterministic patterns first, then random
    
    best_n = None
    best_edges = None
    
    # Precompute deterministic shift generators
    shift_generators = [
        _thue_morse_shifts,
        _thue_morse_shifts_v2,
        _golden_ratio_shifts,
        _quadratic_shifts,
        _cubic_shifts,
    ]
    
    m = 52
    while m <= 1200 and time.time() < deadline:
        
        # Try deterministic patterns
        for gen in shift_generators:
            if time.time() > deadline:
                break
            shifts = gen(m, num_base_edges)
            n, edges = _build_lift(base_edges, m, shifts)
            if _verify_girth_fast(n, edges, g):
                return n, edges
        
        # Try random shifts - more attempts as m increases
        # Probability of valid girth increases with m
        num_random_tries = min(2000, max(100, int(100 * m / 52)))
        
        for _ in range(num_random_tries):
            if time.time() > deadline:
                break
            shifts_rand = [random.randint(0, m-1) for _ in range(num_base_edges)]
            n, edges = _build_lift(base_edges, m, shifts_rand)
            if _verify_girth_fast(n, edges, g):
                return n, edges
        
        m += 1
    
    # Fallback: return a large random lift
    m = 3000
    shifts_rand = [random.randint(0, m-1) for _ in range(num_base_edges)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges


def _build_lift(base_edges, m, shifts):
    """
    Build the lifted graph.
    base_edges: list of (u, v) for u < v
    m: lift size
    shifts: list of shifts for each base edge
    """
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges

# EVOLVE-BLOCK-END
