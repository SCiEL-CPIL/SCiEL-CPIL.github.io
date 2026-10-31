import time
import random
import numpy as np
from collections import deque

# Keep the original LUW code as a fallback
def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _get_petersen_cycles(max_len):
    """
    Pre-compute all cycles in the Petersen graph up to max_len.
    Returns a list of tuples of edge indices.
    """
    # Petersen graph edges
    # 0-1, 1-2, 2-3, 3-4, 4-0 (outer)
    # 5-7, 7-9, 9-6, 6-8, 8-5 (inner)
    # 0-5, 1-6, 2-7, 3-8, 4-9 (spokes)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n = 10
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    cycles = []
    # To avoid duplicates, we only start DFS from vertex 0, or canonicalize cycles.
    # For a small graph, we can just find all and deduplicate.
    # DFS from each vertex, but only record if start < min(cycle) to avoid rotation duplicates.
    # And only record if start < max(cycle) to avoid reversal duplicates?
    # Simpler: Just find all, convert to frozenset of edges, and deduplicate.
    
    seen = set()
    
    def dfs(start, curr, visited, edge_path, length):
        if length > max_len:
            return
        for next_v, edge_idx in adj[curr]:
            if next_v == start and length >= 3:
                # Found a cycle
                # Canonicalize: sort edge indices
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
                
    for start in range(n):
        visited = {start}
        dfs(start, start, visited, [], 0)
        
    return cycles

def _find_voltages(m, cycles, deadline):
    """
    Find voltages in Z_m for the Petersen edges such that no cycle in 'cycles' sums to 0 mod m.
    """
    n_edges = 15
    # Initialize random voltages
    voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
    
    def count_bad():
        bad = 0
        for cycle in cycles:
            s = sum(voltages[i] for i in cycle) % m
            if s == 0:
                bad += 1
        return bad
    
    current_bad = count_bad()
    if current_bad == 0:
        return voltages
        
    for _ in range(10000):
        if time.time() > deadline:
            return None
            
        if current_bad == 0:
            return voltages
            
        # Pick a random bad cycle
        # We can pre-compute which cycles are bad, but it's cheap enough to scan
        bad_cycles = []
        for cycle in cycles:
            if sum(voltages[i] for i in cycle) % m == 0:
                bad_cycles.append(cycle)
                
        if not bad_cycles:
            return voltages
            
        cycle = random.choice(bad_cycles)
        edge_idx = random.choice(cycle)
        
        # Change voltage
        old_v = voltages[edge_idx]
        new_v = random.randint(0, m - 1)
        if new_v == old_v:
            continue
        voltages[edge_idx] = new_v
        
        new_bad = count_bad()
        if new_bad < current_bad:
            current_bad = new_bad
        else:
            # Revert (or keep if equal, but revert is safer to avoid plateau)
            voltages[edge_idx] = old_v
            # Try a different edge or random restart if stuck?
            # For now, just revert.
            
    return None

def _petersen_lift_with_voltages(m, voltages, seed=None):
    """
    Construct the lift using specific voltages.
    """
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    n = 10 * m
    edges = []
    for idx, (u, v) in enumerate(petersen_edges):
        vol = voltages[idx]
        for j in range(m):
            a = u * m + j
            b = v * m + ((j + vol) % m)
            edges.append((a, b))
            
    return n, edges

def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_depth = (min_girth - 1) // 2
    
    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])
        
        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    if k != 3:
        # Fallback for non-3-regular
        return _luw_construct(k, g)
        
    # Pre-compute cycles for Petersen graph
    # We need girth >= g. So we forbid cycles of length < g.
    # The lift girth is determined by base cycles.
    # We need to forbid base cycles of length L such that the lift cycle length < g.
    # If the base cycle length is L, the lift cycle length is a multiple of L?
    # No, the lift cycle length is exactly L if the voltage sum is 0 mod m?
    # Yes, if sum(vol) = 0 mod m, the lift has a cycle of length L.
    # If sum(vol) != 0, the lift does NOT have a cycle of length L.
    # It might have a cycle of length k*L? No, it's a cover.
    # So we just need to ensure that for all base cycles C with length L < g, sum(vol(C)) != 0 mod m.
    # Note: If L >= g, we don't care if sum is 0, because the cycle length L is already >= g.
    
    cycles_to_forbid = _get_petersen_cycles(g - 1)
    
    # Try different m values
    # n = 10 * m
    # We want small n.
    # Moore bound is ~500. So m ~ 50.
    # But we need enough "room" in Z_m to satisfy constraints.
    # Let's try m from 50 to 200.
    
    for m in [50, 60, 80, 100, 120, 150, 200, 300]:
        if time.time() > deadline - 5:
            break
            
        # Try to find voltages
        found = False
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            random.seed(attempt * 1000 + m) # Deterministic-ish for reproducibility in tests?
            # Actually, just use random
            voltages = _find_voltages(m, cycles_to_forbid, deadline - 5)
            
            if voltages is not None:
                n, edges = _petersen_lift_with_voltages(m, voltages)
                
                # Verify
                if _check_girth_fast(n, edges, g):
                    return n, edges
                else:
                    # Should not happen if voltage search was correct
                    pass
        
    # Fallback to LUW
    return _luw_construct(k, g)
