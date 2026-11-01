import time
import random
import sys
from itertools import combinations, permutations

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    if k != 5 or g != 9:
        return _luw_construct(k, g, deadline)
    
    # Strategy: Lift K6 (5-regular, 6 vertices) by prime p
    # K6 has 15 edges. We need voltage assignment such that no cycle has zero sum.
    # Cycles in K6: lengths 3, 4, 5, 6
    
    # K6 edges (15 total)
    k6_edges = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    
    # Enumerate all simple cycles in K6
    cycles = _enumerate_k6_cycles()
    
    # Try primes: n = 6*p, want n close to but >= Moore bound (~426)
    # 426/6 = 71, so try primes around 71
    primes = [71, 67, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    for p in primes:
        if time.time() > deadline - 10:
            break
            
        n = 6 * p
        if n < 426:
            continue  # Below Moore bound, impossible
        
        # Search for valid voltage assignment using backtracking with constraint propagation
        found = _find_voltage_assignment(k6_edges, cycles, p, deadline)
        if found is not None:
            voltages = found
            # Construct the lifted graph
            edges = []
            for u, v in k6_edges:
                w = voltages[(u, v)]
                for i in range(p):
                    u_idx = u * p + i
                    v_idx = v * p + (i + w) % p
                    edges.append((u_idx, v_idx))
            
            # Verify girth
            if _check_girth(edges, n, g):
                return n, edges
    
    # Fallback to LUW
    return _luw_construct(k, g, deadline)


def _enumerate_k6_cycles():
    """Enumerate all simple cycles in K6 (lengths 3-6)."""
    cycles = []
    n = 6
    
    # Length 3: all triples
    for i in range(n):
        for j in range(i + 1, n):
            for l in range(j + 1, n):
                cycles.append([(i, j), (j, l), (l, i)])
    
    # Length 4: choose 4 vertices, 3 Hamiltonian cycles on K4
    for verts in combinations(range(n), 4):
        v0, v1, v2, v3 = verts
        cycles.append([(v0, v1), (v1, v2), (v2, v3), (v3, v0)])
        cycles.append([(v0, v1), (v1, v3), (v3, v2), (v2, v0)])
        cycles.append([(v0, v2), (v2, v1), (v1, v3), (v3, v0)])
    
    # Length 5: choose 5 vertices, (5-1)!/2 = 12 cycles each
    for verts in combinations(range(n), 5):
        v0 = verts[0]
        others = list(verts[1:])
        seen = set()
        for perm in permutations(others):
            cycle_edges = []
            prev = v0
            for u in perm:
                cycle_edges.append((min(prev, u), max(prev, u)))
                prev = u
            cycle_edges.append((min(prev, v0), max(prev, v0)))
            key = tuple(sorted(cycle_edges))
            if key not in seen:
                seen.add(key)
                cycles.append(cycle_edges)
    
    # Length 6: Hamiltonian cycles on K6, (6-1)!/2 = 60
    seen = set()
    for perm in permutations(range(1, 6)):
        cycle_edges = []
        prev = 0
        for u in perm:
            cycle_edges.append((min(prev, u), max(prev, u)))
            prev = u
        cycle_edges.append((min(prev, 0), max(prev, 0)))
        key = tuple(sorted(cycle_edges))
        if key not in seen:
            seen.add(key)
            cycles.append(cycle_edges)
    
    # Remove duplicates
    unique_cycles = []
    seen_cycles = set()
    for c in cycles:
        key = tuple(sorted(c))
        if key not in seen_cycles:
            seen_cycles.add(key)
            unique_cycles.append(c)
    
    return unique_cycles


def _find_voltage_assignment(k6_edges, cycles, p, deadline):
    """
    Find voltage assignment on K6 edges such that no cycle has zero voltage sum mod p.
    Uses backtracking with forward checking.
    """
    # Precompute: for each edge, which cycles it belongs to
    edge_to_cycles = {}
    for i, c in enumerate(cycles):
        for edge in c:
            if edge not in edge_to_cycles:
                edge_to_cycles[edge] = []
            edge_to_cycles[edge].append(i)
    
    # Order edges by number of cycles they participate in (most constrained first)
    edge_order = sorted(k6_edges, key=lambda e: -len(edge_to_cycles.get(e, [])))
    
    # For each edge, precompute the cycles it's in and the other edges in those cycles
    edge_cycle_info = {}
    for edge in k6_edges:
        cyc_list = edge_to_cycles.get(edge, [])
        info = []
        for ci in cyc_list:
            c = cycles[ci]
            other_edges = [e for e in c if e != edge]
            info.append((ci, other_edges))
        edge_cycle_info[edge] = info
    
    # Backtracking search
    assignment = {}
    
    def backtrack(idx, current_time):
        if current_time > deadline - 2:
            return None
        
        if idx == len(edge_order):
            # All edges assigned, verify all cycles (should be satisfied by construction)
            return dict(assignment)
        
        edge = edge_order[idx]
        
        # Try each possible voltage value
        for v in range(p):
            assignment[edge] = v
            
            # Forward checking: for each cycle containing this edge,
            # check if the remaining edges can still make the sum non-zero
            valid = True
            for ci, other_edges in edge_cycle_info[edge]:
                # Current partial sum from assigned edges in this cycle
                partial_sum = v
                unassigned = []
                for oe in other_edges:
                    if oe in assignment:
                        partial_sum = (partial_sum + assignment[oe]) % p
                    else:
                        unassigned.append(oe)
                
                if not unassigned:
                    # All edges in this cycle are assigned
                    if partial_sum == 0:
                        valid = False
                        break
                else:
                    # Check if it's possible for the remaining edges to make sum non-zero
                    # The remaining edges can sum to any value in 0..p-1 (since they're independent)
                    # So this constraint is always satisfiable unless p==1 (not our case)
                    # But we can still do a quick check: if partial_sum is already 0 and 
                    # there's only one unassigned edge, it must be non-zero
                    if len(unassigned) == 1:
                        # The single unassigned edge must make the total non-zero
                        # partial_sum + x != 0 => x != -partial_sum
                        # This is always possible since p >= 2
                        pass
            
            if valid:
                result = backtrack(idx + 1, current_time)
                if result is not None:
                    return result
            
            del assignment[edge]
        
        return None
    
    return backtrack(0, time.time())


def _check_girth(edges, n, min_girth):
    """Check that the graph has girth at least min_girth using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True


def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph construction as fallback."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
        
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
        
    return len(order), edges
