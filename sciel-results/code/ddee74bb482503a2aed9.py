"""Voltage graph lift approach for the cage problem."""
import time
import random
from collections import deque

# Deadline guard
_START_TIME = None
_DEADLINE = 115  # seconds, safety margin before 120s limit

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return max(0, _DEADLINE - (time.time() - _START_TIME))

def construct(k, g):
    global _START_TIME
    _START_TIME = time.time()
    
    # Target: 3-regular graph with girth >= 16
    # Strategy: voltage lift of K_{3,3} over Z_m
    # K_{3,3} has 6 vertices, each of degree 3
    # Lift has 6*m vertices
    # 
    # K_{3,3} structure:
    #   Part A: 0, 1, 2
    #   Part B: 3, 4, 5
    #   Edges: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
    #
    # For girth 16, we need to avoid cycles shorter than 16.
    # In the base graph K_{3,3}, the girth is 4 (cycles like 0-3-1-4-0).
    # In the lift, a cycle of length L in the base lifts to cycles of length L*m
    # if the voltage sum is 0, or no closed cycle if voltage sum != 0.
    # 
    # Key insight: K_{3,3} has 4-cycles, 6-cycles, etc. We need all short cycles
    # to have non-zero voltage sum to eliminate them.
    #
    # For a 4-cycle like 0-3-1-4-0:
    #   voltage = v(0,3) + v(1,3) + v(1,4) + v(0,4) (with signs depending on direction)
    #   Actually, for undirected graph, we need to be careful with orientation.
    #
    # Let's orient all edges from Part A to Part B.
    # Edge (a,b) has voltage alpha[a][b] in Z_m.
    # A cycle 0->3->1->4->0: 
    #   0->3: +alpha[0][3]
    #   3->1: -alpha[1][3]  (going against orientation)
    #   1->4: +alpha[1][4]
    #   4->0: -alpha[0][4]
    #   Sum: alpha[0][3] - alpha[1][3] + alpha[1][4] - alpha[0][4]
    #
    # For this 4-cycle to not close: sum != 0 mod m.
    #
    # Let's use a specific structure. K_{3,3} has 9 edges.
    # We assign voltages alpha[i][j] for i in {0,1,2}, j in {3,4,5}.
    # Let's relabel: A={0,1,2}, B={0,1,2} (offset by 3)
    # alpha[i][j] for i,j in 0..2, edge between A_i and B_j.
    
    base_edges = [(0,0), (0,1), (0,2), 
                  (1,0), (1,1), (1,2),
                  (2,0), (2,1), (2,2)]
    
    # Moore bound for 3-regular, girth 16:
    # n >= 2 * (3 * (3^7 - 1)/(3-1)) = 2 * 3 * 1093 = 6558? 
    # Actually the Moore bound for girth g=2k is: n >= d * (d-1)^(k-1)
    # For d=3, k=8: n >= 3 * 2^7 = 384
    # For girth 16 = 2*8, k=8, n >= 3*2^7 = 384
    # Best known lower bound is 512, smallest known is 936.
    
    # We need 6*m >= 512, so m >= 86.
    # Let's try m values and see if we can find valid voltage assignments.
    
    # For each m, we need to find voltages such that all cycles of length < 16
    # in the base have non-zero voltage sum.
    # 
    # Cycles in K_{3,3}:
    # Length 4: There are C(3,2)*C(3,2) = 9 four-cycles
    # Length 6: More complex
    # 
    # Actually, for the lift to have girth >= 16, we need:
    # - No 4-cycles close (voltage sum != 0)
    # - No 6-cycles close
    # - No 8-cycles close
    # - No 10-cycles close
    # - No 12-cycles close
    # - No 14-cycles close
    #
    # This is a constraint satisfaction problem on the voltages.
    
    # Let's try different m values and use random search + local optimization.
    
    best_n = None
    best_edges = None
    
    # Try m from small to large
    for m in range(85, 200):
        if _time_left() < 10:
            break
        
        # Initialize voltages randomly
        voltages = [[random.randint(0, m-1) for _ in range(3)] for _ in range(3)]
        
        # Local search: try to fix violations
        best_voltages = None
        best_violations = None
        
        for trial in range(50):
            if _time_left() < 5:
                break
            
            # Evaluate current assignment
            violations = _count_violations(voltages, m, g)
            
            if best_violations is None or violations < best_violations:
                best_violations = violations
                best_voltages = [row[:] for row in voltages]
            
            if violations == 0:
                break
            
            # Mutate a random voltage
            i = random.randint(0, 2)
            j = random.randint(0, 2)
            old_val = voltages[i][j]
            voltages[i][j] = random.randint(0, m-1)
            
            new_violations = _count_violations(voltages, m, g)
            if new_violations > violations and random.random() > 0.1:
                voltages[i][j] = old_val  # reject
        
        if best_violations == 0:
            # Build the lift
            n = 6 * m
            edges = []
            for i in range(3):
                for j in range(3):
                    v = voltages[i][j]
                    for gval in range(m):
                        u_idx = i * m + gval
                        v_idx = (3 + j) * m + (gval + v) % m
                        edges.append((u_idx, v_idx))
            
            # Verify girth
            if _verify_girth(n, edges, g):
                return n, edges
    
    # Fallback: use LUW construction
    return _construct_luw(k, g)


def _count_violations(voltages, m, g):
    """Count number of short cycles (length < g) that close in the lift."""
    # In K_{3,3}, find all cycles of length < g
    # and check if their voltage sum is 0 mod m
    
    # For simplicity, check 4-cycles and 6-cycles
    # 4-cycles: choose 2 from A and 2 from B
    #   A: i1, i2; B: j1, j2
    #   Cycle: A_i1 -> B_j1 -> A_i2 -> B_j2 -> A_i1
    #   Voltage: v[i1][j1] - v[i2][j1] + v[i2][j2] - v[i1][j2]
    
    violations = 0
    for i1 in range(3):
        for i2 in range(i1+1, 3):
            for j1 in range(3):
                for j2 in range(j1+1, 3):
                    s = voltages[i1][j1] - voltages[i2][j1] + voltages[i2][j2] - voltages[i1][j2]
                    if s % m == 0:
                        violations += 1
    
    # 6-cycles: more complex, skip for now
    return violations


def _verify_girth(n, edges, g):
    """Verify that the graph has girth >= g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS to find shortest cycle
        dist = [-1] * n
        dist[start] = 0
        parent = [-1] * n
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] * 2 >= g:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _construct_luw(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph."""
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
