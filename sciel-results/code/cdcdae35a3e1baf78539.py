import numpy as np
import random
import time
import sys
from collections import defaultdict

def construct(k, g):
    deadline = time.time() + 110
    
    if k != 4 or g != 10:
        # Fallback for other parameters: LUW
        return _luw_fallback(k, g)

    # Construct PG(2,3) incidence graph
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0: continue
                if x != 0:
                    inv_x = 1 if x == 1 else 2
                    p = (1, (y*inv_x)%3, (z*inv_x)%3)
                elif y != 0:
                    inv_y = 1 if y == 1 else 2
                    p = (0, 1, (z*inv_y)%3)
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
    
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0: continue
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b*inv_a)%3, (c*inv_a)%3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c*inv_b)%3)
                else:
                    n = (0, 0, 1)
                if n in [tuple(l) for l in lines]: continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0]*p[0] + n[1]*p[1] + n[2]*p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)
    
    n_base = 26
    base_edges = []
    adj = {i: [] for i in range(n_base)}
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
                adj[i].append(13+j)
                adj[13+j].append(i)
                
    # Find all 6-cycles and 8-cycles
    # We need to identify cycles by the sequence of edges.
    # A cycle is a set of edges.
    # For the linear system, we need the incidence vector of the edges in the cycle.
    
    cycles = []
    
    # Helper to add a cycle (list of vertices) to the set of edge-sets
    # We store the sorted tuple of edge indices.
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    def get_edge_set(vertices):
        edges_in_cycle = set()
        for i in range(len(vertices)):
            u = vertices[i]
            v = vertices[(i+1) % len(vertices)]
            edges_in_cycle.add(edge_index[(u, v)])
        return tuple(sorted(edges_in_cycle))

    # Enumerate 6-cycles
    # p1 - l1 - p2 - l2 - p3 - l3 - p1
    seen_6 = set()
    for p1 in range(13):
        for l1 in adj[p1]:
            for p2 in adj[l1]:
                if p2 == p1: continue
                for l2 in adj[p2]:
                    if l2 == l1: continue
                    for p3 in adj[l2]:
                        if p3 in [p1, p2]: continue
                        for l3 in adj[p3]:
                            if l3 == l2: continue
                            if p1 in adj[l3]:
                                # Check if l3 is connected to p1
                                # We need to ensure the cycle is simple and closed
                                # p1-l1-p2-l2-p3-l3-p1
                                # Is l3 adjacent to p1? Yes, by check.
                                # Is this a valid 6-cycle?
                                # Vertices: p1, l1, p2, l2, p3, l3
                                # All distinct?
                                if len({p1, l1, p2, l2, p3, l3}) == 6:
                                    e_set = get_edge_set([p1, l1, p2, l2, p3, l3])
                                    if e_set not in seen_6:
                                        seen_6.add(e_set)
                                        cycles.append(e_set)

    # Enumerate 8-cycles
    # p1 - l1 - p2 - l2 - p3 - l3 - p4 - l4 - p1
    seen_8 = set()
    for p1 in range(13):
        for l1 in adj[p1]:
            for p2 in adj[l1]:
                if p2 == p1: continue
                for l2 in adj[p2]:
                    if l2 == l1: continue
                    for p3 in adj[l2]:
                        if p3 in [p1, p2]: continue
                        for l3 in adj[p3]:
                            if l3 == l2: continue
                            for p4 in adj[l3]:
                                if p4 in [p1, p2, p3]: continue
                                for l4 in adj[p4]:
                                    if l4 == l3: continue
                                    if p1 in adj[l4]:
                                        if len({p1, l1, p2, l2, p3, l3, p4, l4}) == 8:
                                            e_set = get_edge_set([p1, l1, p2, l2, p3, l3, p4, l4])
                                            if e_set not in seen_8:
                                                seen_8.add(e_set)
                                                cycles.append(e_set)
                                                
    # Build linear system over Z2
    # Variables: x_e for each edge e in base_edges
    # Equations: sum(x_e for e in cycle) = 1
    n_edges = len(base_edges)
    n_cycles = len(cycles)
    
    if n_cycles == 0:
        # No cycles to constrain? Unlikely.
        voltages = [0] * n_edges
    else:
        # Matrix A (n_cycles x n_edges), b (n_cycles)
        # Use Gaussian elimination over GF(2)
        # Represent rows as integers (bitmasks) for speed
        rows = []
        for cyc in cycles:
            mask = 0
            for e in cyc:
                mask |= (1 << e)
            rows.append(mask)
            
        # Augmented matrix: [A | b]
        # b is all 1s
        # We perform elimination
        # We want to find a solution.
        # If the system is consistent, we find one.
        
        # Standard GE over GF(2)
        # We'll keep the rows and track the pivot columns.
        # Since n_edges is 52, we can use Python integers.
        
        # Augment with RHS bit
        # If we just want a solution, we can do:
        # 1. Find rank of A.
        # 2. Check consistency.
        # 3. Back substitute.
        
        # Simpler: Iterate through all 2^free_vars? No, free vars might be large.
        # Let's just solve it.
        
        A = rows[:]
        b = [1] * n_cycles
        
        # GE
        rank = 0
        pivot_cols = []
        
        for col in range(n_edges):
            # Find pivot
            pivot_row = -1
            for r in range(rank, n_cycles):
                if (A[r] >> col) & 1:
                    pivot_row = r
                    break
            if pivot_row == -1:
                continue
                
            # Swap
            A[rank], A[pivot_row] = A[pivot_row], A[rank]
            b[rank], b[pivot_row] = b[pivot_row], b[rank]
            
            # Eliminate
            for r in range(n_cycles):
                if r != rank and ((A[r] >> col) & 1):
                    A[r] ^= A[rank]
                    b[r] ^= b[rank]
                    
            pivot_cols.append(col)
            rank += 1
            
        # Check consistency
        for r in range(rank, n_cycles):
            if A[r] == 0 and b[r] == 1:
                # Inconsistent
                # Try m=3
                break
        else:
            # Consistent
            # Find solution
            # Set free variables to 0
            x = [0] * n_edges
            for r in range(rank):
                # Find the pivot col for this row
                # It's the first set bit in A[r]
                # Actually, pivot_cols[r] is the col.
                # x[pivot_cols[r]] = b[r] XOR sum(x[other_cols])
                # Since we eliminated all other rows, A[r] has only the pivot bit?
                # No, we only eliminated other rows for the pivot col.
                # The row might have other bits.
                # But if we process in reverse, we can solve.
                pass
            
            # Back substitution
            # The matrix is in RREF?
            # In the loop above, we eliminated for ALL rows (r != rank).
            # So the pivot column has 1 only in the pivot row.
            # So x[pivot_col] = b[rank] XOR (sum of x[col] for other cols in A[rank])
            # Wait, if we eliminated all other rows, then A[rank] has 1 at pivot_col and potentially other 1s.
            # But the other rows have 0 at pivot_col.
            # So we can solve for x[pivot_col] if we know the other x's.
            # Let's set free variables to 0.
            
            pivot_set = set(pivot_cols)
            # Determine free variables
            # x = 0 for all
            # Then for each pivot row, x[pivot] = b[r] (since other x's are 0? No, other x's in the row might be free vars, which are 0).
            # So x[pivot] = b[r].
            
            for r in range(rank):
                col = pivot_cols[r]
                x[col] = b[r]
                
            voltages = x
    
    # If we broke out due to inconsistency, or if we want to try m=3
    # Let's check if voltages is defined and valid
    valid = True
    if 'voltages' not in locals():
        valid = False
    else:
        # Verify
        for cyc in cycles:
            s = 0
            for e in cyc:
                s += voltages[e]
            if s % 2 == 0:
                valid = False
                break
                
    if not valid:
        # Try m=3
        # System: sum = 1 mod 3
        # This is more complex.
        # For now, if m=2 fails, fall back to random search for m=10 or LUW.
        # But m=2 should work for PG(2,3) because the cycle space is well-behaved.
        
        # Let's try a simple heuristic for m=3
        # Or just use LUW fallback if m=2 fails.
        # Given time, let's assume m=2 works or use a small random search for m=3.
        
        # Actually, let's just use the LUW fallback if m=2 fails, to be safe.
        # But wait, if m=2 fails, it means the system is inconsistent.
        # Is it?
        # The sum of all 6-cycles incident to a vertex?
        # Let's assume it works. If not, the code will fall through.
        
        # If m=2 failed, try m=4?
        # No, let's just return the LUW result if m=2 fails.
        # But LUW is 2880.
        # Let's try to find a solution for m=2.
        
        # If inconsistent, try to relax? No, must be exact.
        
        # Let's try a random search for m=3 if m=2 fails.
        # But m=2 is very likely to work.
        
        if time.time() > deadline - 5:
             return _luw_fallback(k, g)
             
        # Try m=3 random
        for attempt in range(1000):
            random.seed(attempt)
            voltages3 = [random.randint(0, 2) for _ in range(n_edges)]
            ok = True
            for cyc in cycles:
                s = sum(voltages3[e] for e in cyc)
                if s % 3 == 0:
                    ok = False
                    break
            if ok:
                voltages = voltages3
                m = 3
                break
        else:
            return _luw_fallback(k, g)
            
    else:
        m = 2

    # Build lift
    n_vertices = n_base * m
    lift_edges = []
    for i, (u, v) in enumerate(base_edges):
        vu = voltages[i]
        for i_lift in range(m):
            lift_edges.append((u*m + i_lift, v*m + (i_lift + vu) % m))
            
    # Verify girth
    adj_lift = [[] for _ in range(n_vertices)]
    for u, v in lift_edges:
        adj_lift[u].append(v)
        adj_lift[v].append(u)
        
    min_girth = 999
    for start in range(n_vertices):
        dist = [-1] * n_vertices
        parent = [-1] * n_vertices
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for nb in adj_lift[cur]:
                if dist[nb] == -1:
                    dist[nb] = dist[cur] + 1
                    parent[nb] = cur
                    queue.append(nb)
                elif parent[nb] != cur and parent[cur] != nb:
                    cl = dist[nb] + dist[cur] + 1
                    if cl < min_girth:
                        min_girth = cl
    if min_girth >= g:
        return n_vertices, lift_edges
    else:
        return _luw_fallback(k, g)

def _luw_fallback(k, g):
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
