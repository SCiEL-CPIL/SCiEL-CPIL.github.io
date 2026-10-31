import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    deadline = time.time() + 110  # 10s safety margin

    # Fallback for general cases: LUW construction
    def luw_construct(k, g):
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

    # Specialized strategy for k=3, g=16
    if k == 3 and g == 16:
        # Petersen graph details
        # Vertices: 0-9
        # Edges:
        # Outer cycle: 0-1, 1-2, 2-3, 3-4, 4-0
        # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        n_base = 10
        m_base = 15

        def build_deterministic_lift(m, seed_offset=0):
            """
            Build a lift of the Petersen graph with cyclic group Z_m.
            Uses a deterministic algebraic voltage assignment.
            We try different "types" of voltage assignments.
            """
            # Type 1: Linear voltages v_i = (a_i * seed_offset + b_i) % m
            # We need to pick coefficients that likely avoid short cycles.
            # A common choice is to use distinct non-zero elements if m is prime,
            # or just spread them out.
            
            # Let's use a set of "good" multipliers for the 15 edges.
            # To ensure 3-regularity, we just assign a voltage to each edge.
            # The lift is simple if no two edges (u,v) and (u',v') create the same lifted edge
            # or loops. Since base is simple, loops only if u==v and vol==0 (base has no loops).
            # Multi-edges in lift: (u, l) -> (v, l+vol1) and (u', l') -> (v', l'+vol2).
            # If (u,v) != (u',v'), can they collide?
            # Only if u=u' and v=v' (same base edge) or u=v' and v=u' (opposite base edge).
            # Base is simple, so no opposite edges.
            # However, if (u,v) and (u,v) are the same edge, we have multiple lifted edges.
            # Wait, a lift of a simple graph is simple if the voltage assignment is "proper" in a sense?
            # Actually, a lift of a simple graph can have multi-edges if the voltage group is small.
            # Example: Edge (0,1) with vol 1. Edge (1,0) doesn't exist.
            # But what if (0,1) vol 1 and (0,1) vol 2? We only have one edge (0,1) in base.
            # So for each base edge, we have m lifted edges.
            # Can lifted edge from base edge e1 collide with lifted edge from base edge e2?
            # e1=(u1,v1), e2=(u2,v2).
            # Lifted e1: (u1, l) -> (v1, l+vol1)
            # Lifted e2: (u2, l') -> (v2, l'+vol2)
            # Collision if {u1, v1} = {u2, v2} and layers match.
            # Since base is simple, u1!=v1.
            # If e1 != e2, then {u1,v1} != {u2,v2}. So no collision between different base edges.
            # Within same base edge e=(u,v):
            # (u, l1) -> (v, l1+vol) and (u, l2) -> (v, l2+vol).
            # These are distinct edges unless l1=l2.
            # So the lift is ALWAYS simple for a simple base graph?
            # Yes, the lift of a simple graph is simple.
            # Wait, is it 3-regular? Yes.
            
            # So we just need to find m and voltages such that girth >= 16.
            
            # Deterministic voltage assignment strategy:
            # Use a primitive root or just distinct values spread out.
            # Let's try voltages: v_i = (i * P) % m for i=0..14, where P is a "step".
            # Or better: use a set of generators.
            
            # Let's parameterize by a list of 15 integers.
            # We will scan different patterns.
            
            voltages = []
            for i in range(m_base):
                # Use a pattern that depends on seed_offset to vary the assignment
                # e.g., v_i = (i + seed_offset) % m
                # Or v_i = (i * (seed_offset + 1) + i*i) % m
                vol = (i * (seed_offset + 1) + i * i + seed_offset) % m
                voltages.append(vol)
            
            edges = []
            for i, (u, v) in enumerate(base_edges):
                vol = voltages[i]
                for l in range(m):
                    u_lift = u * m + l
                    v_lift = v * m + (l + vol) % m
                    edges.append((u_lift, v_lift))
            
            n_total = n_base * m
            return n_total, edges

        def check_girth_bitset(adj, n, required_girth):
            """
            Fast girth check using BFS from each vertex.
            Returns True if girth >= required_girth.
            """
            max_depth = required_girth // 2
            
            # Precompute adjacency lists as lists of ints for speed
            adj_lists = [list(s) for s in adj]
            
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                
                while q:
                    u = q.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj_lists[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < required_girth:
                                return False
            return True

        # Strategy: Scan m (lift size) from small to large.
        # We want the smallest n = 10*m.
        # The Moore bound for k=3, g=16 is 512. So m >= 52.
        # Let's try m from 52 upwards.
        # For each m, try a few deterministic voltage patterns.
        
        best_n = float('inf')
        best_edges = None
        
        # We will iterate m.
        # To save time, we can skip m if 10*m >= best_n.
        
        # Let's try a range of m.
        # The known best is 936, so m=93.6 -> m=94 (n=940) or m=93 (n=930).
        # Let's scan m from 52 to 150.
        
        for m in range(52, 150):
            if time.time() > deadline - 5:
                break
            
            n_total = 10 * m
            if n_total >= best_n:
                continue
            
            # Try a few deterministic patterns for this m
            # Pattern 0: linear
            # Pattern 1: quadratic
            # Pattern 2: pseudo-random but deterministic seed
            
            for pattern_idx in range(3):
                if time.time() > deadline - 5:
                    break
                
                # Generate voltages based on pattern
                voltages = []
                for i in range(m_base):
                    if pattern_idx == 0:
                        vol = (i * 7 + 3) % m  # Fixed coefficients
                    elif pattern_idx == 1:
                        vol = (i * i + i + 1) % m
                    else:
                        # Use a simple hash-like function
                        vol = (i * 13 + 5 * i * i + pattern_idx) % m
                    voltages.append(vol)
                
                # Build graph
                edges = []
                for i, (u, v) in enumerate(base_edges):
                    vol = voltages[i]
                    for l in range(m):
                        u_lift = u * m + l
                        v_lift = v * m + (l + vol) % m
                        edges.append((u_lift, v_lift))
                
                # Check simplicity (should be simple, but verify just in case)
                # And 3-regularity
                adj = [set() for _ in range(n_total)]
                valid = True
                for u, v in edges:
                    if u == v:
                        valid = False
                        break
                    if v in adj[u]:
                        valid = False
                        break
                    adj[u].add(v)
                    adj[v].add(u)
                
                if not valid:
                    continue
                
                for i in range(n_total):
                    if len(adj[i]) != 3:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check girth
                if check_girth_bitset(adj, n_total, g):
                    best_n = n_total
                    best_edges = edges
                    break  # Found a valid graph for this m, go to next m (smaller)
            
            # If we found a graph for this m, we can break if we are scanning upwards?
            # No, we want the SMALLEST n. So we should scan m from small to large.
            # If we find a valid graph at m, it's the smallest for that pattern set.
            # But maybe a larger m has a smaller n? No, n=10m, so smaller m is better.
            # So if we find a valid graph at m, we can stop?
            # Not necessarily, because we are scanning m upwards.
            # If we find a valid graph at m=60, n=600.
            # We should keep looking for smaller m? No, we started at 52.
            # So if we find one at m=60, we should check m=52..59 first.
            # So we should scan m from 52 upwards, and the FIRST valid one is the best.
            
            if best_edges is not None and best_n == n_total:
                # We found a graph with this n. Since we are scanning m upwards,
                # this is the smallest n found so far.
                # We can break the m loop if we are confident?
                # Actually, we should continue to see if a smaller m works?
                # No, we are iterating m from 52 to 150.
                # If we find a valid graph at m=55, we should break because m=52..54 failed.
                # So yes, break the m loop.
                break
        
        if best_edges is not None:
            return best_n, best_edges
        
        # If deterministic lifts failed, try a few random lifts with fast check
        # This is a fallback to the original strategy but with faster check
        random.seed(42)
        for m in range(52, 120, 2):
            if time.time() > deadline - 10:
                break
            n_total = 10 * m
            if n_total >= best_n:
                continue
            
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                
                voltages = [random.randint(0, m - 1) for _ in range(m_base)]
                edges = []
                for i, (u, v) in enumerate(base_edges):
                    vol = voltages[i]
                    for l in range(m):
                        u_lift = u * m + l
                        v_lift = v * m + (l + vol) % m
                        edges.append((u_lift, v_lift))
                
                adj = [set() for _ in range(n_total)]
                valid = True
                for u, v in edges:
                    if u == v or v in adj[u]:
                        valid = False
                        break
                    adj[u].add(v)
                    adj[v].add(u)
                
                if not valid:
                    continue
                
                for i in range(n_total):
                    if len(adj[i]) != 3:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                if check_girth_bitset(adj, n_total, g):
                    best_n = n_total
                    best_edges = edges
                    break
            
            if best_edges is not None and best_n == n_total:
                break
        
        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback to LUW
        return luw_construct(k, g)

    # For other (k, g), use LUW
    return luw_construct(k, g)
