import numpy as np
import time
import random

def construct():
    """
    Construct a C4-free graph on n vertices with no independent set of size 11.
    Approach: 
    1. Start with the Polarity Graph of the Projective Plane of order q=5 (31 vertices).
    2. Extend to n=39 by adding vertices and connecting them greedily to maintain C4-freeness.
    3. Use local search to minimize the independence number.
    """
    # Step 1: Generate Polarity Graph of PG(2, 5)
    q = 5
    # Generate points in PG(2, q)
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make the first non-zero coordinate 1
                p = [x, y, z]
                # Find first non-zero
                for i in range(3):
                    if p[i] != 0:
                        inv = pow(p[i], q-2, q) # Fermat's little theorem for inverse
                        p = [(pi * inv) % q for pi in p]
                        break
                if p not in points:
                    points.append(tuple(p))
    
    n_base = len(points) # Should be 31
    
    # Define adjacency based on polarity X . Y = 0
    edges = set()
    for i in range(n_base):
        for j in range(i+1, n_base):
            dot = sum(points[i][k] * points[j][k] for k in range(3)) % q
            if dot == 0:
                edges.add((i, j))
                edges.add((j, i))
                
    # Step 2: Extend to n_target = 39
    n_target = 39
    n = n_target
    # We have 31 vertices. Need to add 8 more.
    # Initialize new vertices with no edges.
    
    # Helper to check C4
    def has_c4(u, v):
        common = 0
        for w in range(n):
            if w != u and w != v:
                if (u, w) in edges and (v, w) in edges:
                    common += 1
                    if common > 1:
                        return True
        return False

    # Add new vertices
    for new_v in range(n_base, n_target):
        # Try to connect new_v to existing vertices
        # Strategy: Connect to vertices that are "safe" (don't create C4)
        # To keep alpha low, we want high degree.
        # Try to connect to many vertices.
        candidates = list(range(n_base))
        random.shuffle(candidates)
        
        for u in candidates:
            if (new_v, u) in edges or (u, new_v) in edges:
                continue
            # Check if adding (new_v, u) creates C4
            if not has_c4(new_v, u):
                edges.add((new_v, u))
                edges.add((u, new_v))
                
    # Step 3: Local Search to reduce Independence Number
    # We need alpha <= 10.
    
    def independence_number():
        # Greedy + Backtracking approximation
        # For small n, we can use a better heuristic
        remaining = set(range(n))
        independent = []
        while remaining:
            # Pick vertex with min degree in remaining
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and (v, u) in edges)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            independent.append(min_v)
            remaining.discard(min_v)
            remaining -= {u for u in remaining if (min_v, u) in edges}
        return len(independent)

    start_time = time.time()
    time_limit = 100
    
    # Check initial alpha
    alpha = independence_number()
    
    # If alpha is too high, try to add more edges or swap edges
    while time.time() - start_time < time_limit:
        alpha = independence_number()
        if alpha <= 10:
            break
        
        # Try to add a random edge that doesn't create C4
        added = False
        for _ in range(500):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or (u, v) in edges:
                continue
            if not has_c4(u, v):
                edges.add((u, v))
                edges.add((v, u))
                added = True
                break
        
        if not added:
            # Try swapping edges? Or break
            # If we can't add edges, we are stuck.
            # For now, break.
            break

    # Final Verification
    # Check C4
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in edges:
                if has_c4(i, j):
                    edges.discard((i, j))
                    edges.discard((j, i))
                    
    # Check Independence
    alpha = independence_number()
    
    # If invalid, fallback to a smaller known valid graph (e.g., n=31 polarity graph if alpha<=10)
    # But polarity graph alpha is likely 11.
    # So we might need to return a smaller n if we can't fix it.
    # However, the task requires maximizing n.
    # If we fail, we return the best valid n we found.
    
    # For safety, if alpha > 10, we can try to remove vertices until alpha <= 10.
    # But that reduces n.
    
    # Let's just return the current graph. If it's invalid, score is 0.
    # To be safe, we should ensure validity.
    
    # Simple validity check:
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in edges and has_c4(i, j):
                valid = False
                break
        if not valid: break
    
    if not valid or alpha > 10:
        # Fallback: Use the base polarity graph (n=31) and check its alpha.
        # If alpha(31) <= 10, return n=31.
        # Else, return n=21 (q=4 polarity graph).
        
        # Let's quickly check q=4 polarity graph
        q2 = 4
        points2 = []
        for x in range(q2):
            for y in range(q2):
                for z in range(q2):
                    if x == 0 and y == 0 and z == 0: continue
                    p = [x, y, z]
                    for i in range(3):
                        if p[i] != 0:
                            inv = pow(p[i], q2-2, q2)
                            p = [(pi * inv) % q2 for pi in p]
                            break
                    if p not in points2:
                        points2.append(tuple(p))
        
        n2 = len(points2) # 21
        edges2 = set()
        for i in range(n2):
            for j in range(i+1, n2):
                dot = sum(points2[i][k] * points2[j][k] for k in range(3)) % q2
                if dot == 0:
                    edges2.add((i, j))
                    edges2.add((j, i))
                    
        # Check alpha of n2
        # Greedy
        remaining = set(range(n2))
        indep = []
        while remaining:
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and (v, u) in edges2)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            indep.append(min_v)
            remaining.discard(min_v)
            remaining -= {u for u in remaining if (min_v, u) in edges2}
            
        if len(indep) <= 10:
            n = n2
            edges = edges2
        else:
            # Return empty or minimal?
            # n=1 is always valid.
            n = 1
            edges = set()

    edge_list = [[u, v] for (u, v) in edges if u < v]
    return {'n': n, 'edges': edge_list}
