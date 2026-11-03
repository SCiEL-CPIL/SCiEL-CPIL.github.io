import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # For k=4, g=10, try the 2-vertex voltage graph construction
    if k == 4 and g == 10:
        result = _voltage_lift_search(deadline)
        if result is not None:
            return result
            
    # Fallback to LUW if voltage search fails or times out
    result = _luw_construct(k, g)
    if result is not None:
        n, edges = result
        if _verify_girth(n, edges, g):
            return n, edges
            
    return _luw_construct(k, g)

def _voltage_lift_search(deadline):
    """
    Search for smallest M and voltages v1..v4 in Z_M such that:
    1. v1, v2, v3, v4 are distinct.
    2. No non-backtracking closed walk of length < 10 has sum 0.
    Uses NumPy for faster constraint checking.
    """
    # Prioritize M=160 as it corresponds to the known 320-vertex cage
    m_candidates = [160] + [m for m in range(4, 300) if m != 160]
    
    for M in m_candidates:
        if time.time() > deadline - 10:
            break
            
        # We need 4 distinct voltages.
        # Fix v1=0, v2=1. Search a, b in 2..M-1 with a < b.
        # To speed up, we can check constraints as we go.
        
        # Precompute indices for non-backtracking walks of length 4, 6, 8
        # For a walk i1, i2, ..., iL with i_t != i_{t+1}, the sum is 
        # v_i1 - v_i2 + v_i3 - ... (+/-) v_iL
        # We can represent each walk as a tuple of indices.
        
        walks_4 = []
        for i1 in range(4):
            for i2 in range(4):
                if i1 == i2: continue
                for i3 in range(4):
                    if i2 == i3: continue
                    for i4 in range(4):
                        if i3 == i4: continue
                        walks_4.append((i1, i2, i3, i4))
                        
        walks_6 = []
        for i1 in range(4):
            for i2 in range(4):
                if i1 == i2: continue
                for i3 in range(4):
                    if i2 == i3: continue
                    for i4 in range(4):
                        if i3 == i4: continue
                        for i5 in range(4):
                            if i4 == i5: continue
                            for i6 in range(4):
                                if i5 == i6: continue
                                walks_6.append((i1, i2, i3, i4, i5, i6))
                                
        walks_8 = []
        for i1 in range(4):
            for i2 in range(4):
                if i1 == i2: continue
                for i3 in range(4):
                    if i2 == i3: continue
                    for i4 in range(4):
                        if i3 == i4: continue
                        for i5 in range(4):
                            if i4 == i5: continue
                            for i6 in range(4):
                                if i5 == i6: continue
                                for i7 in range(4):
                                    if i6 == i7: continue
                                    for i8 in range(4):
                                        if i7 == i8: continue
                                        walks_8.append((i1, i2, i3, i4, i5, i6, i7, i8))

        # Convert to numpy arrays for vectorized operations
        # Signs: + - + - ...
        signs_4 = np.array([1, -1, 1, -1])
        signs_6 = np.array([1, -1, 1, -1, 1, -1])
        signs_8 = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        
        W4 = np.array(walks_4)
        W6 = np.array(walks_6)
        W8 = np.array(walks_8)
        
        for a in range(2, M):
            for b in range(a + 1, M):
                if time.time() > deadline - 5:
                    break
                    
                voltages = np.array([0, 1, a, b])
                
                # Check L=4
                # Sums = W4 @ voltages * signs_4 ? No, W4 contains indices.
                # We need voltages[W4] * signs_4
                sums_4 = (voltages[W4] * signs_4).sum(axis=1)
                if np.any(sums_4 % M == 0):
                    continue
                    
                # Check L=6
                sums_6 = (voltages[W6] * signs_6).sum(axis=1)
                if np.any(sums_6 % M == 0):
                    continue
                    
                # Check L=8
                sums_8 = (voltages[W8] * signs_8).sum(axis=1)
                if np.any(sums_8 % M == 0):
                    continue
                
                # Found valid M and voltages
                n = 2 * M
                edges = []
                for x in range(M):
                    for v in voltages:
                        y = (x + v) % M
                        u = x
                        w = M + y
                        edges.append((u, w))
                
                # Verify
                if _verify_girth(n, edges, 10):
                    return n, edges

    return None

def _verify_girth(n, edges, g):
    """Verify graph is 4-regular and has girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != 4:
            return False
            
    edge_set = set()
    for u, v in edges:
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        edge_set.add((u, v))
        
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] >= g:
                        break
                    queue.append(v)
                elif dist[v] != dist[u] + 1:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _luw_construct(k, g):
    """
    LUW construction fallback.
    """
    q = k
    d = (g + 1) // 2
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
