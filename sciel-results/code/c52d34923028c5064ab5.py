import time
import random
import numpy as np
from collections import deque
import itertools

def _get_cycles_K6(max_len=8):
    """Generate all simple cycles of length 3 to max_len in K6."""
    n = 6
    cycles = []
    # Use a simple DFS to find cycles
    # Since K6 is small, we can just iterate over permutations
    # But we need unique cycles.
    # A cycle is defined by its sequence of vertices, up to rotation and reversal.
    # Canonical form: start with smallest vertex, then next smallest among neighbors?
    # Simpler: just generate all and deduplicate.
    
    all_cycles = set()
    
    def dfs(start, current, path, visited):
        if len(path) >= 3 and current == start:
            # Found a cycle
            # Canonicalize
            # Rotate so smallest is first
            min_idx = path.index(min(path))
            canon = tuple(path[min_idx:] + path[:min_idx])
            # Check reversal
            rev = tuple(canon[::-1])
            if canon < rev:
                all_cycles.add(canon)
            else:
                all_cycles.add(rev)
            return
        
        if len(path) > max_len:
            return
            
        for nb in range(n):
            if nb == start and len(path) < 3:
                continue
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                dfs(start, nb, path, visited)
                path.pop()
                visited.remove(nb)
            elif nb == start and len(path) >= 3:
                # Close the cycle
                pass # Handled above? No, above checks current==start before adding.
                # Let's restructure.
    
    # Better approach for small graph:
    # Iterate over all subsets of vertices of size L
    # For each subset, check if it contains a cycle
    # This is complex.
    
    # Given K6 is so small, let's just hardcode the counts or use a library if available?
    # No, let's write a simple enumerator.
    
    # A cycle of length L in K6 corresponds to a permutation of L distinct vertices.
    # Total permutations for L=3: P(6,3) = 120.
    # For L=4: P(6,4) = 360.
    # For L=5: P(6,5) = 720.
    # For L=6: P(6,6) = 720.
    # Total is small.
    
    for L in range(3, max_len + 1):
        for perm in itertools.permutations(range(6), L):
            # Check if it's a cycle (first connected to last)
            # In K6, all are connected.
            # So any permutation is a cycle?
            # No, a cycle must return to start.
            # The permutation represents the path.
            # The cycle is perm[0] -> perm[1] -> ... -> perm[L-1] -> perm[0].
            # This is a valid cycle in K6.
            # We need to canonicalize to avoid duplicates.
            min_idx = perm.index(min(perm))
            canon = tuple(perm[min_idx:] + perm[:min_idx])
            rev = tuple(canon[::-1])
            if canon < rev:
                all_cycles.add(canon)
            else:
                all_cycles.add(rev)
                
    return list(all_cycles)

def _check_girth(adj, g):
    """Check if graph has girth at least g using non-backtracking walks."""
    n = len(adj)
    for start in range(n):
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        while queue:
            v, prev, depth = queue.popleft()
            if depth >= g:
                return False
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == start and depth + 1 < g:
                    return False
                if (nb, v, depth + 1) not in visited:
                    visited.add((nb, v, depth + 1))
                    queue.append((nb, v, depth + 1))
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k != 5 or g != 9:
        # Fallback to LUW for other parameters
        return _luw_construct(k, g)
        
    # Moore bound for k=5, g=9 is 426
    # Try voltage graph on K6
    cycles = _get_cycles_K6(8)
    
    # Edges of K6
    edges_k6 = [(i, j) for i in range(6) for j in range(i+1, 6)]
    
    for m in range(71, 200):
        if time.time() > deadline - 10:
            break
            
        # Try to find a valid voltage assignment
        found = False
        for trial in range(500):
            if time.time() > deadline - 5:
                break
                
            # Random voltages
            voltages = {}
            for u, v in edges_k6:
                voltages[(u, v)] = random.randint(1, m - 1) # Non-zero to avoid trivial 2-cycles?
                # Actually, 0 is allowed, but non-zero is better for girth
                # If voltage is 0, the edge is "vertical".
                # A cycle of length 2 in the lift would require 2 edges between same nodes.
                # K6 has no multi-edges.
                # So 0 is fine.
                # But let's avoid 0 to be safe?
                # If g_uv = 0, then (u, x) - (v, x).
                # If there is a triangle u-v-w-u with g_uv=0, g_vw=a, g_wu=-a.
                # Sum = 0. Cycle length 3.
                # So we need to check all cycles.
                pass
            
            # Check all cycles
            valid = True
            for cycle in cycles:
                # cycle is a tuple of vertices
                # Calculate voltage sum
                # The cycle is v0-v1-...-vL-1-v0
                # Voltages are directed.
                # We defined voltages for (u,v) with u<v.
                # If the edge in the cycle is (u,v) with u<v, voltage is voltages[(u,v)].
                # If the edge is (v,u) with v>u, voltage is -voltages[(u,v)].
                
                total = 0
                L = len(cycle)
                for i in range(L):
                    u = cycle[i]
                    v = cycle[(i+1) % L]
                    if u < v:
                        total += voltages[(u, v)]
                    else:
                        total -= voltages[(v, u)]
                
                if total % m == 0:
                    valid = False
                    break
            
            if valid:
                found = True
                break
        
        if found:
            # Construct the graph
            n = 6 * m
            adj = [[] for _ in range(n)]
            
            # Map (vertex, group_elem) to index
            # Index = vertex * m + group_elem
            
            for u in range(6):
                for v in range(u+1, 6):
                    g_uv = voltages[(u, v)]
                    for x in range(m):
                        # Edge from (u, x) to (v, x + g_uv)
                        idx_u = u * m + x
                        idx_v = v * m + (x + g_uv) % m
                        adj[idx_u].append(idx_v)
                        adj[idx_v].append(idx_u)
                        
            # Verify
            if all(len(a) == 5 for a in adj):
                # Check girth
                if _check_girth(adj, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
                    
    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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
