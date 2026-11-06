import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def popcount(x):
        return bin(x).count('1')

    def build_polarity_graph_q(q):
        """Construct the polarity graph of PG(2, q)."""
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for y in range(q):
            points.append((1, y, 0))
        points.append((0, 1, 0))
        
        n = len(points)
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            xi, yi, zi = points[i]
            for j in range(i+1, n):
                xj, yj, zj = points[j]
                if (xi*xj + yi*yj + zi*zj) % q == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return adj, points

    def find_independent_set_masked(adj, k, active_mask, time_limit=None):
        """Returns a list of vertices forming an independent set of size k, or None."""
        n = len(adj)
        if k <= 0:
            return []
        if popcount(active_mask) < k:
            return None
        
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            masks[i] = mask
        
        sys.setrecursionlimit(10000)
        
        def search(v, remaining, size, current_set):
            if time_limit is not None and time.time() > time_limit:
                return None
            if size == 0:
                return list(current_set)
            if v == n:
                return None
            if popcount(remaining) < size:
                return None
            
            # Try to include v
            if (remaining & (1 << v)) != 0:
                current_set.append(v)
                new_remaining = remaining & ~(masks[v] | (1 << v))
                res = search(v + 1, new_remaining, size - 1, current_set)
                if res is not None:
                    return res
                current_set.pop()
            
            # Try to exclude v
            res = search(v + 1, remaining, size, current_set)
            if res is not None:
                return res
            
            return None
        
        return search(0, active_mask, k, [])

    # Build PG(2, 7) graph
    adj_57, _ = build_polarity_graph_q(7)
    n = 57
    
    active = [True] * n
    active_mask = (1 << n) - 1
    
    # Iteratively remove vertices to break independent sets of size 11
    for _ in range(20):
        if time.time() > deadline - 15:
            break
            
        indep_set = find_independent_set_masked(adj_57, 11, active_mask, time_limit=deadline - 10)
        
        if indep_set is None:
            # No independent set found (or timeout)
            # Assume alpha < 11
            break
            
        # Remove a vertex from indep_set
        # Choose the one with the lowest degree in the active graph
        min_deg = float('inf')
        min_v = -1
        for v in indep_set:
            deg = 0
            for u in adj_57[v]:
                if active[u]:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_v = v
                
        if min_v != -1:
            active[min_v] = False
            active_mask &= ~(1 << min_v)
        else:
            break

    # Construct the final graph
    new_n = sum(active)
    mapping = {}
    new_idx = 0
    for i in range(n):
        if active[i]:
            mapping[i] = new_idx
            new_idx += 1
            
    edge_list = []
    for u in range(n):
        if not active[u]:
            continue
        for v in adj_57[u]:
            if active[v] and u < v:
                edge_list.append([mapping[u], mapping[v]])
                
    return {'n': new_n, 'edges': edge_list}

# EVOLVE-BLOCK-END
