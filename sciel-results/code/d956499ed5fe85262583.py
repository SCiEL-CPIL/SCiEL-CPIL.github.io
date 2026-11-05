import time
import math
import random
from collections import deque

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _deadline_exceeded(limit=140):
    return _elapsed() > limit

def construct_lw(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def _verify_girth_fast(edges, n, g):
    """
    Fast girth verification using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, BFS on non-backtracking walks
    # State: (prev, curr)
    # We want to find if any cycle of length < g exists.
    # A cycle of length L corresponds to a non-backtracking walk of length L-1 
    # that ends at a neighbor of the start? 
    # Actually, standard check:
    # From start s, walk to neighbor u (len 1).
    # If we reach a state (u, v) where v is a neighbor of s and v != prev(u)?
    # Simpler:
    # BFS from s. dist[s][v] = shortest distance from s to v.
    # If there is an edge (u, v) such that dist[s][u] + dist[s][v] + 1 < g?
    # No, that's for bipartite or specific structures.
    # General check:
    # For each edge (u, v), check if there is a path from u to v of length < g-1 not using the edge (u,v).
    # This is O(E * BFS).
    # E ~ n*k/2. BFS ~ n*k. Total O(n^2 * k^2).
    # For n=1000, k=5: 1000^2 * 25 = 2.5 * 10^7. This is fast.
    
    # Even faster: 
    # For each vertex s, compute shortest paths to all other vertices.
    # Let d_s[v] be dist(s, v).
    # A cycle exists of length < g if there is an edge (u, v) such that 
    # d_s[u] + 1 + d_s[v] < g? No, that's just a path.
    # A cycle containing s of length L < g exists if there is a neighbor v of s 
    # such that dist(s, v) via path not starting with edge (s,v) is < g-1?
    # Actually, girth is the length of the shortest cycle.
    # For each vertex s, find the shortest cycle containing s.
    # This is equivalent to finding the shortest path between two neighbors of s, 
    # excluding the vertex s itself.
    # Let N(s) be neighbors of s.
    # For each pair (u, v) in N(s), find shortest path from u to v in G - {s}.
    # If length L < g-2, then cycle length L+2 < g.
    
    # Complexity:
    # n vertices.
    # For each s:
    #   Remove s.
    #   For each pair of neighbors (k choose 2 ~ 10 pairs):
    #     BFS from u to v.
    #     BFS cost: O(n*k).
    # Total: n * 10 * n * k = 10 * n^2 * k.
    # n=1000, k=5 => 5 * 10^7. Very fast.
    
    # Optimization: 
    # We can run a single multi-source BFS from all neighbors of s simultaneously?
    # No, we need distinct paths for each pair?
    # Actually, we just need to know if ANY pair has a short path.
    # We can run BFS from all neighbors of s at once, keeping track of which neighbor we came from?
    # State: (curr, source_neighbor).
    # If we reach a node from two different source neighbors, we found a cycle.
    # Cycle length = dist_from_u + dist_from_v + 2.
    # We want to minimize this.
    
    for s in range(n):
        neighbors = adj[s]
        if len(neighbors) < 2:
            continue
            
        # Multi-source BFS from neighbors of s
        # dist[v][i] = shortest distance from neighbors[i] to v
        # We only care if dist[v][i] + dist[v][j] + 2 < g for i != j
        
        # Since k is small (5), we can keep an array of distances for each neighbor source.
        # dist[v][i]
        # Initialize: dist[neighbors[i]][i] = 0, others inf.
        # Queue: (v, source_idx)
        
        # To save memory and time, we can just store for each node the best two sources and distances?
        # If a node is reached by source i with dist d_i and source j with dist d_j,
        # cycle length is d_i + d_j + 2.
        # We want to know if min_{i!=j} (d_i + d_j + 2) < g.
        
        # Let's store for each node:
        # best1_dist, best1_src
        # best2_dist, best2_src (where src != best1_src)
        
        best1_dist = [float('inf')] * n
        best1_src = [-1] * n
        best2_dist = [float('inf')] * n
        best2_src = [-1] * n
        
        queue = deque()
        
        for i, nb in enumerate(neighbors):
            best1_dist[nb] = 0
            best1_src[nb] = i
            queue.append((nb, i, 0))
            
        while queue:
            curr, src, d = queue.popleft()
            
            # If we found a cycle shorter than g, return False
            # We update bests when we pop? Or when we push?
            # Let's check when we pop.
            # If src == best1_src[curr], it might be the best.
            # If src == best2_src[curr], it might be the second best.
            
            # Actually, standard multi-source BFS with source tracking.
            # When we visit a node with source src and distance d:
            # If src == best1_src[curr]:
            #    if d < best1_dist[curr]: update (shouldn't happen in BFS)
            # elif src == best2_src[curr]:
            #    if d < best2_dist[curr]: update
            # else:
            #    if d < best1_dist[curr]:
            #        # shift best1 to best2
            #        best2_dist[curr] = best1_dist[curr]
            #        best2_src[curr] = best1_src[curr]
            #        best1_dist[curr] = d
            #        best1_src[curr] = src
            #    elif d < best2_dist[curr] and src != best1_src[curr]:
            #        best2_dist[curr] = d
            #        best2_src[curr] = src
            
            # We can check for cycle when we have two distinct sources reaching a node.
            # If best1_src[curr] != best2_src[curr] and best1_src[curr] != -1 and best2_src[curr] != -1:
            #    cycle_len = best1_dist[curr] + best2_dist[curr] + 2
            #    if cycle_len < g: return False
            
            # To be efficient, we should check as soon as we have two sources.
            
            # Let's implement the update logic carefully.
            # Note: We must not traverse back to s.
            # Since we start from neighbors of s, and s is not in the graph for this BFS (implicitly),
            # we just need to ensure we don't go through s.
            # But s is not a neighbor of any neighbor in the BFS tree unless there is a triangle?
            # If there is a triangle s-u-v-s, then u and v are neighbors of s and each other.
            # dist[u][v] = 1. Cycle len = 1+1+2 = 4.
            # This is handled correctly.
            
            for nxt in adj[curr]:
                if nxt == s:
                    continue
                    
                new_d = d + 1
                
                if src == best1_src[nxt]:
                    if new_d < best1_dist[nxt]:
                        best1_dist[nxt] = new_d
                        queue.append((nxt, src, new_d))
                elif src == best2_src[nxt]:
                    if new_d < best2_dist[nxt]:
                        best2_dist[nxt] = new_d
                        queue.append((nxt, src, new_d))
                else:
                    if new_d < best1_dist[nxt]:
                        # New best1
                        best2_dist[nxt] = best1_dist[nxt]
                        best2_src[nxt] = best1_src[nxt]
                        best1_dist[nxt] = new_d
                        best1_src[nxt] = src
                        
                        # Check cycle
                        if best2_src[nxt] != -1:
                            cyc = best1_dist[nxt] + best2_dist[nxt] + 2
                            if cyc < g:
                                return False
                            
                        queue.append((nxt, src, new_d))
                    elif new_d < best2_dist[nxt] and src != best1_src[nxt]:
                        # New best2
                        best2_dist[nxt] = new_d
                        best2_src[nxt] = src
                        
                        # Check cycle
                        if best1_src[nxt] != -1:
                            cyc = best1_dist[nxt] + best2_dist[nxt] + 2
                            if cyc < g:
                                return False
                            
                        queue.append((nxt, src, new_d))
                        
        # If we didn't return False, no short cycle through s.
        
    return True

def construct_abs_heisenberg(k, g):
    # Try Heisenberg group H_q construction for odd k
    if k != 5:
        return None
    
    # Estimate q
    # n = q^3
    # Moore bound for k=5, g=9 is 426
    # Try q such that q^3 is reasonable
    # Start from small even q
    candidates = [10, 12, 14, 16, 18, 20]
    
    for q in candidates:
        if _deadline_exceeded():
            return None
            
        n = q**3
        # Generators for k=5 on H_q (even q)
        gens = [
            (1, 0, 0),
            (q-1, 0, 0),
            (0, 1, 0),
            (0, q-1, 0),
            (0, 0, q//2)
        ]
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        
        def idx(x, y, z):
            return (x * q + y) * q + z
            
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    u = idx(x, y, z)
                    for gx, gy, gz in gens:
                        vx = (x + gx) % q
                        vy = (y + gy) % q
                        vz = (z + gz + x * gy) % q
                        v = idx(vx, vy, vz)
                        if u < v:
                            adj[u].append(v)
                            adj[v].append(u)
                            
        # Verify girth
        # Use the fast verifier
        # We need to convert adj to edges for the verifier?
        # Or modify verifier to take adj.
        # Let's just build edges for the verifier, it's cheap.
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
                    
        if _verify_girth_fast(edges, n, g):
            return n, edges
            
    return None

def construct(k, g):
    global _start_time
    _start_time = time.time()
    
    # Try Heisenberg for k=5
    if k == 5:
        result = construct_abs_heisenberg(k, g)
        if result is not None:
            return result
            
    # Fallback to LW
    return construct_lw(k, g)
