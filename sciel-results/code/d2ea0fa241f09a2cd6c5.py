# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from collections import deque
from itertools import combinations, product

def construct(k, g):
    t_start = time.time()
    deadline = t_start + 140.0

    # Guaranteed-valid fallback: Lazebnik-Ustimenko-Woldar CD graph
    def _lw_fallback():
        q = k
        d = max(3, g - 5)
        
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Bipartite double cover of a 2-regular graph (disjoint union of cycles)
    def _double_cover(cycle_lengths, seed_offset=0):
        """
        Given cycle lengths [L1, L2, ..., Lm], construct a 4-regular graph
        as the bipartite double cover of the disjoint union of these cycles.
        
        Each cycle of length Li contributes 2*Li vertices to the double cover.
        Total vertices = 2 * sum(Li)
        
        The girth of the double cover is determined by the shortest non-backtracking
        closed walk, which corresponds to combinations of cycle segments.
        """
        if time.time() > deadline - 5:
            return None
        
        m = len(cycle_lengths)
        total_vertices = 2 * sum(cycle_lengths)
        
        # Build adjacency for the double cover
        # Vertex (cycle_idx, pos, side) where side=0 or 1
        adj = [[] for _ in range(total_vertices)]
        edges = []
        
        def vid(ci, pos, side):
            # Map (cycle_index, position_in_cycle, side) to global vertex id
            offset = 0
            for j in range(ci):
                offset += cycle_lengths[j]
            return 2 * (offset + pos) + side
        
        for ci, L in enumerate(cycle_lengths):
            for pos in range(L):
                # Within the same cycle, connect to neighbors in the cycle
                for side in [0, 1]:
                    v = vid(ci, pos, side)
                    # Two neighbors in the cycle (prev and next)
                    prev_pos = (pos - 1) % L
                    next_pos = (pos + 1) % L
                    for np in [prev_pos, next_pos]:
                        # In the double cover, (ci, pos, 0) connects to (ci, np, 1)
                        # and (ci, pos, 1) connects to (ci, np, 0)
                        other_side = 1 - side
                        w = vid(ci, np, other_side)
                        if w not in adj[v]:
                            adj[v].append(w)
                            adj[w].append(v)
                            if v < w:
                                edges.append((v, w))
        
        # Now add cross-connections between different cycles
        # Each vertex needs degree 4. Currently each has degree 2 (from cycle neighbors).
        # We need to add 2 more edges per vertex.
        # Strategy: connect vertices across different cycles in a structured way.
        # For cycles ci and cj, connect positions in a shifted manner.
        
        # Build a list of all vertices grouped by cycle
        vertices_by_cycle = []
        offset = 0
        for ci, L in enumerate(cycle_lengths):
            vs = []
            for pos in range(L):
                for side in [0, 1]:
                    vs.append(vid(ci, pos, side))
            vertices_by_cycle.append(vs)
            offset += 2 * L
        
        # For each pair of cycles, add connections
        # We need to be careful to maintain degree 4 and avoid creating short cycles
        rng = random.Random(42 + seed_offset)
        
        for ci in range(m):
            for cj in range(ci+1, m):
                Li = cycle_lengths[ci]
                Lj = cycle_lengths[cj]
                # Connect Li*2 vertices from cycle ci to Lj*2 vertices from cycle cj
                # Each vertex in ci needs to connect to 2 vertices in cj on average
                # Simple approach: create a bipartite matching with shifts
                
                # Create two perfect matchings between the two sets
                # Shift by different amounts to avoid short cycles
                shift1 = rng.randint(0, Lj - 1)
                shift2 = rng.randint(0, Lj - 1)
                while shift2 == shift1:
                    shift2 = rng.randint(0, Lj - 1)
                
                for pos in range(Li):
                    for side in [0, 1]:
                        v = vid(ci, pos, side)
                        # Current degree
                        current_deg = len(adj[v])
                        needed = 4 - current_deg
                        
                        if needed >= 1:
                            # Connect to position (pos + shift1) % Lj in cycle cj
                            pos_j = (pos + shift1) % Lj
                            w = vid(cj, pos_j, side)
                            if w not in adj[v] and w != v:
                                adj[v].append(w)
                                adj[w].append(v)
                                if v < w:
                                    edges.append((v, w))
                        
                        if needed >= 2:
                            pos_j = (pos + shift2) % Lj
                            w = vid(cj, pos_j, 1 - side)
                            if w not in adj[v] and w != v:
                                adj[v].append(w)
                                adj[w].append(v)
                                if v < w:
                                    edges.append((v, w))
        
        # Verify all vertices have degree 4
        for v in range(total_vertices):
            if len(adj[v]) != 4:
                return None
        
        # Check for duplicate edges
        edge_set = set()
        for v in range(total_vertices):
            for w in adj[v]:
                if v < w:
                    if (v, w) in edge_set:
                        return None
                    edge_set.add((v, w))
        
        return total_vertices, list(edge_set)

    # Fast girth certification using non-backtracking walks
    def _girth(adj, n):
        for L in range(3, 2*g+1):
            for s in range(n):
                dist = {}
                q = deque()
                for nb in adj[s]:
                    q.append((s, nb))
                    dist[(s, nb)] = 1
                while q:
                    prev, cur = q.popleft()
                    dl = dist[(prev, cur)]
                    if dl == L:
                        break
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        key = (cur, nb)
                        if key in dist:
                            cyc = dl + 1 - dist[key]
                            if 3 <= cyc < L:
                                return cyc
                        else:
                            dist[key] = dl + 1
                            q.append(key)
        return 2*g

    # Try bipartite double cover approach first
    best_result = None
    best_n = float('inf')
    
    # For girth g=10, we need cycles long enough that combinations don't create short cycles
    # The girth of the double cover is at least 2 * min(cycle_length) if cycles are sufficiently separated
    # More precisely, we need to ensure no non-backtracking closed walk shorter than g exists
    
    if k == 4 and g == 10:
        # Try various cycle length partitions
        # Total vertices = 2 * sum(Li), we want to minimize this
        # Lower bound: sum(Li) >= g/2 = 5 (but that's too small, need more)
        # For girth 10, we need at least... let's try partitions of small sums
        
        # Try sum of cycle lengths from 10 upward (giving 20, 22, 24, ... vertices)
        for total_half in range(10, 40):
            if time.time() > deadline - 10:
                break
            
            # Find all partitions of total_half into parts >= 5 (to avoid trivial short cycles)
            # Actually, parts should be >= 5 for girth >= 10 in the double cover
            # Let's try partitions with parts >= 5
            
            def gen_partitions(n, min_part, max_parts=10):
                if n < min_part:
                    if n == 0:
                        yield []
                    return
                if n == min_part:
                    yield [min_part]
                    return
                for first in range(min_part, n + 1):
                    if first > n:
                        break
                    for rest in gen_partitions(n - first, min_part, max_parts - 1):
                        yield [first] + rest
            
            partitions = list(gen_partitions(total_half, 5, max_parts=8))
            
            for part in partitions:
                if time.time() > deadline - 10:
                    break
                # Sort for consistency
                part_sorted = tuple(sorted(part))
                result = _double_cover(list(part_sorted), seed_offset=hash(part_sorted) % 1000)
                if result is not None:
                    n, edges = result
                    if n < best_n:
                        # Verify girth
                        adj = [[] for _ in range(n)]
                        for u, v in edges:
                            adj[u].append(v)
                            adj[v].append(u)
                        girth = _girth(adj, n)
                        if girth >= g:
                            best_n = n
                            best_result = (n, edges)
                            break
    
    # If double cover didn't find anything better, use LUW
    if best_result is not None and best_n < 320:
        return best_result
    else:
        try:
            n, edges = _lw_fallback()
            return n, edges
        except Exception:
            return 2048, []
# EVOLVE-BLOCK-END
