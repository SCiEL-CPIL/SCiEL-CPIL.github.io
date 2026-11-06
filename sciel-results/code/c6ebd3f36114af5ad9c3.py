import time
import sys
import math
from random import Random
from collections import deque

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict graph
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
            
        # Conflict graph: edge i conflicts with edge j if they share a vertex
        # or are joined by an edge
        conflict = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    conflict[i].add(j)
                    conflict[j].add(i)
                elif (c in adj_orig[a] or dd in adj_orig[a] or 
                      c in adj_orig[b] or dd in adj_orig[b]):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        # Convert to bitmask representation for fast operations
        # For m <= 64, we can use Python integers as bitmasks
        if m > 64:
            # For larger graphs, use list of sets but still try MIS
            # Fall back to a simpler approach
            color = [-1] * m
            remaining = set(range(m))
            K = 0
            while remaining:
                # Greedy independent set
                color_class = []
                avail = set(remaining)
                # Order by degree in remaining
                deg_in_rem = {v: len(conflict[v] & remaining) for v in remaining}
                for v in sorted(avail, key=lambda x: -deg_in_rem[x]):
                    if v in avail:
                        color_class.append(v)
                        avail -= conflict[v]
                        avail.discard(v)
                for v in color_class:
                    color[v] = K
                remaining -= set(color_class)
                K += 1
            return compress(color)
        
        # Bitmask-based MIS
        # conflict_mask[i] = bitmask of vertices conflicting with i
        conflict_mask = [0] * m
        for i in range(m):
            mask = 0
            for j in conflict[i]:
                mask |= (1 << j)
            conflict_mask[i] = mask
        
        all_mask = (1 << m) - 1
        
        def popcount(x):
            return bin(x).count('1')
        
        def branch_and_bound_mis(vertices_mask, time_limit):
            """Find a large independent set in the subgraph induced by vertices_mask.
            Returns (independent_set_mask, size)."""
            start_time = time.time()
            best_set = 0
            best_size = 0
            
            # Precompute: for each vertex, its non-neighbors within the current set
            # Use a simple branch and bound
            
            def bb(current_mask, current_set, current_size):
                nonlocal best_set, best_size
                if time.time() - start_time > time_limit:
                    return
                
                # Upper bound: current_size + popcount(current_mask)
                if current_size + popcount(current_mask) <= best_size:
                    return
                
                if current_mask == 0:
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set
                    return
                
                # Choose a vertex to branch on
                # Pick vertex with max degree in current subgraph
                vm = current_mask
                # Get list of vertices in current_mask
                verts = []
                temp = vm
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    temp ^= lsb
                
                if not verts:
                    return
                
                # Order by degree in current subgraph (descending)
                degs = []
                for v in verts:
                    # Count neighbors in current_mask
                    nb_mask = conflict_mask[v] & current_mask
                    degs.append((popcount(nb_mask), v))
                degs.sort(reverse=True)
                
                # Branch: include or exclude the highest-degree vertex
                v = degs[0][1]
                vbit = 1 << v
                
                # Include v: remove v and all its neighbors
                new_mask = current_mask & ~(conflict_mask[v] | vbit)
                bb(new_mask, current_set | vbit, current_size + 1)
                
                # Exclude v
                new_mask2 = current_mask & ~vbit
                bb(new_mask2, current_set, current_size)
            
            bb(vertices_mask, 0, 0)
            return best_set, best_size
        
        def greedy_is(vertices_mask):
            """Fast greedy independent set."""
            result = 0
            remaining = vertices_mask
            while remaining:
                # Pick vertex with min degree in remaining
                verts = []
                temp = remaining
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    temp ^= lsb
                
                if not verts:
                    break
                
                # Find vertex with minimum degree in remaining
                min_deg = float('inf')
                min_v = verts[0]
                for v in verts:
                    deg = popcount(conflict_mask[v] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                
                vbit = 1 << min_v
                result |= vbit
                remaining &= ~(conflict_mask[min_v] | vbit)
            
            return result, popcount(result)
        
        # Iteratively extract maximum independent sets
        color = [-1] * m
        remaining_mask = all_mask
        K = 0
        total_time = 0.0
        time_per_mis = 0.05  # Start with 50ms per MIS
        
        while remaining_mask and time.time() < deadline - 2.0:
            # First try greedy to get a lower bound
            g_set, g_size = greedy_is(remaining_mask)
            
            # Try branch-and-bound with limited time
            mis_set, mis_size = branch_and_bound_mis(remaining_mask, time_per_mis)
            
            if mis_size >= g_size:
                chosen = mis_set
            else:
                chosen = g_set
            
            # Assign color K to all edges in chosen
            temp = chosen
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                color[v] = K
                temp ^= lsb
            
            remaining_mask &= ~chosen
            K += 1
            
            # Adaptive time: if we found a large set, we can spend less time
            # If the set is small, try harder next time
            if mis_size > 10:
                time_per_mis = max(0.02, time_per_mis * 0.7)
            elif mis_size < 5:
                time_per_mis = min(0.2, time_per_mis * 1.3)
        
        # If there are still remaining edges, color them greedily
        if remaining_mask:
            temp = remaining_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                color[v] = K
                temp ^= lsb
            K += 1
        
        return compress(color)

    try:
        start = time.time()
        deadline = start + 145.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            if remaining < 5.0:
                for rem_idx in range(idx, len(specs)):
                    edges = instance(*specs[rem_idx])
                    adj = {}
                    for u, v in edges:
                        adj.setdefault(u, set()).add(v)
                        adj.setdefault(v, set()).add(u)
                    colour = {}
                    for e in edges:
                        near = {x for x in e} | {y for x in e for y in adj[x]}
                        taken = {c for f, c in colour.items() if set(f) & near}
                        colour[e] = next(c for c in range(len(edges)) if c not in taken)
                    out.append([colour[e] for e in edges])
                break
            weights = [40, 60, 80]
            total_weight = sum(weights[idx:])
            share = remaining * weights[idx] / total_weight
            share = min(share, 60.0)
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
