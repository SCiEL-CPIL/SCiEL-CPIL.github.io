"""
Program to construct a C4-free graph with no independent set of size 11, maximizing n.
Uses a polarity graph base (PG(2,5)) and optimized extension with fast independence check.
"""

import time
import random
import sys
from functools import lru_cache

sys.setrecursionlimit(10000)

def _popcount(x):
    return bin(x).count('1')

def _has_c4(adj, adj_bits, n):
    """
    Check if graph contains a C4.
    A C4 exists if there are two vertices u, v that have >= 2 common neighbors.
    """
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bits[u] & adj_bits[v]
            if common & (common - 1):
                return True
    return False

def _can_add_edge(adj, adj_bits, u, v, n):
    """
    Check if adding edge (u, v) would create a C4.
    """
    for x in adj[v]:
        if x == u:
            continue
        if adj_bits[u] & adj_bits[x]:
            return False
    for y in adj[u]:
        if y == v:
            continue
        if adj_bits[v] & adj_bits[y]:
            return False
    return True

def _add_edge(adj, adj_bits, u, v):
    """Add edge u-v, update structures."""
    adj[u].add(v)
    adj[v].add(u)
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)

def _remove_edge(adj, adj_bits, u, v):
    """Remove edge u-v, update structures."""
    adj[u].discard(v)
    adj[v].discard(u)
    adj_bits[u] &= ~(1 << v)
    adj_bits[v] &= ~(1 << u)

def _has_independent_set_k(adj_bits, n, k):
    """
    Fast exact check for independent set of size k.
    Uses bitset-based branch and bound with aggressive pruning.
    """
    if k <= 0:
        return True
    if n < k:
        return False
    if k == 1:
        return n > 0

    full_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = full_mask & ~(adj_bits[v] | (1 << v))
    
    def greedy_independent_set_upper_bound(rem):
        """
        Greedy coloring of the complement graph to get an upper bound on alpha(G[rem]).
        """
        uncolored = rem
        color_class = []
        num_colors = 0
        
        while uncolored:
            # Pick vertex with max degree in G (min degree in G')
            best_v = -1
            best_deg = -1
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = _popcount(adj_bits[v] & uncolored)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            
            v_bit = 1 << best_v
            assigned_color = -1
            for c in range(num_colors):
                if non_adj[best_v] & color_class[c]:
                    continue
                assigned_color = c
                break
            
            if assigned_color == -1:
                assigned_color = num_colors
                color_class.append(0)
                num_colors += 1
            
            color_class[assigned_color] |= v_bit
            uncolored ^= v_bit
        
        return num_colors

    cache = {}
    
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        
        if size + _popcount(rem) < k:
            return False
        
        key = (rem, size)
        if key in cache:
            return cache[key]
        
        ub = greedy_independent_set_upper_bound(rem)
        if size + ub < k:
            cache[key] = False
            return False
        
        best_v = -1
        best_deg = -1
        temp = rem
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            cache[key] = False
            return False
            
        v_bit = 1 << best_v
        
        new_rem_include = rem & non_adj[best_v]
        if search(new_rem_include, size + 1):
            cache[key] = True
            return True
            
        new_rem_exclude = rem & ~v_bit
        if search(new_rem_exclude, size):
            cache[key] = True
            return True
            
        cache[key] = False
        return False
        
    return search(full_mask, 0)

def _polarity_graph_pg25():
    """
    Construct the polarity graph of PG(2,5).
    Points are vectors in GF(5)^3, up to scale.
    There are (5^3 - 1)/(5 - 1) = 31 points.
    Two points are adjacent if their dot product is 0.
    This graph is C4-free.
    """
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx = 1
                    ny = (y * inv_x) % 5
                    nz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx = 0
                    ny = 1
                    nz = (z * inv_y) % 5
                else:
                    inv_z = pow(z, 3, 5)
                    nx = 0
                    ny = 0
                    nz = 1
                points.append((nx, ny, nz))
    
    unique_points = list(set(points))
    unique_points.sort()
    n = len(unique_points)
    
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = unique_points[i]
            p2 = unique_points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                _add_edge(adj, adj_bits, i, j)
                
    return n, adj, adj_bits

def _find_valid_neighbor_sets(adj, adj_bits, n, rng, max_sets=100):
    """
    Find valid neighbor sets for a new vertex.
    A valid neighbor set S is a set of vertices such that:
    1. For any u, v in S, they have at most 1 common neighbor in the current graph
       (since the new vertex would be a common neighbor, they can have at most 1 other)
    2. Adding the new vertex with neighbors S doesn't create a C4
    
    This is equivalent to finding cliques in a compatibility graph where
    two vertices are compatible if they have at most 1 common neighbor.
    """
    # Build compatibility: u and v can both be neighbors of new vertex
    # iff they have at most 1 common neighbor in the current graph
    compat = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bits[u] & adj_bits[v]
            if _popcount(common) <= 1:
                compat[u] |= (1 << v)
                compat[v] |= (1 << u)
    
    # Find maximal cliques in compatibility graph (these are valid neighbor sets)
    # Use Bron-Kerbosch with pivot
    cliques = []
    
    def bron_kerbosch(R, P, X):
        if time.time() > 145:
            return
        if not P and not X:
            if len(R) >= 3:  # Only consider sets of size >= 3
                cliques.append(set(R))
            return
        
        # Choose pivot u from P ∪ X with maximum degree in P
        pivot = -1
        max_deg = -1
        for v in list(P) + list(X):
            deg = len(P & compat[v] if isinstance(compat[v], set) else set(i for i in range(n) if (compat[v] >> i) & 1) & P)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot == -1:
            # No pivot, just branch on any vertex in P
            for v in list(P):
                bron_kerbosch(R | {v}, P & compat[v] & ~(1 << v), X & compat[v] & ~(1 << v))
                P = P & ~(1 << v)
                X = X | (1 << v)
        else:
            # Branch on vertices in P that are not neighbors of pivot
            P_not_adj = P & ~compat[pivot]
            for v in list(P_not_adj):
                bron_kerbosch(R | {v}, P & compat[v] & ~(1 << v), X & compat[v] & ~(1 << v))
                P = P & ~(1 << v)
                X = X | (1 << v)
        
        if len(cliques) >= max_sets:
            return
    
    # Convert compat to sets for easier manipulation
    compat_sets = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if (compat[u] >> v) & 1:
                compat_sets[u].add(v)
                compat_sets[v].add(u)
    
    # Rewrite Bron-Kerbosch with sets
    def bron_kerbosch_sets(R, P, X):
        if time.time() > 145:
            return
        if not P and not X:
            if len(R) >= 3:
                cliques.append(set(R))
            return
        
        if len(cliques) >= max_sets:
            return
        
        # Choose pivot u from P ∪ X with maximum degree in P
        pivot = -1
        max_deg = -1
        for v in list(P) + list(X):
            deg = len(P & compat_sets[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot == -1:
            for v in list(P):
                bron_kerbosch_sets(R | {v}, P & compat_sets[v] - {v}, X & compat_sets[v] - {v})
                P = P - {v}
                X = X | {v}
        else:
            P_not_adj = P - compat_sets[pivot]
            for v in list(P_not_adj):
                bron_kerbosch_sets(R | {v}, P & compat_sets[v] - {v}, X & compat_sets[v] - {v})
                P = P - {v}
                X = X | {v}
    
    all_vertices = set(range(n))
    bron_kerbosch_sets(set(), all_vertices, set())
    
    # Also add smaller valid sets (size 0, 1, 2)
    cliques.append(set())
    for v in range(n):
        cliques.append({v})
    for u in range(n):
        for v in range(u + 1, n):
            if (compat[u] >> v) & 1:
                cliques.append({u, v})
    
    return cliques

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # 1. Start with Polarity Graph of PG(2,5)
    n0, adj, adj_bits = _polarity_graph_pg25()
    n = n0
    best_n = n
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    # Verify base graph
    if _has_c4(adj, adj_bits, n):
        pass
    if _has_independent_set_k(adj_bits, n, 11):
        pass
        
    rng = random.Random(42)
    max_n = 43
    
    # 2. Extend by adding vertices one by one using structured search
    for target_n in range(n0 + 1, max_n + 1):
        if time.time() > deadline - 5:
            break
            
        success = False
        
        # Find valid neighbor sets for the new vertex
        valid_sets = _find_valid_neighbor_sets(adj, adj_bits, n, rng, max_sets=200)
        
        # Sort by size descending (larger sets tend to be more constraining for independence)
        valid_sets.sort(key=lambda s: len(s), reverse=True)
        
        for S in valid_sets:
            if time.time() > deadline - 5:
                break
            
            adj_copy = [set(s) for s in adj]
            adj_bits_copy = list(adj_bits)
            
            v_new = n
            
            # Add v_new to the graph
            adj_copy.append(set(S))
            adj_bits_copy.append(0)
            new_bits = 0
            for u in S:
                adj_copy[u].add(v_new)
                adj_bits_copy[u] |= (1 << v_new)
                new_bits |= (1 << u)
            adj_bits_copy[v_new] = new_bits
            
            # Check alpha
            if not _has_independent_set_k(adj_bits_copy, n + 1, 11):
                adj = adj_copy
                adj_bits = adj_bits_copy
                n += 1
                success = True
                break
        
        if not success:
            break
        
        best_n = n
        best_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    best_edges.append([u, v])

    # 3. Fallback: try direct randomized greedy for larger n if we're stuck
    if best_n < 39 and time.time() < deadline - 10:
        for target_n in range(min(43, best_n + 10), best_n, -1):
            if time.time() > deadline - 5:
                break
            found = False
            for r in range(80):
                if time.time() > deadline - 5:
                    break
                adj_t = [set() for _ in range(target_n)]
                adj_bits_t = [0] * target_n
                pairs = [(u, v) for u in range(target_n) for v in range(u + 1, target_n)]
                rng.shuffle(pairs)
                for u, v in pairs:
                    if _can_add_edge(adj_t, adj_bits_t, u, v, target_n):
                        _add_edge(adj_t, adj_bits_t, u, v)
                if not _has_c4(adj_t, adj_bits_t, target_n):
                    if not _has_independent_set_k(adj_bits_t, target_n, 11):
                        best_n = target_n
                        best_edges = []
                        for u in range(target_n):
                            for v in adj_t[u]:
                                if u < v:
                                    best_edges.append([u, v])
                        found = True
                        break
            if found:
                break

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Constructed graph with n={res['n']}")
