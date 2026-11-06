import time
import random
import numpy as np
from itertools import combinations

def construct():
    # Define the two instances
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    total_start_time = time.time()
    total_time_limit = 150.0
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start_time
        remaining_time = total_time_limit - elapsed - 10.0
        
        if remaining_time < 1.0:
            cliques = trivial_partition(n, p, seed)
        else:
            max_time_for_this = 60.0 if idx == 0 else 80.0
            instance_time_limit = min(remaining_time, max_time_for_this)
            
            if idx == len(instances) - 1:
                instance_time_limit = remaining_time
            
            cliques = solve_instance(n, p, seed, instance_time_limit)
            
        results.append(cliques)
        
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def generate_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def solve_instance(n, p, seed, time_limit):
    edges = generate_edges(n, p, seed)
    if not edges:
        return []
        
    # Build adjacency as bitsets for fast operations
    res_adj = [0] * n  # bitmask of neighbors
    edgeset = set(edges)
    for u, v in edges:
        res_adj[u] |= (1 << v)
        res_adj[v] |= (1 << u)
        
    cliques = []
    start_time = time.time()
    
    # Phase 1: Greedy extraction of large cliques with randomization
    extraction_budget = time_limit * 0.65
    while time.time() - start_time < start_time + extraction_budget:
        if all(adj == 0 for adj in res_adj):
            break
            
        # Find a large clique using randomized greedy
        clique = find_large_clique_bitset(res_adj, n, random.Random(int(time.time() * 1000) % 10000))
        
        if not clique:
            # Find any edge
            found = False
            for u in range(n):
                if res_adj[u]:
                    v_bit = res_adj[u]
                    v = v_bit.bit_length() - 1
                    clique = [u, v]
                    found = True
                    break
            if not found:
                break
                
        # Remove edges in clique
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                res_adj[u] &= ~(1 << v)
                res_adj[v] &= ~(1 << u)
                
        cliques.append(clique)
        
    # Phase 2: Merge optimization
    merge_start = time.time()
    merge_budget = time_limit - (merge_start - start_time) - 2.0
    
    changed = True
    while changed and (time.time() - start_time < start_time + time_limit - 2.0):
        changed = False
        best_pair = None
        best_cross_edges = 0
        
        k = len(cliques)
        for i in range(k):
            c1 = cliques[i]
            if not c1:
                continue
            c1_set = set(c1)
            for j in range(i + 1, k):
                c2 = cliques[j]
                if not c2:
                    continue
                c2_set = set(c2)
                
                # Check if all cross-edges are in res_adj
                valid = True
                cross_count = 0
                for u in c1:
                    for v in c2:
                        if u == v:
                            continue
                        if not (res_adj[u] & (1 << v)):
                            valid = False
                            break
                        cross_count += 1
                    if not valid:
                        break
                    
                if valid and cross_count > 0:
                    if cross_count > best_cross_edges:
                        best_cross_edges = cross_count
                        best_pair = (i, j)
        
        if best_pair is not None:
            i, j = best_pair
            c1 = cliques[i]
            c2 = cliques[j]
            
            # Remove cross-edges from res_adj
            for u in c1:
                for v in c2:
                    if u == v:
                        continue
                    res_adj[u] &= ~(1 << v)
                    res_adj[v] &= ~(1 << u)
                    
            new_clique = list(set(c1) | set(c2))
            cliques[i] = new_clique
            cliques.pop(j)
            changed = True
    
    # Phase 3: Try to absorb remaining edges into existing cliques
    # This can reduce the number of cliques if a single edge can be added to an existing clique
    absorb_start = time.time()
    absorb_budget = time_limit - (absorb_start - start_time) - 1.0
    
    # First, collect all remaining edges
    remaining_edges = []
    for u in range(n):
        adj = res_adj[u]
        while adj:
            v_bit = adj & (-adj)
            v = v_bit.bit_length() - 1
            if u < v:
                remaining_edges.append((u, v))
            adj ^= v_bit
    
    # Try to merge each remaining edge into an existing clique
    for u, v in remaining_edges:
        if time.time() - start_time > start_time + time_limit - 1.0:
            break
        if not (res_adj[u] & (1 << v)):
            continue  # Already covered
        
        # Try to find a clique that can absorb this edge
        best_clique_idx = -1
        best_clique_size = 0
        for idx, clique in enumerate(cliques):
            if not clique:
                continue
            if u in clique and v not in clique:
                # Check if v connects to all in clique
                can_absorb = True
                for w in clique:
                    if w != v and not (res_adj[v] & (1 << w)):
                        can_absorb = False
                        break
                if can_absorb:
                    if len(clique) > best_clique_size:
                        best_clique_size = len(clique)
                        best_clique_idx = idx
            elif v in clique and u not in clique:
                # Check if u connects to all in clique
                can_absorb = True
                for w in clique:
                    if w != u and not (res_adj[u] & (1 << w)):
                        can_absorb = False
                        break
                if can_absorb:
                    if len(clique) > best_clique_size:
                        best_clique_size = len(clique)
                        best_clique_idx = idx
        
        if best_clique_idx >= 0:
            # Absorb the edge
            clique = cliques[best_clique_idx]
            if u not in clique:
                clique.append(u)
                for w in clique:
                    if w != u:
                        res_adj[u] &= ~(1 << w)
                        res_adj[w] &= ~(1 << u)
            if v not in clique:
                clique.append(v)
                for w in clique:
                    if w != v:
                        res_adj[v] &= ~(1 << w)
                        res_adj[w] &= ~(1 << v)
            # Remove the edge from residual
            res_adj[u] &= ~(1 << v)
            res_adj[v] &= ~(1 << u)
    
    # Final: add any truly remaining edges as single-edge cliques
    remaining_edges = []
    for u in range(n):
        adj = res_adj[u]
        while adj:
            v_bit = adj & (-adj)
            v = v_bit.bit_length() - 1
            if u < v:
                remaining_edges.append([u, v])
            adj ^= v_bit
    
    if remaining_edges:
        cliques.extend(remaining_edges)
        
    return cliques

def find_large_clique_bitset(res_adj, n, rng):
    # Use bitset-based greedy clique finding with randomization
    degrees = [bin(adj).count('1') for adj in res_adj]
    
    # Sort by degree, but with some randomness
    sorted_vertices = list(range(n))
    rng.shuffle(sorted_vertices)
    sorted_vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    best_clique = []
    
    # Try multiple starting points
    num_starts = min(30, n)
    for start_idx in range(num_starts):
        if start_idx < len(sorted_vertices):
            start_v = sorted_vertices[start_idx]
        else:
            start_v = rng.randint(0, n - 1)
            
        if degrees[start_v] == 0:
            continue
            
        # Greedy expansion
        C = [start_v]
        P = res_adj[start_v]  # bitmask of candidates
        
        while P:
            # Pick vertex in P with max degree in P
            best_u = -1
            best_deg = -1
            temp_p = P
            while temp_p:
                v_bit = temp_p & (-temp_p)
                u = v_bit.bit_length() - 1
                deg = bin(P & res_adj[u]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_u = u
                temp_p ^= v_bit
            
            if best_u == -1:
                break
            C.append(best_u)
            P &= res_adj[best_u]
            
        if len(C) > len(best_clique):
            best_clique = C
            
    return best_clique
