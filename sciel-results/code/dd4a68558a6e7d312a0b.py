"""Clique partitioning of dense random graphs via multi-start greedy clique extraction."""
import time
from random import Random


def _random_graph(n, p, seed):
    """Generate random graph using the same RNG as the verifier."""
    rng = Random(seed)
    adj = [0] * n  # bitmask adjacency
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj


def _expand_clique(rem, start, n, strategy="max", rng=None):
    """
    Expand a clique starting from `start` in the graph defined by `rem`.
    strategy: 'max' (greedy max degree in candidates), 'min' (greedy min degree), 'random' (random among top 3)
    """
    clique = [start]
    cand = rem[start]
    
    while cand:
        # Find best vertex in cand
        # We want to maximize the number of common neighbors with other candidates
        # This is equivalent to maximizing degree within cand
        
        best_u = -1
        best_score = -1
        second_best_u = -1
        second_best_score = -1
        third_best_u = -1
        third_best_score = -1
        
        temp = cand
        # Iterate through all candidates to find top 3
        while temp:
            low = temp & (-temp)
            u = low.bit_length() - 1
            # Score: number of neighbors of u that are also in cand
            score = (rem[u] & cand).bit_count()
            
            if score > best_score:
                third_best_u, third_best_score = second_best_u, second_best_score
                second_best_u, second_best_score = best_u, best_score
                best_u, best_score = u, score
            elif score > second_best_score:
                third_best_u, third_best_score = second_best_u, second_best_score
                second_best_u, second_best_score = u, score
            elif score > third_best_score:
                third_best_u, third_best_score = u, score
                
            temp ^= low
            
        if best_u == -1:
            break
            
        # Select based on strategy
        if strategy == "max":
            chosen = best_u
        elif strategy == "min":
            # Find min degree in cand? No, 'min' strategy usually means pick least connected to avoid getting stuck?
            # Actually, for clique size, picking max degree in candidates is standard.
            # Let's use 'random' as the variation.
            chosen = best_u
        elif strategy == "random":
            # Pick randomly from top 3
            options = []
            if best_u != -1: options.append(best_u)
            if second_best_u != -1 and second_best_u != best_u: options.append(second_best_u)
            if third_best_u != -1 and third_best_u != best_u and third_best_u != second_best_u: options.append(third_best_u)
            if rng:
                chosen = rng.choice(options)
            else:
                chosen = options[0]
        else:
            chosen = best_u
            
        clique.append(chosen)
        cand &= rem[chosen]
        
    return clique if len(clique) >= 2 else None


def _solve_instance(adj, n, deadline):
    """Find clique partition for a single graph. Returns list of cliques."""
    # Remaining edges bitmask per vertex
    rem = [adj[i] for i in range(n)]
    remaining_edges = sum(rem[i].bit_count() for i in range(n)) // 2
    cliques = []
    
    if remaining_edges == 0:
        return cliques
        
    # Use a local RNG for randomization in clique expansion
    local_rng = Random(42) 
    
    while remaining_edges > 0:
        # Deadline guard
        if time.time() > deadline:
            break
        
        # Find vertex with max degree in remaining graph
        max_deg = -1
        v = -1
        second_v = -1
        second_max_deg = -1
        for i in range(n):
            d = rem[i].bit_count()
            if d > max_deg:
                second_v, second_max_deg = v, max_deg
                max_deg = d
                v = i
            elif d > second_max_deg:
                second_v, second_max_deg = i, d
        
        if v == -1 or max_deg == 0:
            # Shouldn't happen if remaining_edges > 0, but safety
            break
        
        if max_deg == 1:
            # Just add the single edge
            low = rem[v] & (-rem[v])
            u = low.bit_length() - 1
            cliques.append([v, u])
            rem[v] &= ~(1 << u)
            rem[u] &= ~(1 << v)
            remaining_edges -= 1
            continue
        
        # Generate multiple candidate cliques
        candidates = []
        
        # 1. Greedy from v
        c1 = _expand_clique(rem, v, n, "max")
        if c1: candidates.append(c1)
        
        # 2. Greedy from second_v
        if second_v != -1 and second_max_deg > 0:
            c2 = _expand_clique(rem, second_v, n, "max")
            if c2: candidates.append(c2)
            
        # 3. Randomized greedy from v
        c3 = _expand_clique(rem, v, n, "random", local_rng)
        if c3: candidates.append(c3)
        
        # 4. Randomized greedy from v (another trial)
        c4 = _expand_clique(rem, v, n, "random", local_rng)
        if c4: candidates.append(c4)
        
        # 5. Greedy from a random neighbor of v
        if rem[v]:
            low = rem[v] & (-rem[v])
            # Pick a random neighbor
            # Count bits
            bits = rem[v].bit_count()
            if bits > 1:
                # Pick random bit
                r = local_rng.randint(0, bits - 1)
                temp = rem[v]
                for _ in range(r):
                    temp &= temp - 1
                u = temp.bit_length() - 1
                c5 = _expand_clique(rem, u, n, "max")
                if c5: candidates.append(c5)
                
        if not candidates:
            # Fallback: just add one edge
            low = rem[v] & (-rem[v])
            u = low.bit_length() - 1
            cliques.append([v, u])
            rem[v] &= ~(1 << u)
            rem[u] &= ~(1 << v)
            remaining_edges -= 1
            continue
        
        # Select the best candidate (max size)
        best_clique = max(candidates, key=len)
        m = len(best_clique)
        
        # Add the clique and remove all its edges
        cliques.append(best_clique)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = best_clique[i], best_clique[j]
                rem[a] &= ~(1 << b)
                rem[b] &= ~(1 << a)
        remaining_edges -= m * (m - 1) // 2
    
    # Handle any remaining edges (if deadline hit)
    if remaining_edges > 0:
        for i in range(n):
            if time.time() > deadline:
                break
            temp = rem[i]
            while temp:
                if time.time() > deadline:
                    break
                low = temp & (-temp)
                j = low.bit_length() - 1
                cliques.append([i, j])
                rem[i] &= ~(1 << j)
                rem[j] &= ~(1 << i)
                temp &= temp - 1
                # Clear bit in rem[i]
                rem[i] &= ~(1 << j)
    
    return cliques


def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Can't even generate; return empty (invalid but avoids crash)
            results.append([])
            continue
        
        adj = _random_graph(n, p, seed)
        cliques = _solve_instance(adj, n, deadline)
        results.append(cliques)
    
    return results
