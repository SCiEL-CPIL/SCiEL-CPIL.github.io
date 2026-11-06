import time
import random

DEADLINE = 140.0

def _construct_andrasfai(k):
    """Construct the Andrasfai graph A_k on k^2 vertices."""
    n = k * k
    adj = [0] * n
    for i in range(n):
        x1 = i // k
        y1 = i % k
        for j in range(i + 1, n):
            x2 = j // k
            y2 = j % k
            if x1 != x2:
                # Check adjacency condition
                # y2 - y1 == x1 * x2 (mod k)
                if (y2 - y1) % k == (x1 * x2) % k:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    return adj, n

def _has_triangle_fast(adj, n):
    """Fast triangle check using bitsets."""
    for i in range(n):
        ai = adj[i]
        j = ai & ~((1 << (i + 1)) - 1)  # only j > i
        while j:
            lb = j & (-j)
            jj = lb.bit_length() - 1
            common = ai & adj[jj] & ~((1 << (jj + 1)) - 1)
            if common:
                return True
            j ^= lb
    return False

def _alpha_exact_bnb(adj, n, limit=10):
    """Exact max independent set size via branch and bound. Returns >= limit if alpha >= limit."""
    best = [0]
    
    def bnb(chosen_count, candidates):
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        # Ordering heuristic: pick vertex with max degree in candidates
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        # Branch: exclude v first
        bnb(chosen_count, rest)
        # Branch: include v
        new_cands = rest & ~adj[v]
        bnb(chosen_count + 1, new_cands)
    
    bnb(0, (1 << n) - 1)
    return best[0]

def _is_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle_fast(adj, n):
        return False
    if _alpha_exact_bnb(adj, n, limit=10) >= 10:
        return False
    return True

def construct():
    start_time = time.time()
    
    # Target n=40 using a subgraph of Andrasfai graph A_9
    # A_9 has 81 vertices, is triangle-free, and has small alpha.
    k = 9
    adj_full, n_full = _construct_andrasfai(k)
    
    # We want n=40. Take the first 40 vertices.
    n_target = 40
    if n_full >= n_target:
        # Create induced subgraph on vertices 0..39
        adj = [0] * n_target
        for i in range(n_target):
            # adj[i] in full graph, mask to only include vertices < n_target
            adj[i] = adj_full[i] & ((1 << n_target) - 1)
        
        # Verify validity (should be valid by construction properties, but good to check)
        if _is_valid(adj, n_target):
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
    
    # Fallback if something goes wrong (should not happen)
    # Try random construction for n=39
    for n_try in range(39, 25, -1):
        if time.time() > start_time + DEADLINE - 5:
            break
        for attempt in range(100):
            if time.time() > start_time + DEADLINE - 5:
                break
            # Simple random triangle-free generation
            random.seed(attempt * 1000 + n_try)
            adj = [0] * n_try
            order = list(range(n_try))
            random.shuffle(order)
            pairs = []
            for i in range(n_try):
                for j in range(i + 1, n_try):
                    pairs.append((i, j))
            random.shuffle(pairs)
            for i, j in pairs:
                common = adj[i] & adj[j]
                if common == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            
            if _is_valid(adj, n_try):
                edges = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n_try, 'edges': edges}
    
    return {'n': 0, 'edges': []}
