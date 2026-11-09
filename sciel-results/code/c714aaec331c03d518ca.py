import time
import random
import sys

N = 40
DEADLINE = 115  # Safety margin under 150s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    Path: u - a - b - c - d - v
    This is equivalent to checking if there exists a path of length 4 between u and v in the current graph.
    Specifically, we check if there is a path u-a-b-c-d-v.
    This can be rephrased: Is there a pair (a, d) such that a in N(u), d in N(v), and a and d have a common neighbor b?
    i.e., N(a) & N(d) != 0.
    Note: a != v, d != u, a != d (if a=d, path is u-a-a-d-v? No, simple path).
    The path u-a-b-c-d-v must have distinct vertices.
    So a != v, d != u, a != d.
    Also b != u, b != v, b != a, b != d.
    If N(a) & N(d) contains b, we must ensure b is not u, v, a, d.
    Since a is neighbor of u, b=a is not in N(a).
    Since d is neighbor of v, b=d is not in N(d).
    So b != a and b != d is automatically satisfied if b is in N(a) and N(d) and graph is simple?
    Wait, if b=a, then a is in N(a)? No.
    If b=d, then d is in N(d)? No.
    So b is distinct from a and d.
    We must ensure b != u and b != v.
    If b=u, then u is in N(a) and N(d). u is in N(a) by definition. Is u in N(d)?
    If u in N(d), then d-u is an edge.
    Path: u-a-...-d-u? No, path ends at v.
    The path is u-a-b-c-d-v.
    If b=u, path is u-a-u-c-d-v. Not simple.
    So we must exclude b=u and b=v.
    
    Algorithm:
    1. a_mask = N(u) & ~bit(v)
    2. d_mask = N(v) & ~bit(u)
    3. For each a in a_mask:
       For each d in d_mask:
          if a == d: continue
          common = N(a) & N(d)
          # Remove u and v from common
          common &= ~bit(u)
          common &= ~bit(v)
          if common != 0: return True
    4. Return False
    """
    nu = adj[u]
    nv = adj[v]
    
    a_mask = nu & ~(1 << v)
    if not a_mask:
        return False
        
    d_mask = nv & ~(1 << u)
    if not d_mask:
        return False
        
    # Iterate a
    am = a_mask
    while am:
        lsb_a = am & -am
        a = lsb_a.bit_length() - 1
        am ^= lsb_a
        
        na = adj[a]
        
        # Iterate d
        dm = d_mask
        while dm:
            lsb_d = dm & -dm
            d = lsb_d.bit_length() - 1
            dm ^= lsb_d
            
            if a == d:
                continue
                
            nd = adj[d]
            
            # Check intersection of N(a) and N(d), excluding u and v
            # We can optimize by pre-computing or just doing the AND
            common = na & nd
            # Exclude u and v
            if common & ~(1 << u) & ~(1 << v):
                return True
                
    return False

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += bin(adj[i]).count("1")
    return total // 2

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _make_all_edges():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _greedy_build(adj, order, deadline):
    """
    Adds edges from order if possible. Returns count added.
    """
    count = 0
    for u, v in order:
        if time.time() >= deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        count += 1
    return count

def _local_search(adj, deadline, rng):
    """
    Local search to maximize edges.
    Strategy:
    1. Try to add any valid edge.
    2. If stuck, try "Greedy Repair": Remove a random edge, then greedily add as many as possible.
       If the result is better, keep it.
    """
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    current_count = best_count
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Simple additions
        # Shuffle a subset or all? All is 780.
        # We can maintain a list of "missing" edges to speed up.
        # But for simplicity, we iterate all.
        missing = []
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1):
                missing.append((u, v))
                
        if missing:
            rng.shuffle(missing)
            for u, v in missing:
                if time.time() >= deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if _creates_c6(adj, u, v):
                    continue
                _add_edge(adj, u, v)
                current_count += 1
                improved = True
                # Update missing list? It's expensive. 
                # Just break and re-calculate missing in next loop?
                # Or just continue checking others.
                # Since we added one, others might become invalid or valid.
                # Let's just break and restart the loop to recalculate missing.
                break
        
        if not improved:
            # Pass 2: Greedy Repair
            # Pick a random existing edge to remove
            existing = []
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    existing.append((u, v))
            
            if not existing:
                break
                
            # Try a few random removals
            # Limit attempts to avoid wasting time
            attempts = 0
            max_attempts = 5
            while attempts < max_attempts and time.time() < deadline:
                attempts += 1
                u1, v1 = rng.choice(existing)
                
                # Save state
                adj_copy = adj[:]
                
                _remove_edge(adj, u1, v1)
                current_count -= 1
                
                # Greedily add edges
                # We can use the same missing list logic, but updated
                # For speed, just try all edges
                # This is O(E * C6_check). E=780.
                # If C6 check is fast, this is okay.
                
                added = 0
                # To be faster, we can prioritize edges that were previously blocked by u1-v1?
                # Hard to know. Just random order.
                order = all_edges[:]
                rng.shuffle(order)
                
                for u, v in order:
                    if time.time() >= deadline:
                        break
                    if (adj[u] >> v) & 1:
                        continue
                    if _creates_c6(adj, u, v):
                        continue
                    _add_edge(adj, u, v)
                    added += 1
                    current_count += 1
                    # Stop if we've recovered and gained?
                    # If added > 1, we are ahead.
                    # We can continue to add more.
                
                if current_count > best_count:
                    best_count = current_count
                    improved = True
                    # Keep the new state
                else:
                    # Revert
                    adj = adj_copy
                    current_count = best_count
                    
        if not improved:
            break
            
    return adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    all_edges = _make_all_edges()
    
    best_adj = [0] * N
    best_count = 0
    
    # Run multiple iterations
    while time.time() < deadline:
        # Phase 1: Greedy Build
        adj = [0] * N
        order = all_edges[:]
        rng.shuffle(order)
        
        # Randomized greedy
        _greedy_build(adj, order, deadline)
        
        current_count = _count_edges(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # Phase 2: Local Search
        if time.time() < deadline - 5: # Leave time for safety
            adj, current_count = _local_search(adj, deadline, rng)
            
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
                
        # Early exit if we found the known optimum?
        if best_count >= 105:
            break

    # Reconstruct edge list
    edges = []
    for i in range(N):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            m ^= lsb
    return edges

if __name__ == "__main__":
    construct()
