import time
import random

N = 40
DEADLINE = 105  # Safety margin

def _init_N2(adj):
    """Initialize N2 for all vertices."""
    N2 = [0] * N
    for u in range(N):
        mu = adj[u]
        while mu:
            lsb = mu & -mu
            v = lsb.bit_length() - 1
            mu ^= lsb
            N2[u] |= adj[v]
    return N2

def _add_edge(adj, N2, u, v):
    """Add edge (u,v) and update N2."""
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    
    # Update N2
    # For k in N(u), N2[k] |= N(v)
    mu = adj[u] & ~(1 << v) # Exclude v itself from neighbors for the loop? 
    # Actually N(u) now includes v.
    # The neighbors of u are the bits in adj[u].
    # We need to iterate over all k in N(u) BEFORE adding v? 
    # No, the change is that N(u) gained v. 
    # So for any k in N(u) (old), N(k) didn't change, but N2[k] = union of N(z) for z in N(k).
    # If k is in N(u), then u is in N(k). N(u) changed. So N2[k] must include the new part of N(u).
    # The new part is {v}. So we add N(v) to N2[k].
    
    # Also N2[u] = union of N(z) for z in N(u).
    # N(u) gained v. So N2[u] must include N(v).
    
    # So:
    # 1. For all k in N(u) (including v? No, k is neighbor of u. If k=v, N2[v] |= N(u).)
    #    Let's just iterate all bits in adj[u].
    mu = adj[u]
    while mu:
        lsb = mu & -mu
        k = lsb.bit_length() - 1
        mu ^= lsb
        N2[k] |= adj[v]
        
    mv = adj[v]
    while mv:
        lsb = mv & -mv
        k = lsb.bit_length() - 1
        mv ^= lsb
        N2[k] |= adj[u]

def _remove_edge(adj, N2, u, v):
    """Remove edge (u,v) and update N2. 
    Note: Dynamic removal of N2 is hard because of overlaps.
    We might need to recalculate N2 from scratch or use a counter.
    Given the complexity, maybe we only use addition in local search?
    Or we recalculate N2 after every removal? 
    Recalculation is O(N * E) ~ 40 * 100 = 4000 ops. 
    If we do 1000 swaps, that's 4M ops. Acceptable.
    """
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    # Recalculate N2
    for i in range(N):
        N2[i] = 0
        mi = adj[i]
        while mi:
            lsb = mi & -mi
            j = lsb.bit_length() - 1
            mi ^= lsb
            N2[i] |= adj[j]

def _creates_c6(adj, N2, u, v):
    """
    Check if adding (u,v) creates a C6.
    C6 exists iff there is a path u-a-b-c-d-v.
    This is iff exists a in N(u), d in N(v) such that exists b,c with a~b~c~d.
    a~b~c~d implies c in N(b) and c in N(d) and b in N(a).
    Equivalent to: N2[a] & N[d] != 0.
    """
    mu = adj[u]
    while mu:
        lsb = mu & -mu
        a = lsb.bit_length() - 1
        mu ^= lsb
        ma = N2[a]
        mv = adj[v]
        while mv:
            lsb_d = mv & -mv
            d = lsb_d.bit_length() - 1
            mv ^= lsb_d
            if ma & adj[d]:
                return True
    return False

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += adj[i].bit_count()
    return total // 2

def _greedy_fill(adj, N2, all_edges, rng):
    count = 0
    for u, v in all_edges:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, N2, u, v):
            continue
        _add_edge(adj, N2, u, v)
        count += 1
    return count

def _local_search(adj, N2, deadline, rng):
    """
    Local search: add any possible edge.
    If no edge can be added, try swaps.
    """
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding edges
        # Use a shuffled order to break symmetry? Or just iterate.
        # Iterating is deterministic. Let's shuffle for diversity in ILS restarts.
        shuffled = all_edges[:]
        rng.shuffle(shuffled)
        for u, v in shuffled:
            if time.time() >= deadline:
                return adj, N2
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, N2, u, v):
                continue
            _add_edge(adj, N2, u, v)
            improved = True
            # After adding, N2 is updated. 
            # We break out of the inner loop to restart the scan? 
            # Or continue? Continuing is fine, but the graph changed.
            # The current shuffled list might have edges that are now invalid or valid.
            # It's safer to restart the scan or just continue.
            # Let's continue.
        
        if not improved:
            # Try swaps
            present_edges = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            if not present_edges:
                break
            rng.shuffle(present_edges)
            swap_found = False
            for u1, v1 in present_edges:
                if time.time() >= deadline:
                    return adj, N2
                # Remove u1,v1
                _remove_edge(adj, N2, u1, v1)
                # Try to add any missing edge
                added = False
                for u2, v2 in all_edges:
                    if (adj[u2] >> v2) & 1:
                        continue
                    if _creates_c6(adj, N2, u2, v2):
                        continue
                    _add_edge(adj, N2, u2, v2)
                    added = True
                    break
                if added:
                    swap_found = True
                    break
                else:
                    # Revert
                    _add_edge(adj, N2, u1, v1)
            if not swap_found:
                break
    
    return adj, N2

def _ils_search(adj, N2, deadline, rng):
    """
    Iterated Local Search.
    """
    best_adj = adj[:]
    best_N2 = N2[:]
    best_count = _count_edges(adj)
    
    while time.time() < deadline:
        # Perturb
        cur_adj = best_adj[:]
        cur_N2 = best_N2[:]
        
        # Remove a few random edges
        present_edges = [(u, v) for u in range(N) for v in range(u + 1, N) if (cur_adj[u] >> v) & 1]
        if not present_edges:
            break
        num_remove = rng.randint(2, 5)
        for _ in range(num_remove):
            if not present_edges:
                break
            u, v = rng.choice(present_edges)
            _remove_edge(cur_adj, cur_N2, u, v)
            present_edges = [e for e in present_edges if e != (u, v)]
        
        # Local Search
        cur_adj, cur_N2 = _local_search(cur_adj, cur_N2, deadline, rng)
        cur_count = _count_edges(cur_adj)
        
        if cur_count > best_count:
            best_count = cur_count
            best_adj = cur_adj[:]
            best_N2 = cur_N2[:]
            
    return best_adj, best_N2

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = random.Random(42)
    adj = [0] * N
    N2 = _init_N2(adj)
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Initial Greedy
    rng.shuffle(all_edges)
    _greedy_fill(adj, N2, all_edges, rng)
    
    # Local Search
    adj, N2 = _local_search(adj, N2, deadline, rng)
    
    # ILS
    adj, N2 = _ils_search(adj, N2, deadline, rng)
    
    # Reconstruct edge list
    edges = []
    for i in range(N):
        m = adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            m ^= lsb
    return edges

if __name__ == "__main__":
    construct()
