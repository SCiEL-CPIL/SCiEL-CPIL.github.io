import time
import random
import sys

N = 40
DEADLINE = 110  # Safety margin under 150s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    Path: u - a - b - c - d - v
    """
    nu = adj[u]
    nv = adj[v]

    # a in N(u), d in N(v)
    # a != v, d != u, a != d
    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        na = adj[a]
        
        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            if d == a:
                continue

            nd = adj[d]
            
            # We need b in N(a), c in N(d)
            # b != u, v, a, d
            # c != u, v, a, d
            # b != c
            # b ~ c
            
            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            
            b_mask = na & ~forbidden
            if not b_mask:
                continue
            
            c_mask = nd & ~forbidden
            if not c_mask:
                continue
            
            # Check if any b in b_mask is connected to any c in c_mask
            # Iterate over smaller set
            if b_mask.bit_count() < c_mask.bit_count():
                bm = b_mask
                while bm:
                    lsb_b = bm & -bm
                    b = lsb_b.bit_length() - 1
                    bm ^= lsb_b
                    if adj[b] & c_mask:
                        return True
            else:
                cm = c_mask
                while cm:
                    lsb_c = cm & -cm
                    c = lsb_c.bit_length() - 1
                    cm ^= lsb_c
                    if adj[c] & b_mask:
                        return True

    return False

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += adj[i].bit_count()
    return total // 2

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _make_all_edges():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _greedy_fill(adj, deadline):
    """
    Repeatedly try to add any valid edge until none found or deadline.
    Returns number of edges added.
    """
    all_edges = _make_all_edges()
    count = 0
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u, v in all_edges:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            count += 1
            changed = True
    return count

def _local_search(adj, deadline, rng):
    """
    Local search with random restarts/kicks.
    """
    current_count = _count_edges(adj)
    best_count = current_count
    best_adj = adj[:]
    
    all_edges = _make_all_edges()
    n_edges = len(all_edges)
    
    # Precompute existing edges for removal
    def get_existing_edges(adj):
        edges = []
        for u in range(N):
            m = adj[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append((u, v))
                m ^= lsb
        return edges

    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Try to add a random missing edge
        improved = False
        for _ in range(20): # Try a few random candidates
            u, v = rng.choice(all_edges)
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            current_count += 1
            improved = True
            break
        
        if improved:
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= max_no_improve:
            # Perform a kick: remove some edges and refill
            # Remove 2-5 random edges
            existing = get_existing_edges(adj)
            if existing:
                k = rng.randint(2, 5)
                removed = rng.sample(existing, min(k, len(existing)))
                for u, v in removed:
                    _remove_edge(adj, u, v)
                    current_count -= 1
                
                # Refill greedily
                _greedy_fill(adj, deadline)
                current_count = _count_edges(adj)
                
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                
                no_improve_count = 0
            else:
                break

    return best_adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    
    best_adj = [0] * N
    best_count = 0
    
    # Phase 1: Multiple random greedy starts
    # We can try different densities or orderings
    for trial in range(20):
        if time.time() >= deadline - 5:
            break
        adj = [0] * N
        # Random greedy: shuffle edges and add if valid
        edges = _make_all_edges()
        rng.shuffle(edges)
        for u, v in edges:
            if time.time() >= deadline - 5:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
        
        count = _count_edges(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        # Try local search from this start if it's promising
        if count > 80 and time.time() < deadline - 10:
            adj_ls, count_ls = _local_search(adj, deadline - 5, rng)
            if count_ls > best_count:
                best_count = count_ls
                best_adj = adj_ls[:]

    # Phase 2: Local search from the best found so far
    if time.time() < deadline - 5:
        adj_ls, count_ls = _local_search(best_adj, deadline - 1, rng)
        if count_ls > best_count:
            best_count = count_ls
            best_adj = adj_ls[:]

    # Final greedy polish
    if time.time() < deadline - 1:
        _greedy_fill(best_adj, deadline - 0.5)
        best_count = _count_edges(best_adj)
    
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
