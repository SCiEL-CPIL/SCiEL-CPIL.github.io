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

    # a in N(u), a != v
    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        # d in N(v), d not in {u, a}
        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)

            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Check if any b in b_mask is connected to any c in c_mask (b != c)
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                # c must be in c_mask and not b
                if adj[b] & (c_mask & ~(1 << b)):
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

def _try_add_any(adj, all_edges, deadline):
    """
    Tries to add any valid edge from the list. Returns True if added.
    """
    # Shuffle to avoid bias towards low indices
    random.shuffle(all_edges)
    for u, v in all_edges:
        if time.time() >= deadline:
            return False
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        return True
    return False

def _try_swap(adj, all_edges, deadline, rng):
    """
    Tries to remove one edge and add another.
    Returns True if a swap was performed (count unchanged).
    """
    # Get current edges
    current_edges = []
    for u, v in all_edges:
        if (adj[u] >> v) & 1:
            current_edges.append((u, v))
    
    if not current_edges:
        return False
        
    # Try a few random swaps
    for _ in range(5):
        if time.time() >= deadline:
            return False
        e_out = rng.choice(current_edges)
        e_in = rng.choice(all_edges)
        
        if e_in == e_out:
            continue
        if (adj[e_in[0]] >> e_in[1]) & 1:
            continue # e_in is already an edge
            
        # Temporarily remove e_out
        _remove_edge(adj, e_out[0], e_out[1])
        
        # Check if e_in is valid
        if _creates_c6(adj, e_in[0], e_in[1]):
            # Revert
            _add_edge(adj, e_out[0], e_out[1])
            continue
        else:
            # Add e_in
            _add_edge(adj, e_in[0], e_in[1])
            # Swap successful
            return True
            
    return False

def _try_break(adj, all_edges, deadline, rng, current_count):
    """
    Removes a random edge, then tries to greedily add as many as possible.
    Returns True if the resulting count is strictly greater than current_count.
    """
    current_edges = []
    for u, v in all_edges:
        if (adj[u] >> v) & 1:
            current_edges.append((u, v))
            
    if not current_edges:
        return False

    # Try a few random breaks
    for _ in range(3):
        if time.time() >= deadline:
            return False
            
        e_out = rng.choice(current_edges)
        
        # Save state
        adj_backup = adj[:]
        count_backup = current_count
        
        # Remove e_out
        _remove_edge(adj, e_out[0], e_out[1])
        
        # Greedy add
        added = 0
        # Use a limited greedy pass
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
            
        new_count = count_backup - 1 + added
        
        if new_count > current_count:
            # Accept the change
            return True
        else:
            # Revert
            for i in range(N):
                adj[i] = adj_backup[i]
                
    return False

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
        
        _greedy_build(adj, order, deadline)
        
        current_count = _count_edges(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # Phase 2: Local Search / SA
        if time.time() < deadline - 2:
            # Loop until stuck or time up
            while time.time() < deadline - 2:
                improved = False
                
                # 1. Try to add any edge
                if _try_add_any(adj, all_edges, deadline):
                    current_count += 1
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                    continue # Restart loop with new state
                
                # 2. Try a swap (neutral move to escape)
                if _try_swap(adj, all_edges, deadline, rng):
                    # Count is same, but structure changed
                    # We don't update best_count, but we continue search
                    # Sometimes a swap unlocks an add
                    continue
                
                # 3. Try a break (remove 1, add many)
                if _try_break(adj, all_edges, deadline, rng, current_count):
                    # _try_break updates adj in place if successful
                    current_count = _count_edges(adj)
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                    improved = True
                    continue
                
                # If none worked, we are stuck
                break
                
        # Early exit if we found a very high count?
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
