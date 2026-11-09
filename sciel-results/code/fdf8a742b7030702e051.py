"""C6-free graph on 40 vertices - maximize edges."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _has_c6(adj_mask, n, u, v):
    """Check if adding edge (u,v) to graph with adjacency masks adj_mask creates a C6.
    A C6 is u-a-b-c-d-v with all distinct vertices.
    """
    # For each neighbor a of u (a != v):
    #   For each neighbor b of a (b not in {u,v,a}):
    #     For each neighbor c of b (c not in {u,v,a,b}):
    #       For each neighbor d of c (d not in {u,v,a,b,c}):
    #         If d is neighbor of v -> C6 found
    #
    # Optimize: precompute masks
    mu = adj_mask[u]
    mv = adj_mask[v]
    
    # a: neighbor of u, a != v
    a_mask = mu & ~(1 << v)
    
    while a_mask:
        lsb = a_mask & -a_mask
        a = lsb.bit_length() - 1
        a_mask ^= lsb
        
        # b: neighbor of a, b not in {u, v, a}
        ma = adj_mask[a]
        b_mask = ma & ~(1 << u) & ~(1 << v) & ~(1 << a)
        
        while b_mask:
            lsb = b_mask & -b_mask
            b = lsb.bit_length() - 1
            b_mask ^= lsb
            
            # c: neighbor of b, c not in {u, v, a, b}
            mb = adj_mask[b]
            c_mask = mb & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
            
            while c_mask:
                lsb = c_mask & -c_mask
                c = lsb.bit_length() - 1
                c_mask ^= lsb
                
                # d: neighbor of c, d not in {u, v, a, b, c}
                mc = adj_mask[c]
                d_mask = mc & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                
                # Check if any d is neighbor of v
                if d_mask & mv:
                    return True
                    
    return False


def _greedy_build(n, rng, deadline):
    """Build a C6-free graph using a greedy approach with smart edge ordering."""
    adj = [set() for _ in range(n)]
    adj_mask = [0] * n
    edges = set()
    
    # Start with 8 disjoint K5s (each K5 has 10 edges, total 80)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
    
    # Generate all non-edge candidates
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                candidates.append((u, v))
    
    # Multiple passes with different orderings
    best_edges = edges.copy()
    best_adj = [s.copy() for s in adj]
    best_adj_mask = adj_mask.copy()
    
    for trial in range(3):
        if time.time() > deadline:
            break
        
        # Shuffle candidates
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        
        # Sort by degree sum (prefer low-degree vertices)
        shuffled.sort(key=lambda e: len(adj[e[0]]) + len(adj[e[1]]) + rng.random() * 0.5)
        
        cur_edges = edges.copy()
        cur_adj = [s.copy() for s in adj]
        cur_mask = adj_mask.copy()
        
        for u, v in shuffled:
            if time.time() > deadline:
                break
            if (u, v) in cur_edges:
                continue
            if not _has_c6(cur_mask, n, u, v):
                cur_edges.add((u, v))
                cur_adj[u].add(v)
                cur_adj[v].add(u)
                cur_mask[u] |= (1 << v)
                cur_mask[v] |= (1 << u)
        
        if len(cur_edges) > len(best_edges):
            best_edges = cur_edges
            best_adj = cur_adj
            best_adj_mask = cur_mask
    
    return best_edges, best_adj, best_adj_mask


def _local_search(edges, adj, adj_mask, n, deadline, rng):
    """Local search with 2-opt swaps: remove one edge, add another."""
    current_edges = edges.copy()
    current_adj = [s.copy() for s in adj]
    current_mask = adj_mask.copy()
    current_count = len(current_edges)
    
    best_edges = current_edges.copy()
    best_adj = [s.copy() for s in current_adj]
    best_mask = current_mask.copy()
    best_count = current_count
    
    # Precompute all possible edges
    all_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            all_pairs.append((u, v))
    
    edge_set = set(current_edges)
    
    no_improve = 0
    max_no_improve = 200
    
    # Build list of current edges for removal
    edge_list = list(current_edges)
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random edge to remove
        if not edge_list:
            break
        removed_idx = rng.randrange(len(edge_list))
        removed = edge_list[removed_idx]
        ru, rv = removed
        
        # Temporarily remove
        current_edges.discard(removed)
        current_adj[ru].discard(rv)
        current_adj[rv].discard(ru)
        current_mask[ru] &= ~(1 << rv)
        current_mask[rv] &= ~(1 << ru)
        
        # Try to find a valid edge to add
        added = None
        
        # Strategy 1: Try random non-edges
        for _ in range(80):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in current_edges:
                continue
            if not _has_c6(current_mask, n, u, v):
                added = (u, v)
                break
        
        # Strategy 2: If not found, try targeted search for low-degree pairs
        if added is None:
            for u in range(n):
                if len(current_adj[u]) > 12:
                    continue
                for v in range(u + 1, n):
                    if len(current_adj[v]) > 12:
                        continue
                    if (u, v) in current_edges:
                        continue
                    if not _has_c6(current_mask, n, u, v):
                        added = (u, v)
                        break
                if added is not None:
                    break
        
        if added is None:
            # Revert removal
            current_edges.add(removed)
            current_adj[ru].add(rv)
            current_adj[rv].add(ru)
            current_mask[ru] |= (1 << rv)
            current_mask[rv] |= (1 << ru)
            no_improve += 1
            continue
        
        au, av = added
        
        # Add the new edge
        current_edges.add(added)
        current_adj[au].add(av)
        current_adj[av].add(au)
        current_mask[au] |= (1 << av)
        current_mask[av] |= (1 << au)
        
        new_count = len(current_edges)
        
        if new_count > best_count:
            best_count = new_count
            best_edges = current_edges.copy()
            best_adj = [s.copy() for s in current_adj]
            best_mask = current_mask.copy()
            no_improve = 0
        else:
            # Revert: remove added, restore removed
            current_edges.discard(added)
            current_adj[au].discard(av)
            current_adj[av].discard(au)
            current_mask[au] &= ~(1 << av)
            current_mask[av] &= ~(1 << au)
            
            current_edges.add(removed)
            current_adj[ru].add(rv)
            current_adj[rv].add(ru)
            current_mask[ru] |= (1 << rv)
            current_mask[rv] |= (1 << ru)
            
            no_improve += 1
        
        # Update edge list
        edge_list = list(current_edges)
    
    return best_edges, best_adj, best_mask, best_count


def _verify_no_c6(edges, n):
    """Verify that the graph has no C6 by checking each edge's addition would create one."""
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    for u, v in edges:
        # Temporarily remove edge
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        if _has_c6(adj_mask, n, u, v):
            return False
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 40
    best_overall_count = 0
    best_overall_edges = set()
    
    # Phase 1: Multi-start greedy construction
    num_starts = 6
    time_per_phase = (deadline - start_time) / 3.0
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        seed = 42 + s * 7919
        rng = random.Random(seed)
        greedy_deadline = min(deadline - 60, start_time + time_per_phase * 0.5)
        
        edges, adj, adj_mask = _greedy_build(n, rng, greedy_deadline)
        
        if len(edges) > best_overall_count:
            best_overall_count = len(edges)
            best_overall_edges = edges.copy()
        
        # Phase 2: Local search on this construction
        if time.time() < deadline - 30:
            ls_deadline = min(deadline - 5, start_time + time_per_phase * 0.8)
            edges, adj, adj_mask, count = _local_search(edges, adj, adj_mask, n, ls_deadline, rng)
            
            if count > best_overall_count:
                best_overall_count = count
                best_overall_edges = edges.copy()
    
    # Phase 3: More local search on the best found so far
    if best_overall_count > 0 and time.time() < deadline - 10:
        adj = [set() for _ in range(n)]
        adj_mask = [0] * n
        for u, v in best_overall_edges:
            adj[u].add(v)
            adj[v].add(u)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        rng = random.Random(12345)
        edges, adj, adj_mask, count = _local_search(
            best_overall_edges, adj, adj_mask, n, deadline - 3, rng
        )
        
        if count > best_overall_count:
            best_overall_count = count
            best_overall_edges = edges.copy()
    
    # Final verification
    if best_overall_edges and not _verify_no_c6(best_overall_edges, n):
        # Fallback: just the K5s
        fallback = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    fallback.add((u, v))
        return [list(e) for e in fallback]
    
    return [list(e) for e in best_overall_edges]

# EVOLVE-BLOCK-END
