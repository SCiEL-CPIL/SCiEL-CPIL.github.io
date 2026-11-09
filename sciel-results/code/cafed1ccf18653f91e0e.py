"""Rewritten program for C6-free graph on 40 vertices maximizing edges."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _creates_c6_bitset(adj_bits, n, u, v):
    """Check if adding edge (u,v) creates a C6.
    
    A C6 through (u,v) means there exists a path u-a-b-c-d-v of length 5
    with all vertices distinct.
    
    We check: exists a in N(u)\{v}, b in N(a)\{u,v}, c in N(b)\{u,v,a},
    d in N(c)\{u,v,a,b} such that d in N(v)\{u}.
    
    Using bitsets for speed.
    """
    nu = adj_bits[u]
    nv = adj_bits[v]
    
    # a must be in N(u), not v
    a_mask = nu & ~(1 << v)
    
    while a_mask:
        a_bit = a_mask & (-a_mask)
        a = a_bit.bit_length() - 1
        a_mask ^= a_bit
        
        na = adj_bits[a]
        # b must be in N(a), not u, v
        b_mask = na & ~(1 << u) & ~(1 << v)
        
        while b_mask:
            b_bit = b_mask & (-b_mask)
            b = b_bit.bit_length() - 1
            b_mask ^= b_bit
            
            nb = adj_bits[b]
            # c must be in N(b), not u, v, a
            c_mask = nb & ~(1 << u) & ~(1 << v) & ~(1 << a)
            
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                
                nc = adj_bits[c]
                # d must be in N(c), not u, v, a, b, and d must be in N(v)
                # d in N(c) \ {u,v,a,b} ∩ N(v) \ {u}
                d_candidates = nc & nv & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                if d_candidates:
                    return True
    
    return False


def _verify_c6_free(edges, n):
    """Verify that the graph with given edges contains no C6."""
    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj_bits[u] & (1 << v):
                # Temporarily remove edge
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                if _creates_c6_bitset(adj_bits, n, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    return False
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    return True


def _greedy_build(n, rng, deadline):
    """Build a C6-free graph greedily."""
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    edges = set()
    
    # Start with 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    
    # Generate candidate edges sorted by degree sum
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                candidates.append((u, v))
    
    # Multiple passes
    progress = True
    while progress and time.time() < deadline:
        progress = False
        candidates.sort(key=lambda uv: len(adj[uv[0]]) + len(adj[uv[1]]))
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            if (u, v) in edges:
                continue
            if not _creates_c6_bitset(adj_bits, n, u, v):
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                progress = True
    
    return edges, adj, adj_bits


def _local_search(edges, adj, adj_bits, n, deadline, rng):
    """Local search with pure adds and swap moves."""
    current_count = len(edges)
    best_count = current_count
    best_edges = edges.copy()
    
    no_improve = 0
    max_no_improve = 300
    
    edge_list = list(edges)
    all_non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edges]
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try pure adds first
        added = None
        for _ in range(30):
            if not all_non_edges:
                break
            idx = rng.randrange(len(all_non_edges))
            u, v = all_non_edges[idx]
            if not _creates_c6_bitset(adj_bits, n, u, v):
                added = (u, v)
                break
        
        if added is not None:
            au, av = added
            for i, e in enumerate(all_non_edges):
                if e == added:
                    all_non_edges.pop(i)
                    break
            edges.add(added)
            adj[au].add(av)
            adj[av].add(au)
            adj_bits[au] |= (1 << av)
            adj_bits[av] |= (1 << au)
            edge_list.append(added)
            no_improve = 0
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges.copy()
            continue
        
        # Try swaps
        if not edge_list:
            break
        removed_idx = rng.randrange(len(edge_list))
        removed = edge_list[removed_idx]
        ru, rv = removed
        
        found = False
        for _ in range(50):
            if not all_non_edges:
                break
            idx = rng.randrange(len(all_non_edges))
            u, v = all_non_edges[idx]
            if (u, v) == removed:
                continue
            # Check if we can add (u,v) after removing (ru,rv)
            adj_bits[ru] &= ~(1 << rv)
            adj_bits[rv] &= ~(1 << ru)
            ok = not _creates_c6_bitset(adj_bits, n, u, v)
            adj_bits[ru] |= (1 << rv)
            adj_bits[rv] |= (1 << ru)
            
            if ok:
                # Perform swap
                edges.discard(removed)
                adj[ru].discard(rv)
                adj[rv].discard(ru)
                adj_bits[ru] &= ~(1 << rv)
                adj_bits[rv] &= ~(1 << ru)
                
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
                edge_list[removed_idx] = (u, v)
                
                for i, e in enumerate(all_non_edges):
                    if e == (u, v):
                        all_non_edges.pop(i)
                        break
                all_non_edges.append(removed)
                
                found = True
                no_improve = 0
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = edges.copy()
                break
        
        if not found:
            no_improve += 1
    
    return best_edges, best_count


def _spectral_perturb(base_adj, n, rng, deadline):
    """Apply small random perturbations to a base adjacency matrix."""
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    edges = set()
    
    # Copy base
    for u in range(n):
        for v in base_adj[u]:
            if v > u:
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    
    # Try random edge swaps
    edge_list = list(edges)
    all_non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edges]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for _ in range(200):
            if time.time() > deadline:
                break
            if not edge_list or not all_non_edges:
                break
            ri = rng.randrange(len(edge_list))
            removed = edge_list[ri]
            ru, rv = removed
            ni = rng.randrange(len(all_non_edges))
            u, v = all_non_edges[ni]
            if (u, v) == removed:
                continue
            
            # Try swap
            adj_bits[ru] &= ~(1 << rv)
            adj_bits[rv] &= ~(1 << ru)
            ok = not _creates_c6_bitset(adj_bits, n, u, v)
            adj_bits[ru] |= (1 << rv)
            adj_bits[rv] |= (1 << ru)
            
            if ok:
                edges.discard(removed)
                adj[ru].discard(rv)
                adj[rv].discard(ru)
                adj_bits[ru] &= ~(1 << rv)
                adj_bits[rv] &= ~(1 << ru)
                
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
                edge_list[ri] = (u, v)
                all_non_edges[ni] = removed
                improved = True
    
    return edges, adj, adj_bits


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    n = 40
    best_overall = 0
    best_edges_overall = set()
    
    # Multi-start
    num_starts = 8
    time_per_start = (deadline - start_time) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        seed = 12345 + s * 7919
        rng = random.Random(seed)
        
        # Phase 1: Greedy build
        greedy_deadline = min(deadline - 10, start_time + time_per_start * 0.4)
        edges, adj, adj_bits = _greedy_build(n, rng, greedy_deadline)
        
        if len(edges) > best_overall:
            best_overall = len(edges)
            best_edges_overall = edges.copy()
        
        # Phase 2: Local search
        ls_deadline = min(deadline - 5, start_time + time_per_start * 0.8)
        edges, count = _local_search(edges, adj, adj_bits, n, ls_deadline, rng)
        
        if count > best_overall:
            best_overall = count
            best_edges_overall = edges.copy()
        
        # Phase 3: Spectral perturbation on best found so far
        if s >= 2 and time.time() < deadline - 10:
            pert_deadline = min(deadline - 3, start_time + time_per_start)
            # Build base from best_edges_overall
            base_adj = [set() for _ in range(n)]
            for u, v in best_edges_overall:
                base_adj[u].add(v)
                base_adj[v].add(u)
            
            p_edges, p_adj, p_bits = _spectral_perturb(base_adj, n, rng, pert_deadline)
            if len(p_edges) > best_overall:
                best_overall = len(p_edges)
                best_edges_overall = p_edges.copy()
    
    # Final verification
    if best_edges_overall and not _verify_c6_free(best_edges_overall, n):
        # Fallback to 8 K5s
        fallback = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    fallback.add((u, v))
        return [list(e) for e in fallback]
    
    return [list(e) for e in best_edges_overall]

# EVOLVE-BLOCK-END
