import time
import random
import itertools
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before 150s kill

def _deadline():
    return time.time() >= _DEADLINE

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _random_graph(n, p=0.5, rng=None):
    rng = rng or random
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _is_regular_mask(adj, mask):
    """True if induced subgraph on mask is regular."""
    if mask == 0:
        return False
    first = -1
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        m ^= lsb
        deg = (adj[v] & mask).bit_count()
        if first == -1:
            first = deg
        elif deg != first:
            return False
    return True

def _count_regular_subgraphs(adj, n, min_size=6):
    """Count number of regular induced subgraphs of size >= min_size."""
    cnt = 0
    for k in range(min_size, n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular_mask(adj, mask):
                cnt += 1
    return cnt

def _try_local_search(adj, n, rng, max_iters=5000):
    """
    Local search with delta evaluation.
    Maintains a count of regular subgraphs.
    """
    if _deadline():
        return adj, 0
    
    # Initialize state
    # We need to know which subsets are regular to update counts efficiently.
    # Storing all regular masks might be memory heavy if many, but usually few.
    # Let's just re-calculate count from scratch if delta is too complex?
    # No, delta is better.
    
    # Precompute all masks of size >= 6?
    # For n=20, total masks 2^20 = 1M.
    # We can precompute a list of all masks of size >= 6.
    # But iterating 1M masks to init is slow.
    
    # Alternative: 
    # Just do a standard local search but with a *faster* validity check?
    # The bottleneck is `_count_regular_subgraphs`.
    
    # Let's try a different approach:
    # 1. Start with a random graph.
    # 2. Count regular subgraphs.
    # 3. If > 0, find one.
    # 4. Flip an edge in that subgraph.
    # 5. Re-count.
    
    # This is what the current code does, but re-counting is slow.
    
    # Optimized Re-count:
    # When flipping (u, v), only subsets containing both u and v change.
    # Let's implement this.
    
    best_adj = adj[:]
    best_cnt = _count_regular_subgraphs(best_adj, n, 6)
    if best_cnt == 0:
        return best_adj, 0
    
    cur = adj[:]
    cur_cnt = best_cnt
    
    # Precompute list of masks containing specific pairs?
    # Or just iterate combinations of the remaining n-2 vertices.
    
    no_improve = 0
    for it in range(max_iters):
        if _deadline():
            break
            
        # Pick random edge to flip
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
            
        # Apply flip
        bit_u = 1 << u
        bit_v = 1 << v
        cur[u] ^= bit_v
        cur[v] ^= bit_u
        
        # Calculate Delta
        # Subsets S of size >= 6 containing u and v.
        # Let S = {u, v} U T, where T is a subset of V \ {u, v}.
        # |S| >= 6 => |T| >= 4.
        
        delta = 0
        remaining = [x for x in range(n) if x != u and x != v]
        
        # Iterate T subsets of remaining with size >= 4
        # This is sum_{k=4}^{n-2} C(n-2, k)
        # For n=20, sum_{k=4}^{18} C(18, k) = 2^18 - C(18,0)-C(18,1)-C(18,2)-C(18,3)
        # 2^18 = 262144. C(18,3) = 816. So ~261k subsets.
        # Checking each is O(1) if we can update degrees fast?
        # No, checking regularity is O(|S|).
        # 261k * 6 avg size = 1.5M ops per flip.
        # 5000 flips = 7.5B ops. Too slow.
        
        # We need to be smarter.
        # Only check subsets that *were* regular or *might* become regular?
        # Hard to know "might".
        
        # Alternative: 
        # Keep a list of "critical" subsets?
        
        # Let's try a simpler heuristic:
        # If cur_cnt is small, finding the specific regular subgraphs is fast.
        # If cur_cnt is large, random flips might not help much.
        
        # Let's just do a limited re-count.
        # If we assume the number of regular subgraphs is small, we can find them.
        
        # Revised Strategy:
        # 1. Find ONE regular subgraph.
        # 2. Flip a random edge in it.
        # 3. Check if that specific subgraph is still regular.
        # 4. Also check if any OTHER subgraph became regular? (Hard)
        
        # Actually, if we just flip edges in existing regular subgraphs, we break them.
        # We might create new ones, but probability is low if we start random.
        
        # Let's implement "Find one, break it" loop.
        
        if cur_cnt == 0:
            return cur, 0
            
        # Find one regular subgraph
        found_mask = None
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v_ in comb:
                    mask |= (1 << v_)
                if _is_regular_mask(cur, mask):
                    found_mask = mask
                    break
            if found_mask:
                break
        
        if not found_mask:
            # Should not happen if cur_cnt > 0, but maybe race condition or logic error
            # Re-count to be safe
            cur_cnt = _count_regular_subgraphs(cur, n, 6)
            if cur_cnt == 0:
                return cur, 0
            # If still not found, just break
            break
            
        # Flip a random edge within found_mask
        vertices_in_mask = [v_ for v_ in range(n) if found_mask & (1 << v_)]
        # Pick two distinct vertices from this list
        if len(vertices_in_mask) < 2:
             # Should not happen for size >= 6
             continue
             
        i_idx = rng.randrange(len(vertices_in_mask))
        j_idx = rng.randrange(len(vertices_in_mask))
        while i_idx == j_idx:
            j_idx = rng.randrange(len(vertices_in_mask))
            
        u = vertices_in_mask[i_idx]
        v = vertices_in_mask[j_idx]
        
        # Flip edge (u, v)
        bit_u = 1 << u
        bit_v = 1 << v
        cur[u] ^= bit_v
        cur[v] ^= bit_u
        
        # Check if found_mask is still regular
        if _is_regular_mask(cur, found_mask):
            # It is still regular. Try flipping another edge or accept?
            # If it's still regular, we didn't break it.
            # Undo flip?
            cur[u] ^= bit_v
            cur[v] ^= bit_u
            no_improve += 1
            if no_improve > 50:
                 # Perturbation
                 for _ in range(10):
                     a = rng.randrange(n)
                     b = rng.randrange(n)
                     if a != b:
                         cur[a] ^= (1 << b)
                         cur[b] ^= (1 << a)
                 no_improve = 0
                 cur_cnt = _count_regular_subgraphs(cur, n, 6)
                 if cur_cnt == 0:
                     return cur, 0
        else:
            # We broke this one.
            # But did we create others?
            # Re-count is expensive.
            # Let's just update cur_cnt optimistically?
            # No, we must know.
            # Let's re-count only if we suspect we are close to 0.
            # Or just re-count every 100 successful breaks?
            
            # For now, let's just re-count. It's the safest.
            # Optimization: Only re-count if we have broken many?
            # Actually, if we break one, count decreases by at least 1.
            # It might increase by others.
            
            # Let's re-count.
            new_cnt = _count_regular_subgraphs(cur, n, 6)
            
            if new_cnt < cur_cnt:
                cur_cnt = new_cnt
                if cur_cnt < best_cnt:
                    best_adj = cur[:]
                    best_cnt = cur_cnt
                no_improve = 0
            else:
                # Count didn't decrease (created new ones or count was wrong)
                # Undo?
                cur[u] ^= bit_v
                cur[v] ^= bit_u
                no_improve += 1
                
                if no_improve > 50:
                    for _ in range(10):
                        a = rng.randrange(n)
                        b = rng.randrange(n)
                        if a != b:
                            cur[a] ^= (1 << b)
                            cur[b] ^= (1 << a)
                    cur_cnt = _count_regular_subgraphs(cur, n, 6)
                    no_improve = 0
                    if cur_cnt == 0:
                        return cur, 0
                        
    return best_adj, best_cnt

def construct():
    rng = random.Random(12345)
    best_n = 5
    best_adj = [0] * 5
    best_edges = []

    # Try increasing n from 6 to 22
    for target_n in range(6, 23):
        if _deadline():
            break
        found = False
        
        # Strategy 1: Random graphs + Local Search
        for attempt in range(100):
            if _deadline():
                break
            
            # Vary p
            p = rng.uniform(0.3, 0.7)
            adj = _random_graph(target_n, p, rng)
            
            # Check validity quickly
            if _count_regular_subgraphs(adj, target_n, 6) == 0:
                found = True
                best_n = target_n
                best_adj = adj
                best_edges = _edges_from_adj(adj, target_n)
                break
            
            # Local Search
            improved, cnt = _try_local_search(adj, target_n, rng, max_iters=2000)
            if cnt == 0:
                found = True
                best_n = target_n
                best_adj = improved
                best_edges = _edges_from_adj(improved, target_n)
                break
        
        if found:
            continue
        
        # If not found, stop
        break

    # Final validation
    if _count_regular_subgraphs(best_adj, best_n, 6) == 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    return {'n': 5, 'edges': []}
