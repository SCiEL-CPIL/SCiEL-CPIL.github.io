import time
import random
import numpy as np

N = 40
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]

def _init_adj():
    return [set() for _ in range(N)]

def _add(adj, u, v):
    adj[u].add(v)
    adj[v].add(u)

def _remove(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)

def _has_c6_fast(adj, u, v):
    """
    Check if adding edge (u,v) creates a C6.
    C6: u-v-a-b-c-d-u
    Requires d in N(u)\{v} and path v-a-b-c-d of length 4 in G.
    Path vertices a,b,c must be distinct and not u,v,d.
    """
    nu = adj[u]
    if v not in nu: # u and v must be non-adjacent for this to be a new edge check, but logic holds generally
        pass
    
    # We look for d in N(u) such that d != v
    for d in nu:
        if d == v:
            continue
        # Check if there is a path of length 4 from v to d in G
        # v - a - b - c - d
        # a in N(v) \ {u}
        # b in N(a) \ {u, v, d}
        # c in N(b) \ {u, v, a, d}
        # d in N(c)
        
        # Optimization: If d is not reachable from v in 4 steps, skip.
        # Simple iteration
        for a in adj[v]:
            if a == u:
                continue
            for b in adj[a]:
                if b == u or b == v or b == d:
                    continue
                for c in adj[b]:
                    if c == u or c == v or c == a or c == d:
                        continue
                    if d in adj[c]:
                        return True
    return False

def _edge_count(adj):
    return sum(len(s) for s in adj) // 2

def _greedy_build(rng):
    adj = _init_adj()
    order = list(ALL_PAIRS)
    rng.shuffle(order)
    for (u, v) in order:
        if v in adj[u]:
            continue
        if not _has_c6_fast(adj, u, v):
            _add(adj, u, v)
    return adj

def _local_search_sa(adj, rng, time_limit):
    """
    Simulated Annealing / Local Search
    """
    cur = [set(s) for s in adj]
    cur_count = _edge_count(cur)
    best = [set(s) for s in cur]
    best_count = cur_count
    
    start_time = time.time()
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if T < 0.01:
            # Greedy mode: only accept improvements
            # Try to add an edge
            improved = False
            for _ in range(50): # random attempts
                u = rng.randrange(N)
                v = rng.randrange(N)
                if u == v or v in adj[u]:
                    continue
                if not _has_c6_fast(cur, u, v):
                    _add(cur, u, v)
                    cur_count += 1
                    if cur_count > best_count:
                        best_count = cur_count
                        best = [set(s) for s in cur]
                    improved = True
                    break
            if not improved:
                # Try a swap
                # Pick random edge to remove
                edges = []
                for u in range(N):
                    for v in cur[u]:
                        if u < v:
                            edges.append((u, v))
                if not edges:
                    break
                u1, v1 = rng.choice(edges)
                _remove(cur, u1, v1)
                cur_count -= 1
                
                # Pick random non-edge to add
                for _ in range(50):
                    u2 = rng.randrange(N)
                    v2 = rng.randrange(N)
                    if u2 == v2 or v2 in cur[u2]:
                        continue
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    if not _has_c6_fast(cur, u2, v2):
                        _add(cur, u2, v2)
                        cur_count += 1
                        if cur_count > best_count:
                            best_count = cur_count
                            best = [set(s) for s in cur]
                        break
                    else:
                        # revert add attempt
                        pass
                else:
                    # couldn't add, revert remove
                    _add(cur, u1, v1)
                    cur_count += 1
        else:
            # SA mode: allow worse moves
            # Pick random edge to remove
            edges = []
            for u in range(N):
                for v in cur[u]:
                    if u < v:
                        edges.append((u, v))
            if not edges:
                break
            u1, v1 = rng.choice(edges)
            _remove(cur, u1, v1)
            cur_count -= 1
            
            # Pick random non-edge to add
            swap_found = False
            for _ in range(100):
                u2 = rng.randrange(N)
                v2 = rng.randrange(N)
                if u2 == v2 or v2 in cur[u2]:
                    continue
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                if not _has_c6_fast(cur, u2, v2):
                    _add(cur, u2, v2)
                    cur_count += 1
                    delta = 0 # net 0 change in count for a swap
                    # Wait, a swap is remove 1, add 1. Count stays same.
                    # To improve, we need to find a state where we can add MORE edges.
                    # Standard SA on edge count: 
                    # Move: Remove (u1,v1), Add (u2,v2). Count unchanged.
                    # This doesn't help maximize count directly unless it unlocks more edges.
                    # Better move: Just remove an edge? No, count drops.
                    # Better move: Add an edge if possible.
                    
                    # Let's stick to: 
                    # 1. Try to add a random non-edge. If success, count+1.
                    # 2. If stuck, do a swap to rearrange structure.
                    
                    swap_found = True
                    break
            if not swap_found:
                _add(cur, u1, v1)
                cur_count += 1
            else:
                # We performed a swap. Count is same.
                # We accept the swap if it leads to a "better" state? 
                # Hard to define. Let's just accept swaps randomly to explore.
                # Or, check if this swap allows immediate addition of another edge?
                # Too complex. Let's just accept the swap to explore.
                if cur_count > best_count:
                    best_count = cur_count
                    best = [set(s) for s in cur]
        
        T *= cooling

    return best, best_count

def construct():
    deadline = 120.0
    start = time.time()
    rng = random.Random(12345)
    
    best_adj = None
    best_count = -1
    
    # Phase 1: Multiple greedy builds
    for i in range(10):
        if time.time() > deadline - 60:
            break
        adj = _greedy_build(rng)
        cnt = _edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
            
    # Phase 2: Local Search on best
    if best_adj:
        remaining = deadline - 10 - (time.time() - start)
        if remaining > 5:
            adj2, cnt2 = _local_search_sa(best_adj, rng, remaining * 0.7)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2
                
    # Phase 3: More greedy + LS if time
    for i in range(5):
        if time.time() > deadline - 20:
            break
        adj = _greedy_build(rng)
        cnt = _edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
            
        remaining = deadline - 10 - (time.time() - start)
        if remaining > 5:
            adj2, cnt2 = _local_search_sa(adj, rng, min(remaining * 0.3, 10))
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2

    # Phase 4: Final polish
    if best_adj:
        remaining = deadline - 5 - (time.time() - start)
        if remaining > 2:
            adj2, cnt2 = _local_search_sa(best_adj, rng, remaining * 0.8)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2

    if best_adj is None:
        # Fallback
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges

    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    return edges
