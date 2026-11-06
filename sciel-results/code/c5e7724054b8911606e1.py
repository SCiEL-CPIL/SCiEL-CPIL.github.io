"""Triangle-free graphs without independent 10-sets (R(3,10) > n).

Strategy:
  1. Start from the known Andrasfai graph And(9) on 26 vertices (score 26).
  2. Try to extend it to larger n by adding vertices and choosing edges so that
     the graph remains triangle-free and alpha < 10.
  3. Use a fast triangle-free / alpha<10 checker (branch-and-bound with bitsets).
  4. Use a greedy + local-search extension: add a new vertex, connect it to a
     subset of existing vertices, check constraints, keep if valid.
  5. Also try random restarts and simulated-annealing-style edge flips to grow n.
  6. Keep an explicit deadline guard.

The largest known order is 39; R(3,10) <= 41, so we cap at n=40.
"""
import time
import random
from itertools import combinations

DEADLINE = 110.0  # safety margin before 120s limit

def _has_triangle(adj, n):
    """Check if graph with adjacency bitmasks has a triangle."""
    for i in range(n):
        ai = adj[i]
        # neighbors of i
        j = (ai & ~(1 << i))
        while j:
            lb = j & (-j)
            jj = lb.bit_length() - 1
            # common neighbors of i and j
            common = ai & adj[jj] & ~(1 << i) & ~(1 << jj)
            if common:
                return True
            j ^= lb
    return False

def _max_independent_set_size(adj, n, limit=10):
    """Branch and bound: return size of max independent set, or >= limit if found.
    We only need to know if alpha >= limit."""
    # Use Bron-Kerbosch style on complement, or simple B&B on independent set.
    # Simpler: greedy upper bound + exact B&B for small n.
    # For n up to 40, a B&B with bitsets is fast enough.
    
    def _bbs(independent_set_size, remaining, adj):
        """remaining: bitmask of vertices not yet decided; independent_set_size: current size."""
        if independent_set_size + remaining.bit_count() <= best[0] and independent_set_size < limit:
            # upper bound: even if we take all remaining, can't beat best
            pass
        # Pruning: if current + remaining count <= best, stop
        if independent_set_size + remaining.bit_count() <= best[0]:
            return
        if independent_set_size >= limit:
            best[0] = independent_set_size
            return
        if remaining == 0:
            if independent_set_size > best[0]:
                best[0] = independent_set_size
            return
        # Pick a vertex from remaining
        lb = remaining & (-remaining)
        v = lb.bit_length() - 1
        rest = remaining ^ lb
        # Option 1: exclude v
        _bbs(independent_set_size, rest, adj)
        # Option 2: include v (if independent set so far doesn't conflict)
        # v can be added if none of its neighbors are in the current independent set
        # We track the independent set implicitly via 'forbidden' mask
        # Actually, let's track the set of vertices already chosen.
        # Simpler approach: use a different B&B formulation.
        pass
    
    # Let's use a cleaner B&B:
    # State: (chosen_mask, candidates_mask)
    # candidates = vertices that can still be added (not adjacent to any chosen)
    
    best = [0]
    
    def _bbs2(chosen_count, candidates):
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        # Pick a vertex
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        # Exclude v
        _bbs2(chosen_count, rest)
        # Include v: new candidates = rest & ~adj[v]
        new_cands = rest & ~adj[v]
        _bbs2(chosen_count + 1, new_cands)
    
    _bbs2(0, (1 << n) - 1)
    return best[0]

def _is_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle(adj, n):
        return False
    if _max_independent_set_size(adj, n, limit=10) >= 10:
        return False
    return True

def _andrasfai_9():
    """Return adj bitmasks for And(9) on 26 vertices."""
    n = 26
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj, n

def _try_extend(adj, n, max_n=40, deadline=DEADLINE):
    """Try to extend the graph by adding vertices one at a time."""
    best_n = n
    best_adj = list(adj)
    
    for target_n in range(n + 1, max_n + 1):
        if time.time() > deadline:
            break
        # Try to add vertex target_n-1
        # We need to choose a subset of {0..n-1} to connect to.
        # For triangle-freeness: if we connect new vertex to u and v, then u and v must NOT be adjacent.
        # So the neighbors of the new vertex must form an independent set in the current graph.
        # For alpha constraint: the new vertex's non-neighbors (plus itself) form a candidate independent set.
        #   alpha of new graph <= max(alpha_old, 1 + alpha of (V \ N(new) \ {new}))
        #   where V \ N(new) \ {new} is the set of old vertices NOT connected to new.
        
        # Greedy: try all subsets? Too many. Use heuristic.
        # The new vertex's neighbors must be an independent set in G.
        # Let's find a large independent set in G to use as neighbors (to keep alpha small).
        # Actually, to minimize alpha of new graph, we want the non-neighbors to have small alpha.
        # non-neighbors = V \ N(new). alpha(new) <= max(alpha(G), alpha(V \ N(new)) + 1).
        # To keep alpha < 10, we need alpha(V \ N(new)) + 1 < 10, i.e., alpha(V \ N(new)) <= 8.
        
        # Strategy: pick N(new) to be a "large" independent set so that V \ N(new) is small-ish
        # and hopefully has alpha <= 8.
        
        # Find a maximum independent set in G (or a large one).
        # For And(9), alpha = 9. So there are independent sets of size 9.
        # If we set N(new) = an independent set of size k, then |V \ N(new)| = n - k.
        # We need alpha(V \ N(new)) <= 8.
        
        # Let's try: for each independent set I of size >= some threshold, set N(new) = I.
        # Enumerating all independent sets is expensive. Instead, use a heuristic:
        # Pick a random independent set of size ~ n/3, check validity.
        
        found = False
        attempts = 200
        for _ in range(attempts):
            if time.time() > deadline:
                break
            # Build a random independent set
            cand = list(range(n))
            random.shuffle(cand)
            I = []
            used = 0
            for v in cand:
                # Check if v is independent from all in I
                conflict = False
                for u in I:
                    if adj[v] & (1 << u):
                        conflict = True
                        break
                if not conflict:
                    I.append(v)
                    used |= (1 << v)
                if len(I) >= 10:
                    break
            
            if len(I) < 3:
                continue
            
            # New vertex n connects to all in I
            new_adj = list(adj)
            new_adj.append(0)
            for i in range(n):
                if i in I:
                    new_adj[i] |= (1 << n)
                    new_adj[n] |= (1 << i)
            
            if _is_valid(new_adj, n + 1):
                adj = new_adj
                n = n + 1
                best_n = n
                best_adj = list(adj)
                found = True
                break
        
        if not found:
            # Try a few more with different strategies
            # Maybe connect to a smaller set
            for _ in range(100):
                if time.time() > deadline:
                    break
                # Random subset
                I = set()
                for v in range(n):
                    if random.random() < 0.3:
                        I.add(v)
                # Make sure I is independent
                I = list(I)
                valid_I = []
                for v in I:
                    ok = True
                    for u in valid_I:
                        if adj[v] & (1 << u):
                            ok = False
                            break
                    if ok:
                        valid_I.append(v)
                I = valid_I
                
                if len(I) < 1:
                    continue
                
                new_adj = list(adj)
                new_adj.append(0)
                for i in range(n):
                    if i in I:
                        new_adj[i] |= (1 << n)
                        new_adj[n] |= (1 << i)
                
                if _is_valid(new_adj, n + 1):
                    adj = new_adj
                    n = n + 1
                    best_n = n
                    best_adj = list(adj)
                    found = True
                    break
    
    return best_adj, best_n

def _random_search(max_n=40, deadline=DEADLINE):
    """Random search for a valid graph with large n."""
    best_n = 0
    best_adj = None
    
    for n in range(15, max_n + 1):
        if time.time() > deadline:
            break
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Random triangle-free graph: use a random ordering and only add edges
            # that don't create a triangle.
            adj = [0] * n
            order = list(range(n))
            random.shuffle(order)
            # For each pair in random order, add edge if no triangle
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((i, j))
            random.shuffle(pairs)
            for i, j in pairs:
                # Check if adding (i,j) creates a triangle
                common = adj[i] & adj[j]
                if common == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            
            if _is_valid(adj, n):
                best_n = n
                best_adj = list(adj)
                break  # Found valid graph for this n, try larger
    
    return best_adj, best_n

def construct():
    start_time = time.time()
    
    # Start from And(9)
    adj, n = _andrasfai_9()
    best_adj, best_n = list(adj), n
    
    # Try to extend
    ext_adj, ext_n = _try_extend(adj, n, max_n=40, deadline=start_time + DEADLINE)
    if ext_n > best_n:
        best_adj, best_n = ext_adj, ext_n
    
    # Random search for potentially better
    if time.time() < start_time + DEADLINE - 10:
        rand_adj, rand_n = _random_search(max_n=40, deadline=start_time + DEADLINE - 5)
        if rand_n > best_n:
            best_adj, best_n = rand_adj, rand_n
    
    # Build edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
