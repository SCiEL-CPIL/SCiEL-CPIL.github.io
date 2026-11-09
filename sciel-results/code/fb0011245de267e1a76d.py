import random
import time
import sys

_START_TIME = time.time()
DEADLINE = 110.0  # seconds, safety margin before 120s

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _is_k4_free(adj, n=18):
    """Check no K4 in graph with adjacency bitmasks adj (list of ints)."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            # Check if common contains an edge (i.e., two vertices in common are adjacent)
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    """Exact branch-and-bound for largest triangle-free induced subgraph."""
    # We want the maximum size of a subset S of vertices such that the induced
    # subgraph on S has no triangles.
    
    # Precompute for each vertex v, the set of vertices that form a triangle with v
    # i.e., for an edge (u, w), v is in tri[u] if adj[u][v] and adj[w][v] and adj[u][w].
    # Actually, it's easier to check when adding a vertex to a set.
    
    # Let's use a recursive approach.
    # State: current_set (bitmask), available (bitmask of vertices not yet considered)
    # We process vertices in a specific order to prune effectively.
    
    # Optimization: If we have a candidate set S, and we consider adding v,
    # v can be added iff there is no edge (u, w) in S such that v is adjacent to both u and w.
    # Equivalently, v is not in the "forbidden" set induced by S.
    # Forbidden(v, S) = { v | exists u, w in S, u~w, u~v, w~v }
    
    # Let's precompute for every pair (u, w) that is an edge, the set of common neighbors.
    # common[u][w] = adj[u] & adj[w]
    # Then v is forbidden for S if v is in common[u][w] for some edge (u, w) in S.
    
    # To make it faster, we can maintain a bitmask 'forbidden' which is the union of
    # common[u][w] for all edges (u, w) in the current set S.
    # When we add a vertex v to S, we must ensure v is not in 'forbidden'.
    # Then the new forbidden set becomes forbidden | (union of common[v][u] for all u in S).
    
    # This is still potentially expensive if we recompute the union.
    # However, n=18 is small.
    
    # Let's try a different angle:
    # Iterate over all subsets? 2^18 = 262,144.
    # For each subset, check if it's triangle-free.
    # Checking if a subset is triangle-free:
    # For each u in subset, check if adj[u] & subset is an independent set.
    # This is O(k^2) or O(k * 2^k) per subset? No, O(k^2) per subset is fine.
    # Total time: 2^18 * 18^2 ~ 262k * 324 ~ 85M ops. This might be too slow in Python (seconds to minutes).
    # We need branch and bound.
    
    # Branch and Bound:
    # Order vertices.
    # Recurse on index i.
    # State: current_mask (vertices included), forbidden_mask (vertices that cannot be included).
    # Upper bound: popcount(current_mask) + popcount(remaining vertices that are not forbidden).
    
    # Let's refine the forbidden logic.
    # If we have selected a set S.
    # A vertex v can be added to S iff S does not contain any edge (u, w) such that v is adjacent to u and w.
    # Let F(S) be the set of vertices that cannot be added to S.
    # F(S) = union over all edges (u, w) in S of (adj[u] & adj[w]).
    # Note: u and w themselves are in S, so they are "used".
    # When we add v to S, the new set is S' = S U {v}.
    # F(S') = F(S) U (union over all u in S of (adj[v] & adj[u])).
    
    # We can maintain F(S) incrementally.
    
    best = [0]
    
    # Precompute common neighbors for all pairs?
    # common[u][v] = adj[u] & adj[v]
    # We only care about edges (u, v).
    
    def popcount(x):
        return bin(x).count('1')
    
    # Order vertices by degree descending? Or just 0..n-1?
    # Let's try 0..n-1 first.
    
    def bb(idx, current_mask, forbidden_mask):
        if _elapsed() > DEADLINE:
            return
        
        # Upper bound
        # Remaining vertices: those with index >= idx that are not in forbidden_mask and not in current_mask
        # Actually, we process vertices 0..n-1 in order.
        # So "remaining" are vertices i where i >= idx.
        # We can only include i if it's not in forbidden_mask.
        
        # Simple bound: current_size + (n - idx)
        # Better bound: current_size + popcount(available)
        # Where available = bitmask of vertices i >= idx such that (forbidden_mask >> i) & 1 == 0
        
        # Computing popcount(available) every step is O(n).
        # n=18, so it's fine.
        
        available = 0
        for i in range(idx, n):
            if not (forbidden_mask >> i) & 1:
                available |= (1 << i)
        
        ub = popcount(current_mask) + popcount(available)
        if ub <= best[0]:
            return
        
        if idx == n:
            sz = popcount(current_mask)
            if sz > best[0]:
                best[0] = sz
            return
        
        v = idx
        bit = 1 << v
        
        # Option 1: Exclude v
        bb(idx + 1, current_mask, forbidden_mask)
        
        # Option 2: Include v
        # Check if v is forbidden
        if not (forbidden_mask & bit):
            # Compute new forbidden set
            # new_forbidden = forbidden_mask
            # For each u in current_mask, if adj[v] & adj[u] is non-zero, add those to new_forbidden?
            # Wait, the definition of F(S) is vertices that form a triangle with an edge in S.
            # If we add v, new edges are (v, u) for u in S.
            # For each edge (v, u), the common neighbors are adj[v] & adj[u].
            # These common neighbors become forbidden for future additions.
            
            new_forbidden = forbidden_mask
            cm = current_mask
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                # If u and v are connected
                if (adj[v] >> u) & 1:
                    new_forbidden |= (adj[v] & adj[u])
            
            # Note: v itself is not in new_forbidden necessarily, but we don't add v again.
            # Also, vertices in current_mask are already "used", so it doesn't matter if they are in forbidden.
            
            bb(idx + 1, current_mask | bit, new_forbidden)
            
    bb(0, 0, 0)
    return best[0]

def construct():
    n = 18
    # Start with K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    current_adj = adj[:]
    best_adj = adj[:]
    
    # Initial evaluation
    current_score = _max_triangle_free_induced(current_adj, n)
    best_score = current_score
    
    random.seed(42)
    
    # Local Search
    # We want to minimize the max triangle-free induced subgraph size.
    # Strategy: Randomly flip edges, check K4-free, evaluate score.
    # Accept if better, or with some probability if worse (Simulated Annealing).
    
    T = 1.0
    step = 0
    no_improve_count = 0
    
    while _time_left() > 5:
        # Pick random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Flip edge
        new_adj = current_adj[:]
        if (new_adj[u] >> v) & 1:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
        
        # Check K4-free
        if not _is_k4_free(new_adj, n):
            T *= 0.99
            step += 1
            continue
        
        # Evaluate
        new_score = _max_triangle_free_induced(new_adj, n)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = new_adj
            current_adj = new_adj
            T = 1.0
            no_improve_count = 0
        else:
            # Accept with probability
            delta = new_score - best_score
            if delta > 0:
                prob = 1.0 / (1.0 + step / 50.0)
                if random.random() < prob:
                    current_adj = new_adj
            else:
                # new_score == best_score
                # Accept with some probability to explore
                if random.random() < 0.5:
                    current_adj = new_adj
            
            no_improve_count += 1
            T *= 0.99
            step += 1
        
        # If we've been stuck for a long time, try a bigger jump?
        # For now, just continue.
        
        if best_score <= 9:
            # Can we go lower? Theoretically, for n=18, if we can get it down to 9, that's great.
            # Break if we hit a hard lower bound? We don't know the exact bound.
            # Just break if we are very confident or time is running out.
            pass

    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
