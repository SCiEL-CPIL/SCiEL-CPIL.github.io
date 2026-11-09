"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import random
import time
import sys

_START_TIME = time.time()
DEADLINE = 115.0  # safety margin before 120s limit

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _is_k4_free(adj, n=18):
    """Check no K4 in graph with adjacency bitmasks adj (list of ints).
    A K4 exists iff some edge (a,b) has a common neighbor x that is also
    adjacent to another common neighbor.  Equivalently, for some edge (a,b),
    there is x in N(a)∩N(b) with N(x)∩N(a)∩N(b) non-empty.
    Fast path: precompute tri[x] = {y : x-y is in a triangle}.
    Then K4 exists iff ∃ edge (a,b) with (tri[a] & tri[b] & (1<<a) & (1<<b) & adj[a] & adj[b]) != 0.
    Simpler: ∃ edge (a,b) such that (tri[a] & tri[b]) has a bit c where adj[a] & adj[b] & (1<<c) is set.
    Actually simplest: ∃ a<b with edge, and (tri[a] & tri[b]) != 0 where tri already encodes
    "x is in a triangle with a" meaning x~a and ∃z: x~z~a.  Then a,b,x form triangle;
    need one more vertex d adjacent to all of a,b,x → d in N(a)∩N(b)∩N(x).
    """
    # Precompute tri[x]: bitmask of vertices y such that {x,y} lies in some triangle
    tri = [0] * n
    for x in range(n):
        ax = adj[x]
        c = ax
        while c:
            lsb = c & -c
            y = lsb.bit_length() - 1
            c -= lsb
            if (adj[y] & ax) != 0:
                tri[x] |= (1 << y)
    # Now check: for every edge (a,b), is there a vertex c in N(a)∩N(b) that is also
    # in a triangle with a (i.e., c in tri[a]) AND in a triangle with b?  Actually we need
    # c adjacent to a, b, and some d adjacent to a, b, c.  Simpler: for edge (a,b),
    # common = adj[a] & adj[b].  If common has two vertices c,d with c~d, then K4.
    # Equivalent: for edge (a,b), let common = adj[a]&adj[b].  K4 iff ∃ c in common
    # such that (adj[c] & common) != 0.
    # Use tri: c in common and c in tri[a] means c~a and ∃z: c~z~a.  Not quite.
    # Let's just use the direct check but with bit tricks:
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            # K4 iff ∃ c in common with (adj[c] & common) != 0
            # i.e., the subgraph induced by common has at least one edge.
            # Fast: for each c in common, check adj[c] & common.
            # Optimize: compute union of adj[c] for c in common, intersect with common.
            # But that's expensive.  Use: if common has <= 1 bit, no K4 through (a,b).
            if common & (common - 1) == 0:
                continue  # at most one common neighbor
            # Check if any pair in common is adjacent
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    """Exact branch-and-bound for largest triangle-free induced subgraph.
    
    A set S is triangle-free iff no two vertices in S have a common neighbor
    that is also in S.  We branch on vertices in order 0..n-1.
    State: (idx, current_mask, forbidden_mask) where forbidden_mask[i]=1 means
    vertex i cannot be added (because adding it would create a triangle with
    two already-included vertices).
    
    Optimizations:
    - Precompute suffix_available[idx] = bitmask of vertices idx..n-1.
    - available = suffix_available[idx] & ~forbidden_mask, computed in O(1).
    - Use int.bit_count() (Python 3.10+) for speed.
    """
    best = [0]
    
    # Precompute suffix masks
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] | (1 << i)
    
    def bb(idx, current_mask, forbidden_mask):
        if _elapsed() > DEADLINE - 2:
            return
        
        # Upper bound: current + all remaining not forbidden
        available = suffix[idx] & ~forbidden_mask
        ub = current_mask.bit_count() + available.bit_count()
        if ub <= best[0]:
            return
        
        if idx == n:
            sz = current_mask.bit_count()
            if sz > best[0]:
                best[0] = sz
            return
        
        v = idx
        bit = 1 << v
        
        # Option 1: Exclude v (try first to prune early)
        bb(idx + 1, current_mask, forbidden_mask)
        
        # Option 2: Include v
        if not (forbidden_mask & bit):
            # New forbidden: any vertex w that is adjacent to v and to some u in current_mask
            # (because then {v, u, w} would be a triangle if w is also added, but actually
            #  we need: w adjacent to v AND w adjacent to some u in current → w forbidden)
            # More precisely: adding v to S creates a triangle iff ∃ u∈S: v~u and ∃w∈S: 
            #  v~w and u~w.  So w is forbidden if w~v and w~u for some u∈S.
            # new_forbidden = forbidden_mask | union over u in S where v~u of (adj[v] & adj[u])
            new_forbidden = forbidden_mask
            cm = current_mask
            av = adj[v]
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                if (av >> u) & 1:
                    new_forbidden |= (av & adj[u])
            
            bb(idx + 1, current_mask | bit, new_forbidden)
    
    bb(0, 0, 0)
    return best[0]

def construct():
    n = 18
    
    # Start with complete tripartite graph K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
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
    
    # Local Search with Simulated Annealing
    T = 1.0
    no_improve_count = 0
    max_no_improve = 300
    
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
            continue
        
        # Evaluate score
        new_score = _max_triangle_free_induced(new_adj, n)
        
        # Accept if better or with probability based on simulated annealing
        if new_score < current_score:
            current_adj = new_adj
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_adj = new_adj[:]
                best_score = new_score
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = new_score - current_score
            if delta == 0:
                current_adj = new_adj
                current_score = new_score
            elif random.random() < (2.718281828 ** (-delta / T)):
                current_adj = new_adj
                current_score = new_score
            no_improve_count += 1
        
        # Cool down
        if no_improve_count >= max_no_improve:
            T *= 0.9
            no_improve_count = 0
            if T < 0.01:
                # Restart with fresh random perturbation
                current_adj = best_adj[:]
                current_score = best_score
                T = 0.5
                no_improve_count = 0
    
    # Return edges from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
