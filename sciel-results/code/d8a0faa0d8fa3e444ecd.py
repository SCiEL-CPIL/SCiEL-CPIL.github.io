import time
import random
from itertools import combinations

def construct():
    """Construct a dense triangle-free graph on 16 vertices that is not 3-colorable."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n = 16
    all_pairs = list(combinations(range(n), 2))
    n_pairs = len(all_pairs)  # 120
    
    # Precompute triangle conflicts: for each edge index, which other edges conflict
    edge_to_uv = {}
    for i, (u, v) in enumerate(all_pairs):
        edge_to_uv[i] = (u, v)
    
    # conflict[i] = set of edge indices that conflict with edge i (would form triangle)
    conflict = [set() for _ in range(n_pairs)]
    for i in range(n_pairs):
        u, v = all_pairs[i]
        for j in range(n_pairs):
            if i == j:
                continue
            w1, w2 = all_pairs[j]
            # Check if adding edge j to a graph containing edge i creates a triangle
            # Triangle exists if {u,v} and {w1,w2} share a vertex and the third vertex is adjacent to both
            if u in (w1, w2) and v not in (w1, w2):
                # Common vertex is u, other vertices are v and the one not u in (w1,w2)
                other = w2 if w1 == u else w1
                # Triangle if u-other and v-other... wait, triangle means 3 vertices all connected
                # Vertices are u, v, other. Need edges (u,v), (u,other), (v,other)
                # We have (u,v)=i. If j=(u,other), need (v,other)
                # If j=(v,other), need (u,other)
                pass
            # More directly: edges i=(a,b) and j=(c,d) conflict if they share a vertex
            # and the two "outer" vertices are also connected (but that's a third edge)
            # Actually for triangle-freeness: we can't have edges (a,b), (a,c), (b,c) all present
            # So edge i=(a,b) conflicts with the pair {(a,c), (b,c)} for any c != a,b
            # For local search: if we add edge (a,b), we must ensure no common neighbor
            # So conflict between adding edge i and edge j: they share a vertex and the 
            # non-shared vertices are... let me think differently.
            # Two edges (a,b) and (c,d) are incompatible if they share exactly one vertex
            # say a=c, and b != d, and we also have edge (b,d). But that's 3 edges.
            # For the constraint "no triangle": a set of edges is triangle-free iff for every
            # triple of vertices, not all 3 edges among them are present.
            # In local search, when we toggle edge i=(u,v), the constraint violated is:
            # for any w, if (u,w) and (v,w) are both present, then (u,v) cannot be present.
            # So edge i=(u,v) is "conflicted" by the presence of both (u,w) and (v,w) for some w.
            # For WalkSAT-style: when considering flipping edge i, check if it's currently
            # causing a triangle or if flipping would create one.
    
    # Better approach: precompute for each edge index, the list of "witness" pairs
    # edge i = (u,v): witness is (w, edge_uw, edge_vw) where edge_uw is index of (u,w)
    # and edge_vw is index of (v,w). Triangle exists iff edge i, edge_uw, edge_vw all present.
    
    edge_index = {}
    for i, (u, v) in enumerate(all_pairs):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # For each edge i=(u,v), list of (w, idx_uw, idx_vw)
    witnesses = [[] for _ in range(n_pairs)]
    for i in range(n_pairs):
        u, v = all_pairs[i]
        for w in range(n):
            if w == u or w == v:
                continue
            if (u, w) in edge_index and (v, w) in edge_index:
                witnesses[i].append((w, edge_index[(u, w)], edge_index[(v, w)]))
    
    def is_triangle_free(x):
        """x is list of 0/1 for each edge. Check triangle-free."""
        for i in range(n_pairs):
            if x[i]:
                for (w, iw, jw) in witnesses[i]:
                    if x[iw] and x[jw]:
                        return False
        return True
    
    def count_violations(x):
        """Count number of triangles (each triangle counted 3 times, once per edge)."""
        cnt = 0
        for i in range(n_pairs):
            if x[i]:
                for (w, iw, jw) in witnesses[i]:
                    if x[iw] and x[jw]:
                        cnt += 1
        return cnt
    
    def is_3_colorable(x):
        """Check 3-colorability using backtracking with bitmasks."""
        # Build adjacency bitmasks
        adj_bits = [0] * n
        for i in range(n_pairs):
            if x[i]:
                u, v = all_pairs[i]
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        
        # Order vertices by degree descending
        degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                conflict = False
                # Check neighbors
                nb = adj_bits[v]
                while nb:
                    u = (nb & -nb).bit_length() - 1
                    if color[u] == c:
                        conflict = True
                        break
                    nb &= nb - 1
                if not conflict:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def edge_count(x):
        return sum(x)
    
    random.seed(42)
    
    best_x = None
    best_score = -1
    
    # WalkSAT-style local search
    # State: x = list of 0/1 for each of the 120 edges
    # Objective: maximize edge_count(x) subject to:
    #   1. Triangle-free (hard)
    #   2. Not 3-colorable (hard)
    
    # Strategy: 
    # - Start with random maximal triangle-free graphs
    # - Use local search: flip edges, accept if still valid and improves, 
    #   or with some probability accept non-improving moves (simulated annealing style)
    
    def random_maximal_tf():
        """Generate a random maximal triangle-free graph."""
        x = [0] * n_pairs
        perm = list(range(n_pairs))
        random.shuffle(perm)
        for i in perm:
            u, v = all_pairs[i]
            # Check if adding edge i creates a triangle
            ok = True
            for (w, iw, jw) in witnesses[i]:
                if x[iw] and x[jw]:
                    ok = False
                    break
            if ok:
                x[i] = 1
        return x
    
    def try_improve(x, time_limit):
        """Local search to improve x. Returns best found."""
        best_x_local = x[:]
        best_score_local = edge_count(x)
        
        # Temperature for simulated annealing
        temp = 2.0
        cool_rate = 0.999
        
        while time.time() < time_limit:
            # Pick a random edge to flip
            i = random.randint(0, n_pairs - 1)
            new_x = x[:]
            new_x[i] = 1 - new_x[i]
            
            # Check triangle-freeness
            if not is_triangle_free(new_x):
                x = new_x  # Still flip for exploration? No, let's be smarter
                # Actually for WalkSAT, we should only consider valid states
                # Let's instead: if flipping creates violation, try to fix
                # Simpler: only accept valid moves
                continue
            
            # Check 3-colorability
            if is_3_colorable(new_x):
                continue
            
            new_score = edge_count(new_x)
            
            if new_score >= best_score_local:
                x = new_x
                best_x_local = new_x[:]
                best_score_local = new_score
            else:
                # Simulated annealing: accept with probability exp((new-old)/T)
                delta = new_score - best_score_local
                if random.random() < min(1.0, pow(2.718281828, delta / temp)):
                    x = new_x
            temp *= cool_rate
        
        return best_x_local, best_score_local
    
    # Phase 1: Generate many random maximal TF graphs, check non-3-colorability
    for trial in range(5000):
        if time.time() > deadline - 30:
            break
        
        x = random_maximal_tf()
        
        if is_triangle_free(x) and not is_3_colorable(x):
            score = edge_count(x)
            if score > best_score:
                best_score = score
                best_x = x[:]
    
    if best_x is None:
        # Fallback: try harder
        for trial in range(10000):
            if time.time() > deadline - 10:
                break
            x = random_maximal_tf()
            if is_triangle_free(x) and not is_3_colorable(x):
                score = edge_count(x)
                if score > best_score:
                    best_score = score
                    best_x = x[:]
    
    if best_x is None:
        # Last resort: return empty or a known small example
        # The 5-cycle blow-up or similar
        # For now, try one more time with different seed
        random.seed(123)
        for trial in range(10000):
            if time.time() > deadline - 5:
                break
            x = random_maximal_tf()
            if is_triangle_free(x) and not is_3_colorable(x):
                score = edge_count(x)
                if score > best_score:
                    best_score = score
                    best_x = x[:]
    
    if best_x is None:
        # Return a trivial non-3-colorable triangle-free graph
        # Odd cycle C5 is not 3-colorable? No, C5 is 3-colorable.
        # C7 is not 3-colorable? No, odd cycles > 3 need 3 colors but are 3-colorable.
        # Actually all odd cycles are 3-colorable.
        # The smallest triangle-free non-3-colorable graph is the Grötzsch graph? No.
        # Mycielski of C5 = Grötzsch graph, 11 vertices, triangle-free, 4-chromatic.
        # Let me just return an empty list as fallback (shouldn't happen)
        return []
    
    # Phase 2: Local search improvement on best graph
    if time.time() < deadline - 5:
        improved_x, improved_score = try_improve(best_x, deadline - 5)
        if improved_score > best_score:
            best_x = improved_x
            best_score = improved_score
    
    # Convert to edge list
    result = []
    for i in range(n_pairs):
        if best_x[i]:
            u, v = all_pairs[i]
            result.append([u, v])
    
    return result
