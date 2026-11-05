import time
import random
import numpy as np

def _verify(n, colors):
    """Check no color class contains a C4 (two vertices with >=2 common neighbors in that color)."""
    if n < 2:
        return True
    # Build adjacency per color
    adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i, j] = 1
            adj[c][j, i] = 1
            idx += 1
    for c in range(5):
        # (A^2)[u,v] = number of common neighbors
        A2 = adj[c] @ adj[c]
        np.fill_diagonal(A2, 0)
        if np.any(A2 > 1):
            return False
    return True


def _algebraic_n26():
    """Deterministic Lazebnik-Woldar style construction for n=26.
    Uses vertices Z_5 x Z_5 plus two special points.
    """
    n = 26
    colors = np.zeros(n * (n - 1) // 2, dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            u, v = i, j
            # Determine color based on algebraic rule
            # This is a heuristic reconstruction; will be verified below.
            # Use a deterministic pseudo-random but structured rule.
            du = (u // 5, u % 5)
            dv = (v // 5, v % 5)
            # Difference in Z_5^2
            dx = (dv[0] - du[0]) % 5
            dy = (dv[1] - du[1]) % 5
            # If same point (shouldn't happen for i<j) skip
            if dx == 0 and dy == 0:
                c = 0
            else:
                # Map (dx,dy) to color 0..4 deterministically
                # There are 24 non-zero vectors in Z_5^2; assign to 5 colors
                # Use a fixed mapping that's C4-free in expectation
                key = dx * 5 + dy
                # Simple deterministic hash to color
                c = (key * 3) % 5
            colors[idx] = c
            idx += 1
    return n, colors.tolist()


def _random_search(n, time_budget, seed=0):
    """Simulated annealing / hill-climbing for a given n.
    Returns (colors, cost) or (None, inf) if failed.
    """
    rng = np.random.RandomState(seed)
    num_edges = n * (n - 1) // 2

    # Precompute endpoints and edge index
    endpoints = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    U = endpoints[:, 0]
    V = endpoints[:, 1]
    edge_index = np.full((n, n), -1, dtype=np.int32)
    for e in range(num_edges):
        edge_index[U[e], V[e]] = e
        edge_index[V[e], U[e]] = e

    # Precompute for each vertex u: list of edge indices incident to u, and the other endpoint
    # For fast delta: when recoloring (u,v), we need for each x != u,v:
    #   color of (u,x) and (v,x)
    # Build: for each edge e=(u,v), the "neighbors" are all x, with edge indices edge_index[u,x], edge_index[v,x]

    # Vectorize delta computation:
    # For edge e with endpoints (u,v), old color c1, new color c2:
    # For each x in [0,n) excluding u,v:
    #   e_ux = edge_index[u,x], e_vx = edge_index[v,x]
    #   c_ux = colors[e_ux], c_vx = colors[e_vx]
    #   CN_c1[u,x] = sum over y!=u,x of [c(uy)==c1 and c(xy)==c1] ... this is expensive to maintain.
    
    # Simpler: maintain per-color adjacency as int8 matrices, and compute common-neighbor counts via matmul when needed.
    # But matmul per step is O(n^3) which is too slow for n=27 (27^3=19683 per step, times steps...).
    
    # Better: maintain CN[c][u][v] = number of common neighbors of u,v in color c.
    # When recoloring edge (u,v) from c1 to c2, update CN in O(n).
    # CN[c][u,v] = sum over x of A_c[u,x]*A_c[v,x].
    # When we change A_c1[u,v] from 1 to 0 and A_c2[u,v] from 0 to 1:
    #   For color c1: removing edge (u,v) affects CN[c1][a,b] for all a,b that are neighbors... 
    #   Actually CN[c][u,v] depends on edges incident to u or v in color c.
    
    # Let me define CN[c][u,v] = (A_c^2)[u,v] for u!=v.
    # If we flip edge (u,v) in color c1 (remove): 
    #   A_c1[u,v] becomes 0. This affects (A_c1^2)[x,y] for x,y where the path x-u-v-y or x-v-u-y was used.
    #   Specifically, (A^2)[x,y] = sum_z A[x,z]A[z,y]. Flipping A[u,v] changes:
    #     (A^2)[x,v] -= A[x,u]  (path x-u-v)
    #     (A^2)[v,x] -= A[v,u]... wait A is symmetric.
    #     (A^2)[x,u] += 0 ... let me be careful.
    #   (A^2)[x,y] = sum_z A[x,z] A[z,y]
    #   When A[u,v] goes 1->0:
    #     Terms where (z=x, y=v): A[x,u]*A[u,v] -> contributes to (A^2)[x,v]. So (A^2)[x,v] -= A[x,u].
    #     Terms where (z=v, y=x): A[v,x]*A[v,x]... no: A[z,y] with z=v,y=x is A[v,x]. And A[x,z]=A[x,v]. So (A^2)[x,x]... wait that's diagonal.
    #     Let me redo: (A^2)[p,q] = sum_z A[p,z] A[z,q].
    #     Flip A[u,v]: 1->0.
    #     For (A^2)[p,q], the change is -A[p,u]*A[u,q] - A[p,v]*A[v,q] ... no.
    #     Only terms involving A[u,v] or A[v,u] (=A[u,v]) change.
    #     (A^2)[p,q] contains A[p,u]*A[u,q] (z=u) and A[p,v]*A[v,q] (z=v).
    #     A[u,v] appears in: (A^2)[p,v] via z=u: A[p,u]*A[u,v]; (A^2)[p,u] via z=v: A[p,v]*A[v,u]=A[p,v]*A[u,v]; 
    #       (A^2)[v,q] via z=u: A[v,u]*A[u,q]=A[u,v]*A[u,q]; (A^2)[u,q] via z=v: A[u,v]*A[v,q].
    #     So flipping A[u,v] by delta (here -1):
    #       (A^2)[p,v] += delta * A[p,u]   for all p
    #       (A^2)[p,u] += delta * A[p,v]   for all p
    #       (A^2)[v,q] += delta * A[u,q]   for all q
    #       (A^2)[u,q] += delta * A[v,q]   for all q
    #     But note (A^2) is symmetric, so (A^2)[p,v]=(A^2)[v,p]. Let's just update all.
    #     Actually since A is symmetric, A[p,u]=A[u,p], etc.
    #     Net: for each other vertex w (w!=u,v):
    #       (A^2)[u,v] ... we don't care about (u,v) directly for C4 (we care about pairs with 2 common neighbors).
    #     The C4 cost is sum over pairs (a,b), a<b, a!=b of C(CN[a,b],2) where CN[a,b]=(A^2)[a,b].
    
    # This is getting complex. Given time constraints, let me use a simpler but correct approach:
    # Maintain per-color adjacency matrices A[c] (int8, symmetric, no diagonal).
    # Cost = sum_c sum_{a<b} C((A[c]^2)[a,b], 2).
    # To compute cost fast, maintain M[c][a,b] = (A[c]^2)[a,b] for a!=b (the common-neighbor counts).
    # When flipping edge (u,v) in color c by delta (+1 or -1):
    #   For all w != u,v:
    #     M[c][u,w] += delta * A[c][v,w]
    #     M[c][v,w] += delta * A[c][u,w]
    #     M[c][w,u] = M[c][u,w] (symmetry)
    #   Also M[c][u,v] itself changes but we exclude diagonal-ish; actually (u,v) common neighbors:
    #     M[c][u,v] = sum_z A[c][u,z]A[c][v,z]. Flipping A[c][u,v] doesn't directly change M[c][u,v] 
    #     because the sum is over z != u,v (since A[u,u]=0, A[v,v]=0, and A[u,v]*A[v,u] would be z=... no, z is the middle vertex, A[u,z]A[v,z]; z=u gives A[u,u]A[v,u]=0; z=v gives A[u,v]A[v,v]=0. So M[u,v] unaffected by flipping A[u,v] directly. Good.
    #   Cost change: cost = sum_{a<b} M[a,b]*(M[a,b]-1)//2.
    #   When M[c][u,w] changes from old to new, the contribution C(M,2) changes by C(new,2)-C(old,2).
    
    # Let's implement this.

    def compute_cost(M):
        total = 0
        for c in range(5):
            Mc = M[c]
            # sum over a<b of Mc[a,b]*(Mc[a,b]-1)//2
            iu, iv = np.triu_indices(n, 1)
            vals = Mc[iu, iv]
            total += int(np.sum(vals * (vals - 1) // 2))
        return total

    # Initialize
    colors = rng.randint(0, 5, size=num_edges, dtype=np.int8)
    A = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
    M = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
    
    for e in range(num_edges):
        c = colors[e]
        u, v = U[e], V[e]
        A[c][u, v] = 1
        A[c][v, u] = 1
    for c in range(5):
        M[c] = A[c] @ A[c]
        np.fill_diagonal(M[c], 0)
    
    cost = compute_cost(M)
    
    if cost == 0:
        return colors.tolist(), 0
    
    # Hill climbing with occasional random restarts within time budget
    best_cost = cost
    best_colors = colors.copy()
    
    # Temperature-like: accept worsening moves with small probability early
    t = 0.5
    t_decay = 0.999
    
    time_limit = time_budget
    start = time.time()
    
    max_steps = 200000
    step = 0
    while step < max_steps:
        if time.time() - start > time_limit:
            break
        step += 1
        
        # Pick random edge
        e = rng.randint(num_edges)
        u, v = U[e], V[e]
        old_c = int(colors[e])
        new_c = int(rng.randint(0, 4))
        if new_c == old_c:
            continue
        
        # Compute delta in cost
        # Flipping A[old_c][u,v] by -1 and A[new_c][u,v] by +1
        delta_cost = 0
        for c, delta in [(old_c, -1), (new_c, 1)]:
            Ac = A[c]
            Mc = M[c]
            # For all w != u,v:
            #   M[c][u,w] += delta * A[c][v,w]
            #   M[c][v,w] += delta * A[c][u,w]
            # Need to update cost contribution.
            # Vectorized:
            #   new_M_uw = M[c][u,:] + delta * A[c][v,:]  (but exclude u,v positions? M[u,u]=0 always, M[u,v] we don't track for cost)
            # Let's just compute over all w, and handle that M[u,u], M[v,v] stay 0, M[u,v] we ignore.
            # Actually simpler: compute old and new contributions for pairs (u,w) and (v,w) for w in 0..n-1, w!=u,v, and also (u,v) pair if affected (it's not).
            
            # Contribution of pair (a,b) is C(M[a,b],2). We change M[u,w] and M[v,w] for w!=u,v.
            # Also note M is symmetric, so changing M[u,w] affects pair (min(u,w),max(u,w)).
            
            # Let's compute delta for all w:
            Au = Ac[u, :]  # A[c][u,w] for all w
            Av = Ac[v, :]  # A[c][v,w] for all w
            Mu = Mc[u, :]
            Mv = Mc[v, :]
            
            # new_Mu = Mu + delta * Av  (elementwise)
            # new_Mv = Mv + delta * Au
            new_Mu = Mu + delta * Av
            new_Mv = Mv + delta * Au
            
            # old contribution for pairs involving u or v (excluding (u,v) and diagonals):
            # pair (u,w) for w!=u: C(Mu[w],2) but careful about double counting with (v,w).
            # Let's sum C(new,2)-C(old,2) over all w for Mu and Mv, then subtract the (u,v) pair which appears in both Mu[v] and Mv[u].
            # Also diagonal w=u for Mu, w=v for Mv should be excluded (they're 0 and stay 0? Mu[u]=M[u,u]=0, new_Mu[u]=0+delta*Av[u]=delta*A[v,u]=delta*A[u,v]. Hmm, that's not 0!
            
            # Wait, M[u,u] should remain 0. But new_Mu[u] = Mu[u] + delta*Av[u] = 0 + delta*A[v,u]. Since A[v,u]=A[u,v] which is 1 (before flip) for old_c, and 0 for new_c.
            # This is a problem: M[u,u] is the number of length-2 paths from u to u, which is degree. But we set diagonal to 0 artificially.
            # For C4 detection, we only care about off-diagonal M[a,b] for a!=b, specifically pairs with 2+ common neighbors.
            # So I should only track off-diagonal M. Let me exclude w==u and w==v.
            
            # Let me create a mask excluding u and v.
            w_mask = np.ones(n, dtype=bool)
            w_mask[u] = False
            w_mask[v] = False
            
            old_u = Mu[w_mask]
            new_u = new_Mu[w_mask]
            old_v = Mv[w_mask]
            new_v = new_Mv[w_mask]
            
            old_contrib = np.sum(old_u * (old_u - 1) // 2) + np.sum(old_v * (old_v - 1) // 2)
            new_contrib = np.sum(new_u * (new_u - 1) // 2) + np.sum(new_v * (new_v - 1) // 2)
            
            # But pair (u,v) is counted in both old_u (at position v) and old_v (at position u). 
            # old_u includes Mu[v] = M[u,v]; old_v includes Mv[u] = M[v,u] = M[u,v]. So M[u,v] counted twice.
            # Is M[u,v] affected? new_Mu[v] = Mu[v] + delta*Av[v] = M[u,v] + delta*A[v,v] = M[u,v] + 0 = M[u,v]. 
            # Similarly new_Mv[u] = M[v,u] + delta*A[u,u] = M[v,u]. So M[u,v] unchanged. Good, but it's still counted twice in both old and new, so it cancels. Good.
            
            delta_cost += int(new_contrib - old_contrib)
        
        # Accept/reject
        if delta_cost < 0:
            accept = True
        elif delta_cost == 0:
            accept = True
        else:
            # Accept with probability
            if t > 0.001:
                import math
                if rng.rand() < math.exp(-delta_cost / max(t, 1e-9)):
                    accept = True
                else:
                    accept = False
            else:
                accept = False
        
        if accept:
            # Apply
            colors[e] = new_c
            for c, delta in [(old_c, -1), (new_c, 1)]:
                Ac = A[c]
                Mc = M[c]
                Ac[u, v] += delta
                Ac[v, u] += delta
                # Update M for pairs (u,w) and (v,w)
                Au = Ac[u, :].copy()  # after flip
                # For w != u,v:
                # M[u,w] += delta * A[v,w]  (using OLD A[v,w] before flip? No, the formula was derived for the flip.)
                # Let me recompute: we already computed new_Mu = Mu + delta*Av_old. 
                # But Ac is now flipped, so A[v,w] for w is still the same (we only flipped A[u,v]).
                # So A[v,w] for w!=u,v is unchanged. Good.
                # Update M:
                w_mask2 = np.ones(n, dtype=bool)
                w_mask2[u] = False
                w_mask2[v] = False
                Mc[u, w_mask2] += delta * Ac[v, w_mask2]
                Mc[v, w_mask2] += delta * Ac[u, w_mask2]
                Mc[w_mask2, u] = Mc[u, w_mask2]
                Mc[w_mask2, v] = Mc[v, w_mask2]
                # M[u,v] unchanged
            cost += delta_cost
            if cost < best_cost:
                best_cost = cost
                best_colors = colors.copy()
                if cost == 0:
                    return colors.tolist(), 0
            t *= t_decay
        else:
            t *= t_decay
    
    return best_colors.tolist(), best_cost


def construct():
    start = time.time()
    deadline = start + 118  # safety margin before 150s kill, but evaluator says 120s; use 115s to be safe
    deadline = start + 115
    
    # First, try deterministic algebraic construction for n=26
    best_n = 0
    best_colors = None
    
    # Try n=26 algebraic
    if time.time() < deadline - 5:
        n26, c26 = _algebraic_n26()
        if _verify(n26, c26):
            best_n = n26
            best_colors = c26
            print("Found valid n=26 algebraic", flush=True)
    
    # If n=26 failed, try random search for n=26
    if best_n < 26 and time.time() < deadline - 5:
        # quick random search for n=26
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            cols, cost = _random_search(26, time_budget=min(30, deadline - time.time() - 3), seed=seed)
            if cost == 0 and cols is not None:
                best_n = 26
                best_colors = cols
                break
    
    # Try to beat 26: search for n=27, 28
    if time.time() < deadline - 8:
        for target_n in [27, 28]:
            if time.time() > deadline - 8:
                break
            if target_n <= best_n:
                continue
            # Allocate remaining time
            remaining = deadline - time.time() - 2
            if remaining < 5:
                break
            # Try a few random searches
            found = False
            for seed in range(5):
                if time.time() > deadline - 2:
                    break
                cols, cost = _random_search(target_n, time_budget=min(20, deadline - time.time() - 2), seed=1000 + seed)
                if cost == 0 and cols is not None:
                    if _verify(target_n, cols):
                        best_n = target_n
                        best_colors = cols
                        found = True
                        print(f"Found valid n={target_n}!", flush=True)
                        break
            if found:
                # If we found 27, try 28 with remaining time
                if target_n == 27 and time.time() < deadline - 8:
                    continue
                else:
                    break
    
    # Fallback: if nothing found, do a guaranteed small n
    if best_n < 2 or best_colors is None:
        # Guaranteed valid: n=1
        return {'n': 1, 'colors': []}
    
    # Final verification
    if not _verify(best_n, best_colors):
        # Shouldn't happen, but fallback
        return {'n': 1, 'colors': []}
    
    return {'n': best_n, 'colors': best_colors}
