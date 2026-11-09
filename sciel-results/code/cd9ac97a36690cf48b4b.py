import random
import time
import sys

_START_TIME = time.time()
DEADLINE = 115.0  # Leave 5s margin for 120s limit

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _is_k4_free(adj, n=18):
    """Check if graph is K4-free using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _greedy_max_tf_induced(adj, n=18):
    """
    Fast greedy heuristic for largest triangle-free induced subgraph.
    Returns the size of the set found.
    """
    best_size = 0
    # Try multiple random orderings or greedy strategies
    # Strategy: Pick vertex with min degree in remaining, or min conflicts
    
    def try_greedy(order):
        current_mask = 0
        forbidden_mask = 0
        for v in order:
            bit = 1 << v
            if forbidden_mask & bit:
                continue
            # Check if v can be added
            # v creates a triangle if it is connected to u in current_mask 
            # and there is a w in current_mask connected to both.
            # Actually, if we add v, we need to check if any existing u in current_mask
            # is connected to v, and if so, are there any w in current_mask 
            # that are common neighbors of u and v?
            # If yes, we cannot add v.
            can_add = True
            cm = current_mask
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                if (adj[v] >> u) & 1:
                    # v connected to u
                    # Check if any common neighbor of u and v is in current_mask
                    if (adj[v] & adj[u]) & current_mask:
                        can_add = False
                        break
            if can_add:
                current_mask |= bit
                # Update forbidden
                # For any u in current_mask (including new v) connected to v,
                # common neighbors of u and v are forbidden.
                cm2 = current_mask
                while cm2:
                    lsb = cm2 & -cm2
                    u = lsb.bit_length() - 1
                    cm2 -= lsb
                    if (adj[v] >> u) & 1:
                        forbidden_mask |= (adj[v] & adj[u])
        
        return current_mask.bit_count()

    # Generate a few random orders
    best = 0
    for _ in range(5):
        order = list(range(n))
        random.shuffle(order)
        # Sort by degree descending? Or ascending?
        # Ascending degree might allow more vertices.
        order.sort(key=lambda x: adj[x].bit_count())
        sz = try_greedy(order)
        if sz > best:
            best = sz
            
    # Also try natural order
    sz = try_greedy(list(range(n)))
    if sz > best:
        best = sz
        
    return best

def _max_triangle_free_induced(adj, n=18):
    """
    Corrected Branch and Bound for largest triangle-free induced subgraph.
    Returns the size of the largest subset S such that G[S] is triangle-free.
    """
    best = [0]
    
    # Precompute common neighbors
    common = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = adj[u] & adj[v]
            common[u][v] = c
            common[v][u] = c
            
    def bb(idx, current_mask, forbidden_mask):
        if time.time() - _START_TIME > DEADLINE:
            return
            
        # Upper bound
        # We can add at most all remaining vertices not forbidden
        # But a tighter bound: 
        # The remaining vertices form a subgraph. The max triangle-free induced subgraph 
        # of that subgraph is an upper bound. But calculating that is as hard as the original problem.
        # Simple bound: count available vertices.
        remaining_count = 0
        for i in range(idx, n):
            if not (forbidden_mask >> i) & 1:
                remaining_count += 1
                
        cur_sz = current_mask.bit_count()
        if cur_sz + remaining_count <= best[0]:
            return
            
        if idx == n:
            if cur_sz > best[0]:
                best[0] = cur_sz
            return
            
        v = idx
        bit = 1 << v
        
        # Option 1: Include v
        if not (forbidden_mask & bit):
            # Check if v creates a triangle with existing vertices in current_mask
            can_include = True
            cm = current_mask
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                if (adj[v] >> u) & 1:
                    # v is connected to u. Check if any common neighbor of v and u is in current_mask
                    if (common[v][u] & current_mask):
                        can_include = False
                        break
                        
            if can_include:
                new_forbidden = forbidden_mask
                cm = current_mask
                while cm:
                    lsb = cm & -cm
                    u = lsb.bit_length() - 1
                    cm -= lsb
                    if (adj[v] >> u) & 1:
                        new_forbidden |= common[v][u]
                        
                bb(idx + 1, current_mask | bit, new_forbidden)
                
        # Option 2: Exclude v
        bb(idx + 1, current_mask, forbidden_mask)

    # Greedy initial bound
    greedy_score = _greedy_max_tf_induced(adj, n)
    best[0] = greedy_score
    
    # Run B&B
    bb(0, 0, 0)
    return best[0]

def construct():
    n = 18
    random.seed(42)
    
    # Start with Turan complement (3 disjoint K6 in H, so G is K6,6,6)
    adj_h = [0] * n
    for i in range(3):
        start = i * 6
        for u in range(start, start + 6):
            for v in range(start, start + 6):
                if u != v:
                    adj_h[u] |= (1 << v)
                    adj_h[v] |= (1 << u)
    
    # G is complement of H
    full = (1 << n) - 1
    adj_g = [0] * n
    for v in range(n):
        adj_g[v] = full & ~(adj_h[v] | (1 << v))
        
    current_g = adj_g[:]
    
    # Get initial exact score
    current_exact_score = _max_triangle_free_induced(current_g, n)
    best_g = current_g[:]
    best_exact_score = current_exact_score
    
    print(f"Initial Exact Score: {best_exact_score}", flush=True)
    
    # Hybrid Local Search
    # Use greedy score for fast evaluation, exact score for confirmation
    
    steps = 0
    last_exact_check = 0
    greedy_score = _greedy_max_tf_induced(current_g, n)
    
    while _time_left() > 5 and steps < 200000:
        steps += 1
        
        # Periodically check exact score if greedy improved
        if steps - last_exact_check > 50 or (greedy_score < best_exact_score and steps % 10 == 0):
            # Only run exact if we have time
            if _time_left() > 20:
                exact_s = _max_triangle_free_induced(current_g, n)
                last_exact_check = steps
                if exact_s < best_exact_score:
                    best_exact_score = exact_s
                    best_g = current_g[:]
                    print(f"Improved Exact Score to {best_exact_score} at step {steps}", flush=True)
        
        # Pick a random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        new_g = current_g[:]
        if (new_g[u] >> v) & 1:
            new_g[u] &= ~(1 << v)
            new_g[v] &= ~(1 << u)
        else:
            new_g[u] |= (1 << v)
            new_g[v] |= (1 << u)
            
        if not _is_k4_free(new_g, n):
            continue
            
        # Fast evaluation
        new_greedy_score = _greedy_max_tf_induced(new_g, n)
        
        # Accept if greedy score is better, or same with some probability
        if new_greedy_score < greedy_score:
            current_g = new_g
            greedy_score = new_greedy_score
        elif new_greedy_score == greedy_score and random.random() < 0.5:
            current_g = new_g
        else:
            # Reject
            pass

    # Final exact check on the best found so far
    if _time_left() > 5:
        final_exact = _max_triangle_free_induced(best_g, n)
        if final_exact < best_exact_score:
            best_exact_score = final_exact
            
    # Also check current_g if it's different
    if current_g != best_g and _time_left() > 5:
        cur_exact = _max_triangle_free_induced(current_g, n)
        if cur_exact < best_exact_score:
            best_exact_score = cur_exact
            best_g = current_g[:]

    # Convert best_g to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_g[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
