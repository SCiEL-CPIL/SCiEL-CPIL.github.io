import time
import random
import math

def has_k33_fast(adj_bitmasks):
    n = len(adj_bitmasks)
    for i in range(n):
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            common_ij = ai & adj_bitmasks[j]
            if common_ij.bit_count() < 3:
                continue
            for k in range(j + 1, n):
                common_ijk = common_ij & adj_bitmasks[k]
                if common_ijk.bit_count() >= 3:
                    return False
    return True

def is_add_valid(adj_bitmasks, u, v):
    """
    Check if adding edge (u,v) keeps graph K3,3-free.
    Only triples containing u or v can become newly invalid.
    Specifically, we only need to check common neighbors of (u,v) that are NEW 
    or interactions with existing neighbors.
    
    A K3,3 is formed by {u, x, y} and {v, a, b} if:
    1. v is connected to x, y, a, b (and u is connected to a, b? No, u is not connected to v yet).
    Actually, adding (u,v) creates new K3,3s only if there exist sets {x,y} and {a,b} such that:
    - u is connected to x,y
    - v is connected to a,b
    - x,y are connected to a,b? No.
    
    Standard K3,3 check: 
    A K3,3 exists if there are 3 vertices A={a1,a2,a3} and 3 vertices B={b1,b2,b3} such that all 9 edges exist.
    
    When adding (u,v), the new K3,3s must include the edge (u,v).
    So u and v must be in different partitions of the K3,3.
    Case 1: u is in A, v is in B.
    We need 2 other vertices x,y in A (neighbors of u) and 2 other vertices a,b in B (neighbors of v).
    Conditions:
    - x,y are neighbors of u.
    - a,b are neighbors of v.
    - x,y are neighbors of a,b. (i.e., x~a, x~b, y~a, y~b)
    - u is not connected to a,b (already true, u not connected to v, but u might be connected to a or b? 
      If u is connected to a, then u,a,v form a triangle? No, simple graph.
      The definition of K3,3 bipartite graph requires no edges within A and no edges within B.
      Wait, the problem says "no three vertices have three common neighbours".
      This is equivalent to K3,3-free.
      Let the 3 vertices be C={c1,c2,c3} and their common neighbors be D={d1,d2,d3}.
      Then C and D form a K3,3.
      
      So, adding (u,v) creates a K3,3 if there exists a set C of 3 vertices and D of 3 vertices 
      such that (u,v) is one of the 9 edges.
      WLOG, u in C, v in D.
      Then C = {u, c2, c3} and D = {v, d2, d3}.
      Requirements:
      1. c2, c3 are neighbors of u.
      2. d2, d3 are neighbors of v.
      3. c2, c3 are neighbors of d2, d3. (i.e., c2~d2, c2~d3, c3~d2, c3~d3)
      4. u is NOT connected to d2, d3? No, u is in C, d2,d3 in D. In K3,3, all cross edges exist.
         So u MUST be connected to d2 and d3.
         But we are ADDING (u,v). u is already connected to d2,d3?
         If u is connected to d2, then u~d2 exists.
         So the condition is:
         - u is connected to d2, d3.
         - v is connected to c2, c3.
         - c2, c3 are connected to d2, d3.
         - u is NOT connected to v (before addition).
         
      So, to check validity of adding (u,v):
      Let N(u) be neighbors of u, N(v) be neighbors of v.
      We need to find c2,c3 in N(u) and d2,d3 in N(v) such that:
      - d2,d3 are in N(u) as well? No.
      Let's re-read carefully.
      K3,3 bipartition A={u, c2, c3}, B={v, d2, d3}.
      Edges required:
      u-v (being added)
      u-d2, u-d3 (must exist)
      c2-v, c3-v (must exist)
      c2-d2, c2-d3, c3-d2, c3-d3 (must exist)
      
      So:
      1. d2, d3 must be in N(u) AND N(v). (Common neighbors of u and v? No, d2 is in B, so u~d2 and v~d2? 
         Yes, u~d2 and v~d2. So d2 is a common neighbor of u and v.
         Similarly d3 is a common neighbor of u and v.
         Wait, if d2 is a common neighbor of u and v, then u~d2 and v~d2.
         And c2 is in A, so c2~v and c2~d2.
         Also c2~u.
         So c2 is a common neighbor of u and v? No.
         c2~u and c2~v. So c2 is ALSO a common neighbor of u and v.
         
      So, if we add (u,v), a K3,3 is formed if there exist 4 vertices x,y,z,w such that:
      - x,y are common neighbors of u and v (so x~u, x~v, y~u, y~v).
      - z,w are common neighbors of u and v?
      Let's check the structure again.
      A={u, c2, c3}, B={v, d2, d3}.
      c2, c3 are in A. They are connected to v, d2, d3.
      They are NOT connected to u (within A). But c2~u is required?
      No! In a bipartite graph, there are no edges within A.
      So c2 is NOT connected to u.
      But the problem statement says "no three vertices have three common neighbours".
      This implies the graph is simple.
      If c2 is in A and u is in A, they are not connected.
      So c2 is NOT a neighbor of u.
      
      Let's re-evaluate the condition "no three vertices have three common neighbours".
      Let the 3 vertices be X={x1,x2,x3}.
      Let their common neighbors be Y={y1,y2,y3}.
      This means xi~yj for all i,j.
      This forms a K3,3 between X and Y.
      In this K3,3, there are no edges within X and no edges within Y.
      
      So, when adding (u,v), if u and v are in different parts, say u in X, v in Y.
      Then we need 2 other vertices in X, say x2, x3.
      And 2 other vertices in Y, say y2, y3.
      Conditions:
      1. u is not connected to x2, x3 (within X).
      2. v is not connected to y2, y3 (within Y).
      3. u is connected to y2, y3 (cross).
      4. x2, x3 are connected to v (cross).
      5. x2, x3 are connected to y2, y3 (cross).
      
      So:
      - x2, x3 are neighbors of v.
      - x2, x3 are NOT neighbors of u.
      - y2, y3 are neighbors of u.
      - y2, y3 are NOT neighbors of v.
      - x2, x3 are connected to y2, y3.
      
      So, to check validity of adding (u,v):
      Let N(u) be neighbors of u, N(v) be neighbors of v.
      Let C_u = N(u) \ N(v) (neighbors of u not v)
      Let C_v = N(v) \ N(u) (neighbors of v not u)
      Let C_both = N(u) & N(v) (common neighbors)
      
      We need to find x2, x3 in C_v and y2, y3 in C_u such that:
      - x2, x3 are connected to y2, y3.
      
      This is a much smaller search space!
      """
    n = len(adj_bitmasks)
    adj_u = adj_bitmasks[u]
    adj_v = adj_bitmasks[v]
    
    # Common neighbors
    common = adj_u & adj_v
    
    # Candidates for X (other than u): neighbors of v, not u, not v
    # They must be in N(v) and NOT in N(u)
    # And they must be connected to Y.
    # Candidates for Y (other than v): neighbors of u, not v, not u
    # They must be in N(u) and NOT in N(v)
    
    # We need to pick 2 from X_cands and 2 from Y_cands
    # such that all 4 cross edges exist between X_cands and Y_cands.
    
    x_cands = []
    m = adj_v & ~adj_u
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        if i != u and i != v:
            x_cands.append(i)
        m -= lsb
        
    y_cands = []
    m = adj_u & ~adj_v
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        if i != u and i != v:
            y_cands.append(i)
        m -= lsb
        
    if len(x_cands) < 2 or len(y_cands) < 2:
        return True
        
    # Check all pairs in x_cands and y_cands
    # For each pair (x1, x2) in x_cands, check if they share 2+ common neighbors in y_cands
    # Actually, we need x1~y1, x1~y2, x2~y1, x2~y2.
    # This means y1 and y2 must be in N(x1) & N(x2).
    # And y1, y2 must be in y_cands.
    
    for i in range(len(x_cands)):
        x1 = x_cands[i]
        for j in range(i+1, len(x_cands)):
            x2 = x_cands[j]
            # Common neighbors of x1 and x2
            common_x1_x2 = adj_bitmasks[x1] & adj_bitmasks[x2]
            # Intersect with y_cands
            # We need at least 2 vertices in y_cands that are in common_x1_x2
            # Let's count how many y_cands are in common_x1_x2
            count = 0
            for y in y_cands:
                if common_x1_x2 & (1 << y):
                    count += 1
                    if count >= 2:
                        return False
    return True

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0

    best_edges = 0
    best_adj_bitmasks = None

    def count_edges(bitmasks):
        return sum(b.bit_count() for b in bitmasks) // 2

    def bitmasks_to_edges(bitmasks):
        edges = []
        for i in range(n):
            m = bitmasks[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    def get_non_edges(bitmasks):
        non_edges = []
        for i in range(n):
            mi = bitmasks[i]
            for j in range(i + 1, n):
                if not (mi & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    def get_all_edges(bitmasks):
        all_edges = []
        for i in range(n):
            m = bitmasks[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    all_edges.append((i, j))
                m -= lsb
        return all_edges

    def greedy_add_exhaustive(bitmasks):
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(bitmasks)
            for u, v in non_edges:
                if is_add_valid(bitmasks, u, v):
                    bitmasks[u] |= (1 << v)
                    bitmasks[v] |= (1 << u)
                    changed = True
        return bitmasks

    def local_search(initial_bitmasks, time_limit=10.0):
        bitmasks = initial_bitmasks[:]
        current_edges = count_edges(bitmasks)
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]

        bitmasks = greedy_add_exhaustive(bitmasks)
        current_edges = count_edges(bitmasks)
        if current_edges > best_local_edges:
            best_local_edges = current_edges
            best_local_bitmasks = bitmasks[:]

        no_improve = 0
        max_no_improve = 50
        t0 = time.time()
        while time.time() < t0 + time_limit and time.time() < deadline and no_improve < max_no_improve:
            moved = False

            # 1. Try adding edges
            non_edges = get_non_edges(bitmasks)
            random.shuffle(non_edges)
            for u, v in non_edges:
                if is_add_valid(bitmasks, u, v):
                    bitmasks[u] |= (1 << v)
                    bitmasks[v] |= (1 << u)
                    current_edges += 1
                    moved = True
                    if current_edges > best_local_edges:
                        best_local_edges = current_edges
                        best_local_bitmasks = bitmasks[:]

            if not moved:
                # 2. Try swap moves: remove one edge, add one different edge
                # Strategy: Find a "conflict" pair (u,v) with 3+ common neighbors.
                # Try to break it by removing an edge incident to u or v and adding a non-edge.
                # Simpler: Random swap but with validation.
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if all_edges and non_edges:
                    for _ in range(100):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
                        bm2 = bitmasks[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        if is_add_valid(bm2, c, d):
                            bm2[c] |= (1 << d)
                            bm2[d] |= (1 << c)
                            if has_k33_fast(bm2):
                                bitmasks = bm2
                                current_edges = count_edges(bitmasks)
                                moved = True
                                if current_edges > best_local_edges:
                                    best_local_edges = current_edges
                                    best_local_bitmasks = bitmasks[:]
                                break

            if not moved:
                no_improve += 1
            else:
                no_improve = 0

        return best_local_bitmasks, best_local_edges

    # Strategy 1: Probabilistic initialization with repair
    densities = [0.3, 0.4, 0.5, 0.6]
    for r in range(50):
        if time.time() > deadline:
            break
        
        p = densities[r % len(densities)]
        init = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    init[i] |= (1 << j)
                    init[j] |= (1 << i)
        
        # Repair by removing edges involved in K3,3
        # Detect K3,3 and remove a random edge from it
        attempts = 0
        while not has_k33_fast(init) and attempts < 1000:
            # Find a K3,3
            found = False
            for i in range(n):
                ai = init[i]
                for j in range(i + 1, n):
                    common_ij = ai & init[j]
                    if common_ij.bit_count() < 3:
                        continue
                    for k in range(j + 1, n):
                        common_ijk = common_ij & init[k]
                        if common_ijk.bit_count() >= 3:
                            # Found K3,3 with vertices i,j,k and common neighbors
                            # Remove a random edge incident to i, j, or k
                            edges_to_remove = []
                            for v in [i, j, k]:
                                m = init[v]
                                while m:
                                    lsb = m & -m
                                    w = lsb.bit_length() - 1
                                    if w != v:
                                        edges_to_remove.append((v, w))
                                    m -= lsb
                            if edges_to_remove:
                                u, v = random.choice(edges_to_remove)
                                init[u] &= ~(1 << v)
                                init[v] &= ~(1 << u)
                                found = True
                                break
                        if found: break
                if found: break
            if not found:
                # Should not happen if has_k33_fast is consistent, but safety break
                break
            attempts += 1

        if not has_k33_fast(init):
            continue

        local_best_bm, local_best_e = local_search(init, time_limit=8.0)
        if local_best_e > best_edges:
            best_edges = local_best_e
            best_adj_bitmasks = local_best_bm[:]

    # Final greedy pass and polish
    if best_adj_bitmasks is not None:
        improved = greedy_add_exhaustive(best_adj_bitmasks[:])
        improved_e = count_edges(improved)
        if improved_e > best_edges:
            best_edges = improved_e
            best_adj_bitmasks = improved

        if time.time() < deadline - 5:
            final_bm, final_e = local_search(best_adj_bitmasks[:], time_limit=30.0)
            if final_e > best_edges:
                best_edges = final_e
                best_adj_bitmasks = final_bm

    if best_adj_bitmasks is None:
        return []

    return bitmasks_to_edges(best_adj_bitmasks)

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Found {len(edges)} edges in {t1-t0:.2f}s")
    n = 20
    adj = [0]*n
    for u,v in edges:
        adj[u] |= (1<<v)
        adj[v] |= (1<<u)
    valid = True
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                if (adj[i]&adj[j]&adj[k]).bit_count() >= 3:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid K3,3-free: {valid}")
