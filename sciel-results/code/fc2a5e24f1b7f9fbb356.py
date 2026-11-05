"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s to be safe under 60s limit
    N = 512
    n = N

    # Precompute compatibility masks:
    # masks[i] is a bitmask where bit j is 1 if lcm(i+1, j+1) <= N
    masks = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(n):
            b = j + 1
            g = gcd(a, b)
            if a * b // g <= N:
                masks[i] |= (1 << j)

    # Degree of each vertex (number of compatible partners)
    degree = [bin(masks[i]).count('1') for i in range(n)]

    best_size = 0
    best_set = set()
    best_mask = 0

    def update_best(s, m):
        nonlocal best_size, best_set, best_mask
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            best_mask = m

    def greedy_clique(order):
        """Greedy: process vertices in given order, add if compatible with all already added."""
        clique = set()
        mask = 0
        for v in order:
            if (masks[v] & mask) == mask:
                clique.add(v)
                mask |= (1 << v)
        return clique, mask

    def local_search(s, m):
        """Local search: greedy add, then single-swap, then double-swap."""
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Phase 1: greedy add
            while True:
                added = False
                for v in range(n):
                    if v not in s:
                        if (masks[v] & m) == m:
                            s.add(v)
                            m |= (1 << v)
                            added = True
                            break
                if not added:
                    break
            if added:
                improved = True
                update_best(s, m)

            # Phase 2: single-swap (remove u, add v)
            swapped = False
            s_list = list(s)
            for u in s_list:
                if time.time() > deadline:
                    break
                u_bit = 1 << u
                m_wo_u = m ^ u_bit
                for v in range(n):
                    if v in s:
                        continue
                    if (masks[v] & m_wo_u) == m_wo_u:
                        # Valid swap: remove u, add v
                        s.remove(u)
                        s.add(v)
                        m = m_wo_u | (1 << v)
                        swapped = True
                        update_best(s, m)
                        break
                if swapped:
                    break
            if swapped:
                improved = True

        return s, m

    def max_closure_min_cut():
        """
        Maximum Closure via Min-Cut (Max-Flow).
        
        For maximum clique in a graph:
        - Create source s, sink t
        - For each vertex v: edge s->v with weight 1
        - For each non-edge (u,v) where u<v: edge u->v with weight INF and v->u with weight INF
          (if we select u but not v, we pay INF penalty)
        - Min-cut gives: selected = vertices reachable from source in residual graph
        - This finds maximum weight closure where selecting a vertex requires... 
        
        Actually, for maximum clique:
        - We want to select a set S such that all pairs in S are edges.
        - Equivalently, for each non-edge (u,v), we cannot have both u and v in S.
        - This is a maximum independent set in the complement graph.
        - Maximum independent set = maximum closure in a specific construction.
        
        Alternative: Use the fact that if we define for each vertex v a "conflict" set,
        we can use max-flow to find the maximum weight set with no conflicts.
        
        Construction for Maximum Independent Set via Min-Cut:
        - Source s, sink t
        - For each vertex v: edge s->v with weight w_v (here w_v = 1)
        - For each edge (u,v) in the CONFLICT graph (i.e., non-edge in original):
          edge u->v with weight INF, edge v->u with weight INF
        - Min s-t cut: let S be set of vertices reachable from s in residual graph.
          The independent set is S.
          Wait, this isn't quite right for general graphs.
        
        Actually, maximum independent set is NP-hard, so min-cut won't solve it exactly.
        But we can use it as a heuristic or for special structures.
        
        Let me try a different approach: use the structure that the optimal set
        is related to divisors of some number M <= 512.
        """
        pass

    def divisor_based_search():
        """
        Key insight: If S is a set where lcm(a,b) <= 512 for all pairs,
        then lcm(S) <= 512. So S ⊆ divisors of L where L = lcm(S) <= 512.
        
        Strategy: For each M in 1..512, consider the set of all divisors of M.
        This set has the property that lcm(a,b) divides M for all a,b | M.
        So if M <= 512, then lcm(a,b) <= M <= 512.
        
        The largest such set for a given M is just all divisors of M.
        We want to maximize |divisors(M)| over M in 1..512.
        
        But we can also take subsets of divisors of M, or combine elements
        from different M's if their lcm is still bounded.
        
        Let's first find the best single-M solution, then try to improve.
        """
        best_m_set = set()
        best_m_size = 0
        best_m_val = 1
        
        for M in range(1, 513):
            # Count divisors of M
            div_count = 0
            for d in range(1, M + 1):
                if M % d == 0:
                    div_count += 1
            if div_count > best_m_size:
                best_m_size = div_count
                best_m_val = M
                # Collect divisors
                divs = [d for d in range(1, M + 1) if M % d == 0]
                best_m_set = set(d - 1 for d in divs)
        
        return best_m_set, best_m_val, best_m_size

    # Run divisor-based search first
    div_set, div_M, div_size = divisor_based_search()
    div_mask = 0
    for i in div_set:
        div_mask |= (1 << i)
    update_best(div_set, div_mask)
    
    # Local search on divisor set
    if time.time() < deadline:
        s, m = local_search(set(div_set), div_mask)

    # Build initial seeds
    inits = []

    # 1. Natural order 1..22
    s1 = set(range(22))
    m1 = 0
    for i in s1:
        m1 |= (1 << i)
    inits.append((s1, m1))

    # 2. Divisors of 512
    divs512 = [i + 1 for i in range(n) if 512 % (i + 1) == 0]
    s2 = set(d - 1 for d in divs512)
    m2 = 0
    for i in s2:
        m2 |= (1 << i)
    inits.append((s2, m2))

    # 3. Powers of 2
    pows = set()
    p = 1
    while p <= 512:
        pows.add(p - 1)
        p *= 2
    m3 = 0
    for i in pows:
        m3 |= (1 << i)
    inits.append((pows, m3))

    # 4. Divisors of 360 (measured to give size 24)
    divs360 = [i + 1 for i in range(n) if 360 % (i + 1) == 0]
    s4 = set(d - 1 for d in divs360)
    m4 = 0
    for i in s4:
        m4 |= (1 << i)
    inits.append((s4, m4))

    # 5. Random seeds
    random.seed(42)
    for _ in range(10):
        s5 = set(random.sample(range(n), 20))
        m5 = 0
        for i in s5:
            m5 |= (1 << i)
        inits.append((s5, m5))

    # Run local search on each seed
    for s, m in inits:
        if time.time() > deadline:
            break
        s, m = local_search(s, m)

    # Greedy with multiple orderings
    # Natural order
    natural = list(range(n))
    c, cm = greedy_clique(natural)
    update_best(c, cm)
    if time.time() < deadline:
        c, cm = local_search(c, cm)

    # Degree descending
    sorted_desc = sorted(range(n), key=lambda x: -degree[x])
    c, cm = greedy_clique(sorted_desc)
    update_best(c, cm)
    if time.time() < deadline:
        c, cm = local_search(c, cm)

    # Degree ascending
    sorted_asc = sorted(range(n), key=lambda x: degree[x])
    c, cm = greedy_clique(sorted_asc)
    update_best(c, cm)
    if time.time() < deadline:
        c, cm = local_search(c, cm)

    # Random orderings
    for iteration in range(50):
        if time.time() > deadline:
            break
        random_order = list(range(n))
        random.shuffle(random_order)
        c, cm = greedy_clique(random_order)
        if len(c) > best_size:
            update_best(c, cm)
            c, cm = local_search(c, cm)
        elif len(c) >= best_size - 2 and time.time() < deadline - 5:
            # Close to best, try to improve
            c, cm = local_search(c, cm)

    # Enhanced local search with double-swap
    def enhanced_local_search(s, m):
        """Enhanced local search with double-swap moves."""
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Phase 1: greedy add
            while True:
                added = False
                for v in range(n):
                    if v not in s:
                        if (masks[v] & m) == m:
                            s.add(v)
                            m |= (1 << v)
                            added = True
                            break
                if not added:
                    break
            if added:
                improved = True
                update_best(s, m)

            # Phase 2: single-swap (remove u, add v)
            swapped = False
            s_list = list(s)
            for u in s_list:
                if time.time() > deadline:
                    break
                u_bit = 1 << u
                m_wo_u = m ^ u_bit
                for v in range(n):
                    if v in s:
                        continue
                    if (masks[v] & m_wo_u) == m_wo_u:
                        # Valid swap: remove u, add v
                        s.remove(u)
                        s.add(v)
                        m = m_wo_u | (1 << v)
                        swapped = True
                        update_best(s, m)
                        break
                if swapped:
                    break
            if swapped:
                improved = True

            # Phase 3: double-swap (remove u1, u2, add v1, v2)
            if time.time() < deadline:
                s_list = list(s)
                double_swapped = False
                for i, u1 in enumerate(s_list):
                    if time.time() > deadline:
                        break
                    for j, u2 in enumerate(s_list):
                        if i == j:
                            continue
                        u1_bit = 1 << u1
                        u2_bit = 1 << u2
                        m_wo = m ^ u1_bit ^ u2_bit
                        # Try to find two vertices v1, v2 not in s that are compatible with m_wo
                        candidates = []
                        for v in range(n):
                            if v in s:
                                continue
                            if (masks[v] & m_wo) == m_wo:
                                candidates.append(v)
                        # Try all pairs of candidates
                        for k in range(len(candidates)):
                            if time.time() > deadline:
                                break
                            for l in range(k + 1, len(candidates)):
                                v1 = candidates[k]
                                v2 = candidates[l]
                                # Check if v1 and v2 are compatible
                                if (masks[v1] & (1 << v2)) and (masks[v2] & (1 << v1)):
                                    # Valid double swap
                                    s.remove(u1)
                                    s.remove(u2)
                                    s.add(v1)
                                    s.add(v2)
                                    m = m_wo | (1 << v1) | (1 << v2)
                                    double_swapped = True
                                    update_best(s, m)
                                    break
                            if double_swapped:
                                break
                        if double_swapped:
                            break
                    if double_swapped:
                        break
                if double_swapped:
                    improved = True

        return s, m

    # Run enhanced local search on best set
    if time.time() < deadline:
        s, m = enhanced_local_search(set(best_set), best_mask)

    # Final polish: try to add any remaining compatible elements to best_set
    if time.time() < deadline:
        s = set(best_set)
        m = best_mask
        while True:
            added = False
            for v in range(n):
                if v not in s:
                    if (masks[v] & m) == m:
                        s.add(v)
                        m |= (1 << v)
                        added = True
                        break
            if not added:
                break
        update_best(s, m)

    result = sorted([i + 1 for i in best_set])
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
