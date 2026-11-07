import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def popcount(x):
        return bin(x).count("1")

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Bitmasks for N(v)
        masks = {}
        for v in range(k, n):
            m = 0
            for u in neighbors[v]:
                m |= (1 << u)
            masks[v] = m

        # Sort I vertices by mask size descending
        sorted_v = sorted(range(k, n), key=lambda v: popcount(masks[v]), reverse=True)

        A = {}  # v -> mask of A_v (submask of N(v))
        # Process vertices in order, ensuring |A[v] & A[prev]| <= 1
        # We want to keep A[v] as large as possible.
        for v in sorted_v:
            # Start with full N(v)
            cur_A = masks[v]
            
            # Resolve conflicts with previous vertices
            # We need to remove bits from cur_A until intersection with every previous A[prev] is <= 1.
            # To maximize size, we should be careful.
            # Greedy approach: iterate through previous vertices.
            # For each prev, compute inter = cur_A & A[prev].
            # If popcount(inter) > 1, remove bits from cur_A (that are in inter) until popcount(inter) <= 1.
            # Which bit to remove? 
            # Removing a bit that is in many OTHER previous A's might be worse? No, we are just satisfying current constraint.
            # Let's just remove the lowest bit in inter for determinism and speed, or perhaps the one that appears in the most other A's?
            # Actually, a better heuristic: remove the bit from inter that appears in the FEWEST other (future or past) masks? 
            # For now, lowest bit is simple. Let's try to remove the bit that is "least critical".
            # Actually, the constraint is pairwise. 
            # Let's stick to the current logic but optimized.
            
            for prev in sorted_v:
                if prev == v:
                    break # processed in order, so prev comes before v in sorted_v? No, sorted_v is list, we iterate range
                if prev not in A:
                    continue
                inter = cur_A & A[prev]
                cnt = popcount(inter)
                if cnt > 1:
                    # Need to reduce
                    # We must remove at least cnt-1 bits from inter that are also in cur_A.
                    # Since cur_A contains inter, we can remove bits from cur_A.
                    # We want to remove bits that are in inter.
                    # Let's pick the bits to remove from inter.
                    # To be deterministic, remove the lowest (cnt-1) bits of inter.
                    bits_to_remove = 0
                    tmp = inter
                    for _ in range(cnt - 1):
                        bit = tmp & -tmp
                        bits_to_remove |= bit
                        tmp &= ~bit
                    cur_A &= ~bits_to_remove
            
            A[v] = cur_A

        # Compute covered K-edges
        covered_mask = 0  # Bitmask of K-edges is too large (k(k-1)/2 bits). k up to 30. 435 bits. Python int handles it, but let's use set of tuples or similar if k is bigger.
        # Actually, representing edges as (u, w) tuples is fine for k <= 30 (435 edges).
        covered_edges = set()
        for v in range(k, n):
            m = A[v]
            # Vertices in A[v]
            # To optimize, iterate set bits
            verts = []
            tmp = m
            while tmp:
                bit = tmp & -tmp
                verts.append(bit.bit_length() - 1)
                tmp &= ~bit
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    covered_edges.add((verts[i], verts[j]))

        # Remaining K-edges
        remaining_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_edges:
                    remaining_edges.append((u, w))

        # Greedy Clique Cover for remaining_edges
        # This graph is on k vertices.
        adj_rem = {u: set() for u in range(k)}
        for u, w in remaining_edges:
            adj_rem[u].add(w)
            adj_rem[w].add(u)
        
        rem_cliques = []
        edges_to_cover = set(remaining_edges)
        
        while edges_to_cover:
            u, w = next(iter(edges_to_cover))
            clique = [u, w]
            # Candidates: common neighbors of u and w in the REMAINING graph
            common = adj_rem[u] & adj_rem[w]
            candidates = list(common)
            # Sort for determinism?
            candidates.sort()
            
            new_clique = []
            for c in candidates:
                # Check if c is connected to all vertices in clique via REMAINING edges
                ok = True
                for a in clique:
                    # Check if edge (a, c) is in edges_to_cover
                    # Since a is in clique, and c is candidate, they are not same.
                    # Note: edges_to_cover contains undirected edges. We stored them as (min, max) ? 
                    # No, we added (u, w) as generated. 
                    # In the loop generating remaining_edges, u < w. So edges are (min, max).
                    # But adj_rem is symmetric.
                    # Let's ensure edges_to_cover lookup works.
                    a, c_pair = sorted((a, c))
                    if (a, c_pair) not in edges_to_cover and (c_pair, a) not in edges_to_cover: # (a, c_pair) is the canonical form
                        # Wait, if I store (min, max), I only need to check that one.
                        if (a, c) if a < c else (c, a) not in edges_to_cover:
                            ok = False
                            break
                if ok:
                    clique.append(c)
                    new_clique.append(c)
            
            rem_cliques.append(clique)
            
            # Remove all edges induced by this clique
            cl = set(clique)
            for e in list(edges_to_cover):
                x, y = e
                if x in cl and y in cl:
                    edges_to_cover.remove(e)

        # Construct final list of cliques
        cliques = []
        
        # 1. Mixed cliques for v in I
        for v in range(k, n):
            m = A[v]
            if m == 0:
                continue # No mixed clique needed (or empty, which is invalid anyway)
            verts = []
            tmp = m
            while tmp:
                bit = tmp & -tmp
                verts.append(bit.bit_length() - 1)
                tmp &= ~bit
            verts.sort() # Deterministic
            cliques.append(verts + [v])

        # 2. Leftover 2-cliques {v, u}
        for v in range(k, n):
            m_N = masks[v]
            m_A = A[v]
            leftover_mask = m_N & ~m_A
            if leftover_mask:
                verts = []
                tmp = leftover_mask
                while tmp:
                    bit = tmp & -tmp
                    verts.append(bit.bit_length() - 1)
                    tmp &= ~bit
                verts.sort()
                for u in verts:
                    cliques.append([u, v])

        # 3. K-only cliques
        for c in rem_cliques:
            if len(c) >= 2:
                cliques.append(sorted(c))

        # Final validation/cleanup?
        # The prompt says "Any broken rule makes the whole result invalid".
        # My construction:
        # - Mixed cliques {v} U A[v]: v in I, A[v] subset N(v). So {v} U A[v] is a clique? 
        #   Vertices in A[v] form a clique in K (since K is a clique). v is connected to all in A[v] (since A[v] subset N(v)). 
        #   So yes, it is a clique.
        #   Size is |A[v]| + 1. If |A[v]| >= 1, size >= 2.
        #   If |A[v]| == 0, we skipped it.
        # - Leftover 2-cliques {v, u}: v in I, u in N(v) \ A[v]. Edge (v, u) exists. Size 2.
        # - K-only cliques: Subset of K. K is a clique. Size >= 2.
        
        # Partition check:
        # - Edges in K:
        #   - If covered by a mixed clique {v} U A[v], both endpoints are in A[v].
        #   - Else, they are in remaining_edges, and covered by exactly one K-only clique (from greedy maximal cliques which partition the edges).
        # - Edges between K and I:
        #   - Edge (v, u) where v in I, u in K.
        #   - If u in A[v], it is covered by the mixed clique {v} U A[v].
        #   - If u not in A[v] (i.e., in N(v)\A[v]), it is covered by the 2-clique {v, u}.
        #   - Since A[v] and N(v)\A[v] partition N(v), and non-neighbors are not edges, every cross-edge is covered exactly once.
        #     Wait, is it possible an edge is covered by TWO mixed cliques?
        #     Edge (u, w) in K. Covered by mixed clique v if u, w in A[v].
        #     Covered by mixed clique v' if u, w in A[v'].
        #     We ensured |A[v] & A[v']| <= 1.
        #     If u, w in A[v] AND u, w in A[v'], then u, w in A[v] & A[v'].
        #     This implies popcount(A[v] & A[v']) >= 2.
        #     But we resolved conflicts to ensure popcount <= 1.
        #     So K-edges are in at most one mixed clique.
        #     If not in any mixed clique, they are in remaining_edges and covered by K-only cliques.
        #     So K-edges are covered exactly once.
        # - No edge in I-I (independent set).

        # Types
        # Must be list of lists of ints.
        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            # Return what we have?
            # "Any broken rule... invalid". If we don't return 3 lists, it's invalid.
            # But we hope to finish in time.
            # If we break early, we should probably return a valid (though maybe suboptimal) result for all 3, or fail?
            # The prompt says "return a list of three lists".
            # If I can't finish, I should try to be fast enough.
            # My algorithm is O(n^2 + k^3) approx. Very fast.
            pass
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    return results
