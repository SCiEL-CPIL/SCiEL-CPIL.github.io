"""
Q3-free graph on 16 vertices: maximize edges.
Formulated as Maximum Independent Set (MIS) on a conflict graph where:
  - 120 nodes = all possible edges of K16
  - Edge (i,j) in conflict graph if edges e_i and e_j cannot both be in a Q3-free graph
  - A Q3 subgraph requires 12 edges forming a specific 4-regular bipartite pattern
    that looks like Q3 in some 8-vertex subset
"""
import time
import random
import itertools
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = t0 + 112  # safety margin before 120s kill

    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_pairs)  # 120

    # --- Encode each edge as an index 0..119 ---
    edge_index = {e: k for k, e in enumerate(all_pairs)}

    # --- Precompute all Q3 subgraphs:
    #     Q3 has 8 vertices, 12 edges.
    #     In Q3, the vertex degrees are all 3. The 8 vertices partition into
    #     two sets of 4 (the "bipartition" of Q3, since Q3 is bipartite).
    #     For each 8-vertex subset, for each 4-subset being one side,
    #     the Q3 is uniquely determined: it's the set of edges that make
    #     each vertex have degree 3 within the 8-vertex set, AND the bipartition
    #     matches the 4+4 split, AND the edge set is exactly 12.
    #
    #     However, Q3 has multiple bipartitions (it's bipartite, but there are
    #     different ways to view it). Actually, Q3 as a graph has a unique
    #     bipartition up to swap: one part is all even-parity bits, other is
    #     odd-parity. So there are C(8,4)/2 = 35 bipartitions per 8-set, but
    #     only those matching the Q3 structure.
    #
    #     Simpler: enumerate all 8-vertex subsets, and for each, check all
    #     12-edge subsets that form a 3-regular graph on those 8 vertices
    #     with bipartition size 4+4, and verify it's isomorphic to Q3.
    #
    #     Even simpler: a graph on 8 vertices is Q3 iff it's 3-regular,
    #     bipartite with parts of size 4, and has girth 4 (no 3-cycles,
    #     no self-loops — automatically satisfied for bipartite), and has
    #     exactly 12 edges. Actually, any 3-regular bipartite graph on
    #     8 vertices with parts of size 4 is either Q3 or K4,4 minus a
    #     perfect matching... let me think.
    #
    #     Q3 = Cartesian product of P2 x P2 x P2. It has 8 vertices, 24 directed
    #     edges, 12 undirected edges. Each vertex has degree 3. It's bipartite
    #     with two parts of size 4.
    #
    #     All 3-regular bipartite graphs on 4+4 vertices:
    #     - Q3
    #     - K4,4 minus a perfect matching (this is actually the 4-vertex cycle
    #       with 4 additional edges... no)
    #     Let me just enumerate directly.
    #
    #     Actually the cleanest approach: generate all 8-vertex subsets. For each,
    #     generate all possible edge subsets of size 12 from the C(8,2)=28 edges,
    #     and check if it's Q3.
    #     C(28,12) is too large.
    #
    #     Better: a 3-regular graph on 8 vertices is Q3 if and only if it's
    #     bipartite with parts of size 4 and has no 4-cycles shorter than...
    #     Actually, Q3 has a specific structure. Let me just hardcode the
    #     adjacency of Q3 and check isomorphism via a smarter approach.
    #
    #     Q3 vertices: 0-7, edges connect i to i^1, i^2, i^4.
    #     The 12 edges of Q3 (0-indexed binary 000-111):
    #     (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    #
    #     For each 8-vertex subset S of {0..15}, and each bijection from
    #     {0..7} to S, check if the edges of Q3 under that bijection all
    #     exist as potential edges. There are C(16,8) * 8! which is too many.
    #
    #     Even smarter: use the bipartition. Q3 has a unique bipartition (up
    #     to swap) of sizes 4 and 4. For each partition of 16 vertices into
    #     A (size 4) and B (size 4), with A ∩ B = ∅, the only possible Q3
    #     using these as bipartition parts is determined by which 12 of the
    #     16 cross-edges between A and B are present. But that's K4,4 minus
    #     4 edges. For Q3 specifically, the 12 edges must make each vertex
    #     degree 3. In K4,4, each vertex has degree 4. Removing 4 edges such
    #     that each vertex loses exactly one edge = removing a perfect matching.
    #     So the Q3 (as bipartite) is K4,4 minus a perfect matching.
    #
    #     BUT: is K4,4 minus a perfect matching always Q3? Yes! K4,4 minus
    #     any perfect matching is isomorphic to Q3. So:
    #
    #     A Q3 subgraph in our 16-vertex graph exists iff there are 4 vertices
    #     A and 4 vertices B, disjoint, such that the 12 edges that form
    #     K4,4(A,B) minus some perfect matching are all present in our graph.
    #
    #     For each 4+4 split, there are 4! = 24 perfect matchings.
    #     Number of 4+4 splits: C(16,4)*C(12,4)/2 = 1820*495/2 = 450,450.
    #     Each gives 24 Q3 edge sets. Total: ~10.8M possible Q3 edge-sets.
    #     Too many to enumerate explicitly.
    #
    #     Alternative: For each 8-vertex subset, there are C(8,4)/2 = 35
    #     bipartitions, each giving 24 Q3s. So 35*24 = 840 per 8-set.
    #     Total 8-sets: C(16,8) = 12,870. Total Q3s: ~10.8M. Same issue.
    #
    #     For the MIS formulation, we need the conflict hypergraph. But 10.8M
    #     hyperedges is too much.
    #
    #     Let me reconsider. The key constraint is: no Q3 subgraph. We need
    #     to build a conflict graph where two edge-nodes conflict if they
    #     cannot both be present. But this isn't a simple pairwise constraint —
    #     it's a 12-way constraint (a set of 12 edges forming Q3 must not all
    #     be present). This is a hypergraph, not a graph.
    #
    #     For a practical approach, let me use a different strategy:
    #     1. Use a fast incremental Q3 checker.
    #     2. Use branch and bound with good heuristics.
    
    # --- Fast Q3 detection ---
    # For each pair of 4-sets (A, B) that are disjoint, check if there's a
    # perfect matching M between A and B such that all 12 edges of K4,4(A,B)
    # minus M are in our graph.
    
    # Precompute all disjoint 4+4 splits
    all_verts = list(range(16))
    four_sets = list(itertools.combinations(all_verts, 4))
    four_sets_set = [set(s) for s in four_sets]
    
    # For each (A, B) disjoint, store the 16 edge indices (a,b) in K4,4
    # We need to check: is there a perfect matching M such that all 12 edges
    # outside M are present?
    # Equivalently: the 4 edges NOT in our graph (within K4,4(A,B)) must form
    # a perfect matching.
    
    # Precompute: for each (A_idx, B_idx), the 16 bit positions and the
    # 24 perfect matchings (as bitmasks of the 16 positions)
    
    # To keep this manageable, let's precompute the structure
    # For efficiency, we'll store for each (A,B) pair:
    #   - list of 16 edge indices
    #   - list of 24 matching masks (16-bit integers)
    
    # This is still a lot of precomputation. Let's do it more carefully.
    
    # Actually, let me use a different but equivalent characterization:
    # A Q3 exists iff there are 4 vertices a1,a2,a3,a4 and 4 vertices
    # b1,b2,b3,b4 such that for each i, a_i is connected to 3 of the b_j's
    # (i.e., not connected to exactly one), and the missing connections
    # form a bijection (each b_j is missing exactly one a_i).
    
    # For the search, let me use a practical approach:
    # - Maintain adjacency as bitmasks
    # - Incremental Q3 check: when adding edge (u,v), check if it completes
    #   a Q3 by seeing if there exists a 4+4 split where (u,v) is the
    #   12th edge of some Q3.
    
    # For a specific edge (u,v), how many Q3s does it participate in?
    # In Q3, an edge connects a vertex from one bipartition to the other.
    # Say u is in A and v is in B (or vice versa). We need to find 3 more
    # vertices for A\{u} and 3 more for B\{v} such that the 12 edges form
    # K4,4 minus a perfect matching, and (u,v) is one of the 12 present edges
    # (i.e., NOT in the matching).
    
    # Number of candidates: C(15,3) * C(12,3) = 455 * 220 = 100,100 per edge.
    # For 120 edges, that's 12M. Still a lot but manageable with bitmasks.
    
    # Let me try yet another approach: since 120 edges and the constraint
    # is "no 12 specific edges all present", I'll use a SAT-like approach
    # with bitmasks.
    
    # --- Build the conflict structure ---
    # For each edge e, compute a bitmask over all other edges that "conflict"
    # with e in the sense that there exists a Q3 containing both. This is
    # too coarse.
    
    # Better: for each Q3 edge-set (12 edges), those 12 nodes form a clique
    # in the conflict hypergraph. For MIS, we can't directly use this.
    
    # PRACTICAL APPROACH:
    # Use a branch-and-bound on the edge set directly.
    # State: partial assignment of include/exclude for each edge.
    # Pruning: if the remaining undecided edges, combined with included edges,
    #   cannot possibly beat the best, prune.
    # Q3 check: maintain the set of included edges; if adding an edge creates
    #   a Q3, force it to be excluded.
    
    # Let me implement a fast Q3 checker first.
    
    # For each edge (u,v), precompute all possible "partner" edge sets that
    # with (u,v) would complete a Q3.
    # Actually, let me just precompute, for each 4+4 split, the 24 Q3 edge-sets.
    # Then for each such Q3 edge-set, mark that those 12 edges are mutually
    # conflicting (in the hypergraph sense).
    
    # For the B&B, I'll use a simpler approach:
    # - Order edges in some sequence
    # - At each step, decide include or exclude
    # - When including an edge, check if it creates a Q3 with already-included edges
    #   (this is the key pruning step)
    # - Bound: current_included + undecided >= best -> prune
    
    # Fast "creates Q3" check for adding edge (u,v):
    # (u,v) is part of a Q3 iff there exists a 4+4 split (A,B) with u,v in
    # opposite parts, and 3 other edges completing the Q3.
    # Precompute for each edge (u,v): the list of Q3 edge-sets (as frozensets
    # or bitmasks of the 11 other edges) that contain (u,v).
    
    # Given the combinatorial explosion, let me precompute per-edge conflict
    # lists differently.
    
    # For each unordered pair of edges {e1, e2} (sharing or not sharing a
    # vertex), determine if there exists a Q3 containing both. There are
    # C(120,2) = 7140 pairs. For each, we need to check if they can be in
    # the same Q3. This is feasible.
    
    # But the constraint is 12-way, not 2-way. Two edges being in a Q3
    # together doesn't mean they conflict; 12 specific edges together conflict.
    
    # I think the most practical approach for this specific size (120 edges)
    # is a well-optimized backtracking search with:
    # 1. Good edge ordering (most constrained first)
    # 2. Fast Q3 detection via precomputed structures
    # 3. Strong upper bounds
    
    # Let me precompute: for each set of k edges (for k up to 12), whether
    # they can be extended to a Q3. No, that's too much.
    
    # Alternative: use the bipartite structure.
    # Precompute all (A, B) disjoint 4-tuples. For each, precompute the 16
    # edges between them and the 24 perfect matchings.
    # Store this in a dict keyed by frozenset(A) and frozenset(B).
    
    # Number of disjoint 4+4 ordered pairs: C(16,4) * C(12,4) = 1820 * 495 = 900,900.
    # Unordered: 450,450. Each has 24 matchings. Total 10.8M entries.
    # This is too much memory.
    
    # Let me reduce: for each 8-vertex subset S, and each bipartition (A,B)
    # of S with |A|=|B|=4, the 24 Q3s are determined. C(16,8)*C(8,4)/2
    # = 12,870 * 35 = 450,450. Same count.
    
    # For memory, I can store only the bitmasks. Each Q3 edge-set is a set
    # of 12 edge-indices. I can represent each as a Python int (120-bit).
    
    # But 10.8M such integers is about 10.8M * 16 bytes = ~173 MB. Marginal.
    
    # Let me try a different approach: precompute per-edge Q3 participation.
    # For each edge e, store the set of all 11-edge subsets that with e
    # form a Q3. The total number of (e, Q3) pairs is 12 * 10.8M / 12 = 10.8M.
    # Still a lot.
    
    # I think for this problem size, the best approach is:
    # 1. A fast "does edge set contain Q3?" checker
    # 2. A good heuristic search (greedy + local search + simulated annealing)
    # 3. Maybe a B&B for the final optimization
    
    # Let me implement a fast checker using the 4+4 split structure.
    # For a given edge set E (represented as a 120-bit mask or adjacency
    # bitmasks), check if there's a Q3:
    #   For each 4+4 split (A,B):
    #     Compute which of the 16 cross-edges are present.
    #     If exactly 12 are present, check if the 4 missing ones form
    #     a perfect matching.
    #     Equivalently: compute the 4x4 submatrix of adjacencies.
    #     Check if it's a permutation matrix complement (i.e., the complement
    #     of a permutation matrix, which has exactly one 0 in each row and col).
    
    # Precompute all 4+4 splits, but store them compactly.
    # For each split, store: A as a 16-bit mask, B as a 16-bit mask,
    # and the mapping from (i,j) in 0..15 to edge-index.
    
    # Actually, for speed, let me store for each (A,B) split:
    #   - adj_A_to_B: 4 lists of 4 edge-indices each
    #   - This allows fast extraction of the 4x4 adjacency submatrix
    
    # But there are 450K splits, each needing 16 edge indices = 7.2M integers.
    # About 29 MB for the indices alone. Plus the A,B masks.
    
    # Let me just build this.
    
    all_sixteen = list(range(16))
    combos4 = list(itertools.combinations(all_sixteen, 4))
    
    # Map vertex -> bit position (identity)
    
    # For each pair (A_idx, B_idx) where A and B are disjoint:
    # Store (A_mask, B_mask, edge_indices[16])
    # where edge_indices[i*4+j] = edge_index[(A[i], B[j])] (sorted)
    
    splits_data = []  # list of (A_mask, B_mask, ei) where ei is tuple of 16 ints
    
    # Also, for each edge, store which splits it participates in
    edge_to_splits = [[] for _ in range(120)]
    
    for Ai in range(len(combos4)):
        A = combos4[Ai]
        Aset = set(A)
        for Bi in range(Ai + 1, len(combos4)):
            B = combos4[Bi]
            if Aset & set(B):
                continue
            # A and B are disjoint
            ei = []
            for i in range(4):
                for j in range(4):
                    u, v = A[i], B[j]
                    e = (u, v) if u < v else (v, u)
                    idx = edge_index[e]
                    ei.append(idx)
            # Avoid duplicate: only process (A,B) once, not (B,A)
            # Since we iterate Ai < Bi, and A,B are disjoint 4-sets,
            # this should be unique. But we need to ensure A is the "first" part.
            # Actually, Q3 bipartition: (A,B) and (B,A) give the same set
            # of 12 edges (just swap roles). So we only need one.
            # But wait: K4,4(A,B) minus matching M on A×B is the same
            # as K4,4(B,A) minus matching M' where M' is M transposed.
            # The 12 edges are the same set. So (A,B) and (B,A) give
            # the same Q3s. Good, we only need one.
            A_mask = 0
            for v in A:
                A_mask |= (1 << v)
            B_mask = 0
            for v in B:
                B_mask |= (1 << v)
            splits_data.append((A_mask, B_mask, tuple(ei)))
            for idx in ei:
                edge_to_splits[idx].append(len(splits_data) - 1)
    
    num_splits = len(splits_data)
    # print(f"Precomputed {num_splits} splits")  # ~450K
    
    # For each split, precompute the 24 perfect matching masks (16-bit)
    # A perfect matching on 4+4 is a permutation of {0,1,2,3}
    # Matching mask: bit i*4+pi = 1 for the matched edge
    perms4 = [p for p in itertools.permutations(range(4))]
    match_masks = []
    for p in perms4:
        mask = 0
        for i in range(4):
            mask |= (1 << (i * 4 + p[i]))
        match_masks.append(mask)
    # 24 entries
    
    # Check function: does the 16-bit mask of present edges in K4,4
    # contain a Q3? i.e., is mask & ~m == full_12 for some matching m?
    # Equivalently: mask has exactly 12 bits set, and the 4 zero bits
    # form a perfect matching.
    # Or: mask ^ matching_mask == 0 for some matching where matching
    # has 4 bits set and mask has 12, and mask | matching == 0xFFFF.
    # i.e., mask & ~m == 0 and mask has 12 bits.
    
    # Precompute: for each of the 65536 possible 16-bit masks, does it
    # contain a Q3? (i.e., can we remove 4 bits to get a perfect matching,
    # leaving 12 bits that are all present?)
    # Equivalently: mask has exactly 12 bits, and the complement (within
    # 16 bits) is a perfect matching.
    # Or more generally: there exists a matching M (4 bits) such that
    # mask & M == 0 and mask | M == 0xFFFF.
    # This means mask has exactly 12 bits set.
    
    # Precompute a 65536-entry boolean table: q3_present[mask]
    q3_table = [False] * 65536
    for mask in range(65536):
        if mask.bit_count() != 12:
            continue
        complement = mask ^ 0xFFFF
        # Check if complement is a perfect matching
        # A perfect matching has exactly one bit in each row and column
        if complement.bit_count() != 4:
            continue
        row_ok = True
        col_ok = True
        for i in range(4):
            row_mask = (complement >> (i * 4)) & 0xF
            if row_mask.bit_count() != 1:
                row_ok = False
                break
        if not row_ok:
            continue
        for j in range(4):
            col_mask = 0
            for i in range(4):
                if complement & (1 << (i * 4 + j)):
                    col_mask |= (1 << i)
            if col_mask.bit_count() != 1:
                col_ok = False
                break
        if row_ok and col_ok:
            q3_table[mask] = True
    
    # Number of masks with Q3: should be 24 * C(8,4)/2 * ... let's not care.
    
    # Now, the checker for a full edge set:
    # Given adj[16] (list of 16-bit masks), for each split, extract the
    # 16-bit mask of present edges and check q3_table.
    
    # To make this fast, precompute for each split the bit positions.
    # splits_data[k] = (A_mask, B_mask, ei_tuple)
    # For split k, the 16-bit mask is:
    #   bit i*4+j = 1 iff edge ei[i*4+j] is present in adj
    
    # For incremental checking (when adding one edge), we only need to
    # check splits that contain that edge.
    
    # Let me now implement the search.
    
    # Strategy:
    # 1. Greedy: build a large Q3-free graph
    # 2. Local search + SA to improve
    # 3. B&B on the remaining degree of freedom
    
    # Seed: K7 + K7 + K2 (57 edges)
    adj = [0] * 16
    for i in range(7):
        for j in range(i + 1, 7):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    for i in range(7, 14):
        for j in range(i + 1, 14):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    adj[14] |= (1 << 15)
    adj[15] |= (1 << 14)
    
    def edge_count():
        return sum(a.bit_count() for a in adj) // 2
    
    best_adj = adj[:]
    best_ec = edge_count()
    
    # For split k, compute the current 16-bit mask
    # split_masks[k] = 16-bit int
    split_masks = [0] * num_splits
    
    def compute_split_mask(k):
        _, _, ei = splits_data[k]
        mask = 0
        for bit in range(16):
            if adj[ei[bit] >> 7]:  # this is wrong, ei is edge index
                pass
        # Correct: for each of the 16 positions, check if the edge is present
        mask = 0
        for bit in range(16):
            e_idx = ei[bit]
            u, v = all_pairs[e_idx]
            if adj[u] & (1 << v):
                mask |= (1 << bit)
        return mask
    
    # Compute initial split masks
    for k in range(num_splits):
        split_masks[k] = compute_split_mask(k)
    
    # Check if current graph has Q3
    def has_q3():
        for k in range(num_splits):
            if q3_table[split_masks[k]]:
                return True
        return False
    
    # count_q3: number of Q3 subgraphs
    def count_q3():
        cnt = 0
        for k in range(num_splits):
            if q3_table[split_masks[k]]:
                cnt += 1
        return cnt
    
    def add_edge(u, v):
        if u > v:
            u, v = v, u
        if adj[u] & (1 << v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        e_idx = edge_index[(u, v)]
        for k in edge_to_splits[e_idx]:
            # Recompute split mask (could be optimized)
            _, _, ei = splits_data[k]
            mask = 0
            for bit in range(16):
                eu, ev = all_pairs[ei[bit]]
                if adj[eu] & (1 << ev):
                    mask |= (1 << bit)
            split_masks[k] = mask
        return True
    
    def remove_edge(u, v):
        if u > v:
            u, v = v, u
        if not (adj[u] & (1 << v)):
            return False
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        e_idx = edge_index[(u, v)]
        for k in edge_to_splits[e_idx]:
            _, _, ei = splits_data[k]
            mask = 0
            for bit in range(16):
                eu, ev = all_pairs[ei[bit]]
                if adj[eu] & (1 << ev):
                    mask |= (1 << bit)
            split_masks[k] = mask
        return True
    
    # Greedy: add edges in random order
    random.seed(123)
    missing = [(u, v) for (u, v) in all_pairs if not (adj[u] & (1 << v))]
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > DEADLINE - 60:
            break
        if add_edge(u, v):
            if not has_q3():
                ec = edge_count()
                if ec > best_ec:
                    best_ec = ec
                    best_adj = adj[:]
            else:
                remove_edge(u, v)
    
    # Simulated Annealing
    if time.time() < DEADLINE - 30:
        adj[:] = best_adj[:]
        for k in range(num_splits):
            split_masks[k] = compute_split_mask(k)
        
        sa_ec = edge_count()
        sa_q3 = count_q3()
        
        T = 5.0
        cooling = 0.9998
        min_T = 0.05
        
        while time.time() < DEADLINE - 15 and T > min_T:
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                remove_edge(u, v)
                new_ec = sa_ec - 1
                new_q3 = count_q3()
                delta = (new_ec - 100 * new_q3) - (sa_ec - 100 * sa_q3)
                if delta >= 0 or random.random() < min(1.0, (2.718281828) ** (delta / T)):
                    sa_ec, sa_q3 = new_ec, new_q3
                    if sa_q3 == 0 and sa_ec > best_ec:
                        best_ec = sa_ec
                        best_adj = adj[:]
                else:
                    add_edge(u, v)
            else:
                add_edge(u, v)
                new_ec = sa_ec + 1
                new_q3 = count_q3()
                delta = (new_ec - 100 * new_q3) - (sa_ec - 100 * sa_q3)
                if delta >= 0 or random.random() < min(1.0, (2.718281828) ** (delta / T)):
                    sa_ec, sa_q3 = new_ec, new_q3
                    if sa_q3 == 0 and sa_ec > best_ec:
                        best_ec = sa_ec
                        best_adj = adj[:]
                else:
                    remove_edge(u, v)
            T *= cooling
        
        # Fix if we have Q3
        while count_q3() > 0 and time.time() < DEADLINE - 10:
            fixed = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    old_q3 = count_q3()
                    remove_edge(u, v)
                    new_q3 = count_q3()
                    if new_q3 < old_q3:
                        fixed = True
                        break
                    add_edge(u, v)
            if not fixed:
                break
    
    # Local search: 1-swap
    adj[:] = best_adj[:]
    for k in range(num_splits):
        split_masks[k] = compute_split_mask(k)
    
    for _ in range(3000):
        if time.time() > DEADLINE - 10:
            break
        present = [(u, v) for (u, v) in all_pairs if adj[u] & (1 << v)]
        if not present:
            break
        ru, rv = random.choice(present)
        remove_edge(ru, rv)
        
        missing = [(u, v) for (u, v) in all_pairs if not (adj[u] & (1 << v))]
        random.shuffle(missing)
        added = 0
        for u, v in missing:
            if added >= 2:
                break
            if add_edge(u, v):
                if not has_q3():
                    added += 1
                else:
                    remove_edge(u, v)
        
        if edge_count() > best_ec:
            best_ec = edge_count()
            best_adj = adj[:]
        else:
            for u, v in missing[:added]:
                remove_edge(u, v)
            add_edge(ru, rv)
    
    # Final exhaustive 1-swap
    adj[:] = best_adj[:]
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        for ru, rv in all_pairs:
            if time.time() > DEADLINE - 5:
                break
            if not (adj[ru] & (1 << rv)):
                continue
            remove_edge(ru, rv)
            for u, v in all_pairs:
                if time.time() > DEADLINE - 5:
                    break
                if adj[u] & (1 << v):
                    continue
                if add_edge(u, v):
                    if not has_q3():
                        ec = edge_count()
                        if ec > best_ec:
                            best_ec = ec
                            best_adj = adj[:]
                            improved = True
                    remove_edge(u, v)
                    break
            add_edge(ru, rv)
    
    adj[:] = best_adj[:]
    
    # Final validity check
    if has_q3():
        # Fallback to known Q3-free graph
        adj = [0] * 16
        for i in range(7):
            for j in range(i + 1, 7):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        for i in range(7, 14):
            for j in range(i + 1, 14):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj[14] |= (1 << 15)
        adj[15] |= (1 << 14)
    
    return [[i, j] for i in range(16) for j in range(i + 1, 16) if adj[i] & (1 << j)]
