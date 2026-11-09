"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    deadline = time.time() + 108  # safety margin
    best_edges = 39
    best_adj = [set() for _ in range(16)]
    for a in range(3):
        for b in range(3, 16):
            best_adj[a].add(b)
            best_adj[b].add(a)

    def contains_q3(adj):
        # Check if graph contains Q3 as a (not necessarily induced) subgraph
        # Q3: 8 vertices, 12 edges, 4-cycles. Bipartite 4+4.
        # Try all subsets of 8 vertices, check if Q3 can be embedded.
        # Optimization: Q3 is 3-regular on 8 vertices.
        for verts in itertools.combinations(range(16), 8):
            vset = set(verts)
            # Count edges within the subset
            internal_edges = 0
            for i in range(len(verts)):
                vi = verts[i]
                for j in range(i + 1, len(verts)):
                    vj = verts[j]
                    if vj in adj[vi]:
                        internal_edges += 1
            # Q3 has exactly 12 edges. Need at least 12, and must be embeddable.
            if internal_edges < 12:
                continue
            # Try to find a Q3 embedding within these 8 vertices
            # Q3 vertices can be labeled by 3-bit strings; edges connect strings differing in 1 bit.
            # We need to assign each of the 8 vertices a 3-bit label such that
            # two graph vertices are adjacent in graph => their labels differ in at least 1 bit
            # (i.e., NOT identical label, which is automatic since distinct vertices)
            # AND every Q3 edge (labels differing in exactly 1 bit) must be a graph edge.
            # So: for every pair of labels (a,b) with Hamming distance 1, the corresponding graph
            # vertices must be adjacent.
            # Equivalent: the 8 chosen vertices must contain a copy of Q3 as a subgraph.
            # Since Q3 is 3-regular, each vertex in the Q3 subgraph has degree >= 3 within the 8.
            # Check: does there exist a bijection from 8 vertices to {0,1}^3 such that
            # for all (x,y) with dist(x,y)=1, the graph edge exists?
            labels = list(itertools.product([0, 1], repeat=3))
            # For each permutation of verts assigned to labels, check all 12 Q3 edges
            # 8! = 40320, times C(16,8)=12870 = ~519M -- too slow in pure Python
            # Use smarter check: try all 4-bit... no, try to fix one vertex to label 0
            # by symmetry, reducing to 7! = 5040 per subset. Still 12870*5040 = 64M, borderline.
            # Better: use the fact that Q3 has a unique structure.
            # For each subset, check if the induced subgraph has Q3 as subgraph.
            # Since we only need 12 specific edges to be present, we can check:
            # Build the 8x8 adjacency within subset.
            sub_adj = [[False] * 8 for _ in range(8)]
            for i in range(8):
                for j in range(i + 1, 8):
                    if verts[j] in adj[verts[i]]:
                        sub_adj[i][j] = True
                        sub_adj[j][i] = True
            # Now check if Q3 (on labels 0..7, edges between labels diff by 1 bit) is a subgraph
            # Try: fix vertex 0 of subset to label 0. Then try permutations of remaining 7.
            # But 7! = 5040. For 12870 subsets that's 64M checks. Each check is 12 edge lookups.
            # ~770M ops. Too slow in Python for 108s.
            # Alternative: use a smarter necessary condition first.
            # In Q3, every vertex has degree 3. So in the subgraph, each of the 8 vertices
            # must have degree >= 3 within the 8.
            min_deg = min(sum(sub_adj[i][j] for j in range(8)) for i in range(8))
            if min_deg < 3:
                continue
            # Further: Q3 is bipartite. The 8 vertices split into two sets of 4 (even/odd parity).
            # Each vertex in one set connects to 3 vertices in the other set.
            # So we need a 4-4 bipartite split where each vertex has degree >= 3 to the other side.
            # Total edges between the two parts must be at least 12.
            # Try all C(8,4) = 70 ways to split into two sets of 4.
            for left_mask in range(256):
                if bin(left_mask).count('1') != 4:
                    continue
                left = set(i for i in range(8) if left_mask >> i & 1)
                right = set(i for i in range(8) if not (left_mask >> i & 1))
                # Check bipartite: no edges within left or within right
                has_internal = False
                for i in left:
                    for j in left:
                        if i != j and sub_adj[i][j]:
                            has_internal = True
                            break
                    if has_internal:
                        break
                if has_internal:
                    continue
                for i in right:
                    for j in right:
                        if i != j and sub_adj[i][j]:
                            has_internal = True
                            break
                    if has_internal:
                        break
                if has_internal:
                    continue
                # Now check if we can match: each left vertex needs >=3 neighbors in right,
                # and each right vertex needs >=3 neighbors in left, and total edges = 12.
                # Actually we need exactly the Q3 structure: it's a 3-regular bipartite graph.
                # Count edges between left and right:
                cross_edges = 0
                for i in left:
                    for j in right:
                        if sub_adj[i][j]:
                            cross_edges += 1
                if cross_edges < 12:
                    continue
                # Need to verify it's exactly Q3 (3-regular bipartite on 4+4)
                # Each left vertex must have exactly 3 neighbors in right (for Q3)
                # But it could have more (superset). We need Q3 as subgraph, so we need
                # to find 3 neighbors per left vertex and 3 per right vertex forming Q3.
                # This is a 3-regular bipartite subgraph problem. For 4x4, it's tractable.
                # Build the 4x4 bipartite adjacency.
                left_list = sorted(left)
                right_list = sorted(right)
                bip = [[sub_adj[li][ri] for ri in right_list] for li in left_list]
                # Check if there's a 3-regular bipartite subgraph (i.e., a perfect matching in the
                # complement sense). Actually, a 3-regular bipartite graph on 4+4 is the complement
                # of a 1-regular bipartite graph (perfect matching) in K4,4.
                # So we need: there exists a perfect matching in K4,4 \ bip_subgraph where
                # bip_subgraph has each vertex degree exactly 3.
                # Equivalent: find a perfect matching M in K4,4 such that for each (i,j) in M,
                # bip[i][j] is False (since the missing edges form a perfect matching).
                # So: is there a perfect matching in the complement of bip?
                comp = [[not bip[i][j] for j in range(4)] for i in range(4)]
                # Check if comp has a perfect matching (4x4, try all 4! = 24)
                found_matching = False
                for perm in itertools.permutations(range(4)):
                    ok = True
                    for i in range(4):
                        if not comp[i][perm[i]]:
                            ok = False
                            break
                    if ok:
                        found_matching = True
                        break
                if found_matching:
                    return True
        return False

    # Greedy with multiple random restarts
    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed)
        adj = [set() for _ in range(16)]
        for a in range(3):
            for b in range(3, 16):
                adj[a].add(b)
                adj[b].add(a)

        missing = [(i, j) for i in range(16) for j in range(i + 1, 16) if j not in adj[i]]
        random.shuffle(missing)

        for (u, v) in missing:
            if time.time() > deadline:
                break
            adj[u].add(v)
            adj[v].add(u)
            if contains_q3(adj):
                adj[u].discard(v)
                adj[v].discard(u)

        cur_edges = sum(len(s) for s in adj) // 2
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = [set(s) for s in adj]

    result = []
    for i in range(16):
        for j in best_adj[i]:
            if i < j:
                result.append([i, j])
    return result
# EVOLVE-BLOCK-END
