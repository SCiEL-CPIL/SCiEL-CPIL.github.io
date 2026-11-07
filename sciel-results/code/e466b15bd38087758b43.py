import time
import random
from collections import defaultdict
from scipy.sparse.csgraph import maximum_bipartite_matching
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency and edge set
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))

    def expand_biclique(L, R, remaining):
        """Maximally expand biclique (L, R) using only edges in remaining."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Try adding to L
            for x in range(n):
                if x in L:
                    continue
                ok = True
                for y in R:
                    if not adj[x][y]:
                        ok = False
                        break
                    e = (min(x, y), max(x, y))
                    if e not in remaining:
                        ok = False
                        break
                if ok:
                    L.add(x)
                    changed = True
            # Try adding to R
            for y in range(n):
                if y in R:
                    continue
                ok = True
                for x in L:
                    if not adj[x][y]:
                        ok = False
                        break
                    e = (min(x, y), max(x, y))
                    if e not in remaining:
                        ok = False
                        break
                if ok:
                    R.add(y)
                    changed = True
        return L, R

    def generate_candidates(remaining_edges, strategy='degree'):
        """Generate a list of candidate bicliques from remaining edges."""
        rem = set(remaining_edges)
        if not rem:
            return []
        
        # Order edges by a strategy
        if strategy == 'degree':
            # Smallest sum of degrees first (edges in sparse regions)
            deg = [0] * n
            for u, v in rem:
                deg[u] += 1
                deg[v] += 1
            ordered = sorted(rem, key=lambda e: deg[e[0]] + deg[e[1]])
        elif strategy == 'reverse_degree':
            deg = [0] * n
            for u, v in rem:
                deg[u] += 1
                deg[v] += 1
            ordered = sorted(rem, key=lambda e: -(deg[e[0]] + deg[e[1]]))
        elif strategy == 'random':
            ordered = list(rem)
            random.shuffle(ordered)
        else:
            ordered = sorted(rem)
        
        candidates = []
        for edge in ordered:
            if edge not in rem:
                continue
            u, v = edge
            L, R = expand_biclique({u}, {v}, rem)
            if not L or not R:
                continue
            b_edges = set((min(a, b), max(a, b)) for a in L for b in R)
            if b_edges and b_edges.issubset(rem):
                candidates.append((list(L), list(R), b_edges))
                rem -= b_edges
        return candidates

    def can_merge(L1, R1, L2, R2):
        """Check if two bicliques can be merged (all cross pairs are edges)."""
        for u in L1:
            for v in R2:
                if not adj[u][v]:
                    return False
        for u in L2:
            for v in R1:
                if not adj[u][v]:
                    return False
        return True

    def min_cover_via_matching(candidates):
        """
        Find minimum number of bicliques using minimum vertex cover on
        bipartite incompatibility graph.
        
        We build a graph where:
        - Left side: candidates indexed 0..m-1
        - Right side: candidates indexed 0..m-1 (same set)
        - Edge (i, j) in incompatibility graph if biclique i and j CANNOT be merged
        
        Wait, that's not quite right for vertex cover. Let me reconsider.
        
        Actually, the problem is: we want to select a minimum set of candidates
        such that every edge is covered by at least one selected candidate, AND
        the selected candidates can be partitioned into groups where each group
        is pairwise compatible (so they can be merged).
        
        This is complex. Let me use a different approach:
        
        The key insight: if we have a set of candidates that covers all edges,
        we want to merge as many as possible. Two candidates can be merged if
        compatible. We want minimum number of merged groups.
        
        For the bipartite matching approach: build a graph where we try to
        pair incompatible candidates and find minimum vertex cover. But the
        structure isn't naturally bipartite.
        
        Alternative: Use the fact that for small k, we can use a different
        polynomial approach. Let me use a greedy + local search hybrid.
        """
        k = len(candidates)
        if k <= 1:
            return candidates
        
        # Build incompatibility matrix
        incompat = [[False] * k for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                L1, R1, _ = candidates[i]
                L2, R2, _ = candidates[j]
                if not can_merge(L1, R1, L2, R2):
                    incompat[i][j] = incompat[j][i] = True
        
        # Build bipartite graph for matching:
        # Left vertices: 0..k-1, Right vertices: 0..k-1
        # Edge (i, j) if incompat[i][j] is True
        # We want minimum vertex cover of this graph.
        # By König's theorem, min vertex cover = max matching in bipartite graph.
        
        # Build adjacency for scipy
        # scipy expects: left nodes 0..n_left-1, right nodes 0..n_right-1
        # Edge from left i to right j if incompat[i][j]
        n_left = k
        n_right = k
        rows = []
        cols = []
        for i in range(k):
            for j in range(k):
                if incompat[i][j]:
                    rows.append(i)
                    cols.append(j)
        
        if not rows:
            # All compatible, merge all
            L_all = set()
            R_all = set()
            for L, R, _ in candidates:
                L_all.update(L)
                R_all.update(R)
            return [[list(L_all), list(R_all)]]
        
        import scipy.sparse as sp
        data = [1] * len(rows)
        matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_left, n_right))
        
        matching = maximum_bipartite_matching(matrix)
        # matching[i] = j means left i matched to right j
        # or matching[i] = -1 means unmatched
        
        # Find minimum vertex cover from matching (König's algorithm)
        # Unmatched left vertices
        unmatched_left = [i for i in range(k) if matching[i] == -1]
        
        # BFS/DFS from unmatched left vertices through alternating paths
        # In the bipartite graph: left -> right via matched edges, right -> left via unmatched edges
        visited_left = set(unmatched_left)
        visited_right = set()
        queue = list(unmatched_left)
        
        while queue:
            u = queue.pop(0)
            # From left u, go to right vertices via edges NOT in matching
            for j in range(k):
                if incompat[u][j] and j not in visited_right:
                    # Check if (u, j) is NOT a matched edge
                    if matching[u] != j:
                        visited_right.add(j)
                        # From right j, go to left vertices via matched edges
                        # Find which left vertex is matched to right j
                        for i2 in range(k):
                            if matching[i2] == j and i2 not in visited_left:
                                visited_left.add(i2)
                                queue.append(i2)
        
        # Minimum vertex cover:
        # Left vertices NOT in visited_left (i.e., matched left vertices not reached)
        # Right vertices IN visited_right
        min_vc_left = set(range(k)) - visited_left
        min_vc_right = visited_right
        
        # The vertices to KEEP are those in the minimum vertex cover
        # The vertices to MERGE (remove) are those NOT in the minimum vertex cover
        # Wait, I need to think about this more carefully.
        
        # Actually, the minimum vertex cover gives us a set of vertices such that
        # every edge (incompatibility) touches at least one vertex in the cover.
        # This means: for every pair of incompatible candidates, at least one is in the cover.
        # Candidates NOT in the cover form an independent set in the incompatibility graph,
        # meaning they are all pairwise compatible and can be merged into one biclique.
        
        # So: candidates in min_vc must be kept separate (or at least, the structure
        # guarantees we need at least |min_vc| bicliques).
        # Candidates NOT in min_vc can all be merged together.
        
        # But wait, we also need the candidates in min_vc to cover all edges.
        # The candidates NOT in min_vc also cover some edges.
        # We need to ensure all edges are covered.
        
        # Let me reconsider. The candidates already cover all edges (by construction).
        # We want to merge as many as possible. The minimum number of merged groups
        # is at least the size of the maximum independent set in the incompatibility
        # graph... no, that's not right either.
        
        # Let me use a simpler approach: greedy merging with the matching as a guide.
        
        # Actually, let me just use the vertex cover result directly:
        # Keep candidates in min_vc separate, merge all others into one group.
        # Then verify and adjust.
        
        keep_indices = min_vc_left  # These must stay separate
        merge_indices = set(range(k)) - min_vc_left  # These can be merged
        
        # But we need to verify that merge_indices are actually pairwise compatible
        # (they should be, since they form an independent set in incompatibility graph)
        
        result = []
        # Add each kept candidate
        for i in sorted(keep_indices):
            L, R, _ = candidates[i]
            result.append([list(L), list(R)])
        
        # Merge all merge_indices into one biclique
        if merge_indices:
            L_merge = set()
            R_merge = set()
            for i in merge_indices:
                L, R, _ = candidates[i]
                L_merge.update(L)
                R_merge.update(R)
            if L_merge and R_merge:
                result.append([list(L_merge), list(R_merge)])
        
        return result

    def validate(bicliques):
        """Validate and convert bicliques."""
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return None
            for u in L:
                if u in R:
                    return None  # Sides must be disjoint
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in covered:
                        return None  # Overlap
                    if not adj[u][v]:
                        return None  # Not an edge
                    covered.add(e)
        if covered != edge_set:
            return None
        result = []
        for L, R in bicliques:
            result.append([[int(x) for x in L], [int(x) for x in R]])
        return result

    def greedy_partition(remaining_edges, strategy='degree'):
        """Generate a partition (not just candidates) using greedy expansion."""
        rem = set(remaining_edges)
        bics = []
        if not rem:
            return bics
        
        deg = [0] * n
        for u, v in rem:
            deg[u] += 1
            deg[v] += 1
        
        if strategy == 'degree':
            ordered = sorted(rem, key=lambda e: deg[e[0]] + deg[e[1]])
        elif strategy == 'reverse_degree':
            ordered = sorted(rem, key=lambda e: -(deg[e[0]] + deg[e[1]]))
        elif strategy == 'random':
            ordered = list(rem)
            random.shuffle(ordered)
        else:
            ordered = sorted(rem)
        
        for edge in ordered:
            if edge not in rem:
                continue
            u, v = edge
            L, R = expand_biclique({u}, {v}, rem)
            b_edges = set((min(a, b), max(a, b)) for a in L for b in R)
            bics.append([list(L), list(R)])
            rem -= b_edges
        return bics

    def local_search(bicliques, time_limit):
        """Try to improve by merging and splitting."""
        best = bicliques
        best_count = len(bicliques)
        
        while time.time() < time_limit:
            improved = False
            # Try merging pairs
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    L1, R1 = best[i]
                    L2, R2 = best[j]
                    if can_merge(L1, R1, L2, R2):
                        # Check if merged biclique is valid (all cross edges exist)
                        # can_merge already checks this
                        new_L = list(set(L1 + L2))
                        new_R = list(set(R1 + R2))
                        if not new_L or not new_R:
                            continue
                        # Verify all edges in new biclique are in edge_set
                        valid = True
                        for u in new_L:
                            for v in new_R:
                                if not adj[u][v]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            new_bics = [best[k] for k in range(len(best)) if k != i and k != j]
                            new_bics.append([new_L, new_R])
                            result = validate(new_bics)
                            if result is not None and len(result) < best_count:
                                best = new_bics
                                best_count = len(result)
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                break
        return best

    best_result = None
    best_count = float('inf')

    # Strategy 1: Greedy with degree ordering + matching-based merge
    for strategy in ['degree', 'reverse_degree']:
        if time.time() > deadline:
            break
        candidates = generate_candidates(edge_set, strategy)
        if candidates:
            merged = min_cover_via_matching(candidates)
            result = validate(merged)
            if result is not None:
                # Local search to improve
                merged2 = local_search(merged, min(time.time() + 10, deadline))
                result2 = validate(merged2)
                if result2 is not None and len(result2) < len(result):
                    result = result2
                if len(result) < best_count:
                    best_count = len(result)
                    best_result = result

    # Strategy 2: Multiple random orderings
    random.seed(123)
    for attempt in range(50):
        if time.time() > deadline:
            break
        bics = greedy_partition(edge_set, 'random')
        if not bics:
            continue
        # Try matching-based merge on this partition
        candidates = [(L, R, set((min(u, v), max(u, v)) for u in L for v in R)) for L, R in bics]
        merged = min_cover_via_matching(candidates)
        result = validate(merged)
        if result is not None:
            merged2 = local_search(merged, min(time.time() + 5, deadline))
            result2 = validate(merged2)
            if result2 is not None and len(result2) < len(result):
                result = result2
            if len(result) < best_count:
                best_count = len(result)
                best_result = result

    # Strategy 3: Greedy with degree ordering (fallback)
    if best_result is None or time.time() < deadline:
        bics = greedy_partition(edge_set, 'degree')
        if bics:
            candidates = [(L, R, set((min(u, v), max(u, v)) for u in L for v in R)) for L, R in bics]
            merged = min_cover_via_matching(candidates)
            result = validate(merged)
            if result is not None:
                merged2 = local_search(merged, min(time.time() + 5, deadline))
                result2 = validate(merged2)
                if result2 is not None and len(result2) < len(result):
                    result = result2
                if result is not None and len(result) < best_count:
                    best_count = len(result)
                    best_result = result

    if best_result is None:
        # Fallback: individual edges
        best_result = [[[u], [v]] for u, v in sorted(edge_set)]

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")
