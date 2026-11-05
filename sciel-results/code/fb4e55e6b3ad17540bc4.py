import time
import random

N = 40
_DEADLINE = time.time() + 85

def construct():
    # Strategy: Use a cycle on a subset of vertices + chords from hub vertices
    # to cover all cycle lengths 3..40.
    #
    # Key insight: A cycle of length L gives us one cycle of length L.
    # If we have a "base cycle" of length M, we can get cycles of length
    # M, M+1, ..., up to some max by adding vertices via chords.
    #
    # Better structural idea: Use TWO large cycles that share some vertices,
    # or use a "ladder" structure. But let's try a simpler approach:
    #
    # Approach: Create a cycle on vertices 0..39 (40-cycle). This gives us
    # length 40. Then add chords to create cycles of all other lengths.
    # A chord (i, j) on a cycle of length n creates two cycles of lengths
    # (j-i) and (n-(j-i)). So one chord gives us 2 cycle lengths.
    # We need lengths 3..39 (37 lengths). Each chord gives 2, so we need
    # ~19 chords. Total edges = 40 (cycle) + 19 (chords) = 59.
    # Score = 59 - 40 = 19.
    #
    # But we can do better: use chords that create overlapping cycles.
    # Actually, let me think differently.
    #
    # Alternative: Use a "star of paths" structure. Place vertex 0 as a hub.
    # Connect 0 to vertices s1, s2, ..., sk. Connect si to si+1 via paths.
    # This is essentially what the seed does. The seed gets score 1.0,
    # meaning edge_count - 40 = some value. Let me compute:
    # Seed: 38 path edges + 10 spoke edges = 48 edges. Score = 48-40 = 8.
    # Wait, the score is 1.0 which is the ratio seed/value. So value=8?
    # No, the score IS the value (edge_count - 40). Lower is better.
    # Actually re-reading: "The evaluator measures the raw value (lower is better)."
    # So score = edge_count - 40. Seed has 48 edges, score = 8.
    #
    # Let me try to minimize edges. We need cycles of lengths 3..40.
    # Each cycle of length L uses L edges (but edges can be shared).
    #
    # Key structural change: Use a CYCLE as the backbone instead of a path.
    # A 40-cycle uses 40 edges. Then add chords. Each chord (u,v) where
    # u and v are on the cycle creates a smaller cycle. If the distance
    # along the cycle between u and v is d, we get cycles of length d and 40-d.
    #
    # We need cycle lengths: 3,4,5,...,39 (length 40 is the cycle itself).
    # So we need to cover {3,4,...,39} with pairs {d, 40-d} for chosen chords.
    # Note that d and 40-d are complementary. We need d in {3,4,...,37}
    # (since d=38 gives 40-38=2 which is invalid, d=39 gives 1 which is invalid).
    # Actually d must be at least 3 for a valid cycle (3 distinct vertices).
    # So d in {3,4,...,37}. The pair is {d, 40-d}.
    # We need to cover all values in {3,4,...,39}.
    # Note 40-d for d in {3,...,37} gives {37,...,3}. So pairs cover {3,...,37}.
    # But we also need 38 and 39. 38 = 40-2, but d=2 is invalid. 39=40-1, d=1 invalid.
    # So we can't get 38 or 39 from a single chord on a 40-cycle.
    #
    # Hmm, this approach has issues. Let me think of another structural idea.
    #
    # New idea: Use a "theta graph" or "multi-path" structure.
    # Take two special vertices A and B. Create multiple paths between A and B.
    # If we have paths of lengths l1, l2, ..., lk between A and B, then
    # any pair of paths gives a cycle of length li + lj.
    # So we need the pairwise sums to cover all required cycle lengths.
    #
    # Let's say we have k paths between A and B with lengths l1 <= l2 <= ... <= lk.
    # The cycles have lengths li + lj for i < j, plus we can have cycles that
    # go A -> path_i -> B -> path_j -> A.
    # We need these sums to cover {3, 4, ..., 40}.
    #
    # Minimum path length is 2 (A-x-B). So li >= 2.
    # To get cycle length 3: need li + lj = 3, so li=1, lj=2. But li>=2, so 2+2=4 min.
    # Can't get 3 with this structure unless we allow li=1 (direct edge A-B).
    # If we have a direct edge A-B (length 1), then with a path of length 2,
    # we get cycle length 1+2=3. Good.
    #
    # So let's include a direct edge A-B (length 1). Then paths of lengths
    # l2, l3, ..., lk. Cycles: 1+li for i=2..k, and li+lj for i<j, i,j>=2.
    # We need to cover {3,4,...,40}.
    # 1+l2=3 => l2=2. 1+l3=4 => l3=3. Etc.
    # If we have paths of lengths 1,2,3,...,m, then 1+li = li+1 covers 3..m+1.
    # And li+lj covers more.
    # With paths 1,2,3,...,20: 1+li covers 3..21. li+lj for i<j: min 2+3=5, max 19+20=39.
    # We need up to 40. 20+20 is not allowed (i<j). Max is 19+20=39. Need 40.
    # Add path of length 21: 1+21=22, and 19+21=40, 20+21=41 (too big).
    # So with paths 1,2,...,21, we can get 3..40? Let me check.
    # 1+li: 3..22. li+lj: 5..40 (19+21=40). Union covers 3..40? 
    # 3,4 from 1+li. 5..22 from both. 23..40 from li+lj. Yes!
    # Total edges: sum of path lengths = 1+2+...+21 = 231. Way too many.
    #
    # This approach uses too many edges. The seed's 48 edges is much better.
    #
    # Let me reconsider. The seed uses 48 edges. Can we do better than 48?
    # The theoretical minimum: we need 38 cycles (lengths 3..40). Each cycle
    # must have at least 3 edges, but edges can be shared. In a sparse graph,
    # each new edge can contribute to at most a few new cycles.
    #
    # Let me try a different structural idea: a "bipartite-like" structure
    # with two hubs.
    #
    # Actually, let me just try to improve the seed's approach by finding
    # a better difference set or using a different hub placement.
    #
    # The seed uses 10 spokes + 38 path edges = 48 edges.
    # Can we use fewer spokes? The spokes must cover all differences 1..38.
    # A difference basis of order 2 for Z_39 needs at least ceil(sqrt(38)) ~ 7 elements.
    # But we need differences 1..38, which is 38 values. With k elements, we get
    # at most k*(k-1)/2 differences. So k*(k-1)/2 >= 38 => k >= 9.
    # So 9 spokes might suffice if a good difference set exists.
    #
    # Let me search for a difference set of size 9 that covers 1..38.
    
    def find_diff_basis(target_size=9, max_val=39):
        """Find a set S of size target_size from 1..max_val-1 such that
        pairwise differences cover 1..max_val-2."""
        target = set(range(1, max_val - 1))  # 1..38
        rng = random.Random(42)
        best = None
        for attempt in range(200):
            if time.time() > _DEADLINE - 5:
                break
            S = set(rng.sample(range(1, max_val), target_size))
            covered = set()
            s_list = sorted(S)
            for i in range(len(s_list)):
                for j in range(i + 1, len(s_list)):
                    covered.add(s_list[j] - s_list[i])
            if target <= covered:
                return sorted(S)
            # Local search
            improved = True
            while improved and time.time() < _DEADLINE - 5:
                improved = False
                s_list = sorted(S)
                covered = set()
                for i in range(len(s_list)):
                    for j in range(i + 1, len(s_list)):
                        covered.add(s_list[j] - s_list[i])
                missing = target - covered
                if not missing:
                    return sorted(S)
                for idx in range(target_size):
                    old = s_list[idx]
                    for nv in range(1, max_val):
                        if nv in S:
                            continue
                        S.remove(old)
                        S.add(nv)
                        new_s = sorted(S)
                        new_cov = set()
                        for i in range(len(new_s)):
                            for j in range(i + 1, len(new_s)):
                                new_cov.add(new_s[j] - new_s[i])
                        if len(new_cov & target) > len(covered & target):
                            covered = new_cov
                            s_list = new_s
                            improved = True
                            break
                        else:
                            S.remove(nv)
                            S.add(old)
                    if not (target - covered):
                        return sorted(S)
        return None
    
    S = find_diff_basis(9, N)
    if S is None:
        S = find_diff_basis(10, N)
    if S is None:
        # Fallback to seed's approach
        S = [1, 3, 7, 12, 18, 25, 33, 37, 38, 39]
    
    S_set = set(S)
    edges = []
    edges_set = set()
    
    def add_edge(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        if (a, b) not in edges_set:
            edges_set.add((a, b))
            edges.append((a, b))
    
    # Path edges
    for i in range(1, N - 1):
        add_edge(i, i + 1)
    # Spoke edges
    for s in S:
        add_edge(0, s)
    
    cycles = []
    for L in range(3, N + 1):
        diff = L - 2
        found = False
        for u in S:
            v = u + diff
            if v in S_set:
                cycles.append([0] + list(range(u, v + 1)))
                found = True
                break
        if not found:
            # Fallback: return a valid but possibly suboptimal solution
            e2 = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
            c2 = []
            for L in range(3, N + 1):
                c2.append(list(range(L)))
            return {'edges': e2, 'cycles': c2}
    
    return {'edges': edges, 'cycles': cycles}
