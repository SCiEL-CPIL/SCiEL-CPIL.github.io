"""Large 4-uniform sunflower-free family via minimum-degree greedy on conflict graph."""

def construct():
    from itertools import combinations
    import time
    import random
    import sys

    start_time = time.time()
    DEADLINE = start_time + 82  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    # Generate all 4-subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n = len(all_sets)  # 1820

    # Convert to bitmasks
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m

    masks = [to_mask(s) for s in all_sets]

    # Precompute pairwise intersections
    inter = [[0] * n for _ in range(n)]
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            v = mi & masks[j]
            inter[i][j] = v
            inter[j][i] = v

    # Precompute popcount for intersection values (0..15)
    popcount = [0] * 16
    for i in range(16):
        popcount[i] = bin(i).count('1')

    # Build conflict graph: edge (i,j) exists if there exists k such that
    # inter[i][j] == inter[i][k] == inter[j][k]
    # i.e., (i,j,k) form a sunflower
    # For each pair (i,j), check if any k has inter[i][k] == inter[j][k] == inter[i][j]
    #
    # Efficient approach: for each k, and each intersection value v that k shares with
    # multiple other sets, those pairs are conflicts.
    #
    # Alternative: for each triple, but that's O(n^3) = ~6 billion. Too slow.
    #
    # Better: For each pair (i,j) with intersection v = inter[i][j],
    # a conflict exists if there's a k with inter[i][k] == v and inter[j][k] == v.
    # This means k must contain exactly the elements in v (and possibly more from i∪j).
    # Since |v| <= 2 (intersection of two 4-sets), and k is a 4-set containing v,
    # k must be v ∪ {two more elements from [16]\v}.
    #
    # So for a given v (0,1,2 bits), the sets k with inter[i][k]==v and inter[j][k]==v
    # are those 4-sets k such that k ∩ i = v and k ∩ j = v.
    # This means: k contains all of v, k contains none of (i\j), k contains none of (j\i),
    # and k contains 4-|v| elements from [16]\(i∪j).
    #
    # Let's precompute: for each 4-set index, store its mask.
    # For conflict checking, we can enumerate candidate k values.

    # Actually, let's use a different approach for building conflicts:
    # For each k (center), and each pair of elements a,b in [16], consider the
    # intersection value v = k ∩ {some pattern}...
    #
    # Simpler: For each k, group all other sets by their intersection with k.
    # For each group G_v (sets sharing intersection v with k), any pair in G_v
    # has inter[a][b] containing v. If inter[a][b] == v exactly, then (a,b,k) is a sunflower.
    #
    # So: for each k, for each v, for each pair (a,b) in G_v where inter[a][b] == v,
    # mark edge (a,b) as conflict.

    # Build adjacency as sets for conflict graph
    # This is O(n * sum_v |G_v|^2) which could be large but let's try.
    # Actually |G_v| for a given k and v: number of 4-sets with exactly v as intersection with k.
    # If |v|=0: C(16,4) - ... it's C(16-4, 4) = C(12,4) = 495
    # If |v|=1: C(15,3) = 455
    # If |v|=2: C(14,2) = 91
    #
    # So for each k, total pairs to check ~ 495^2/2 + 455^2/2 + 91^2/2 ≈ 122500 + 103200 + 4140 ≈ 230000
    # Times 1820 k's = ~418 million. Too slow in Python.
    #
    # Let's use a smarter approach: precompute for each pair of elements which 4-sets contain them.

    # Index sets by their element composition for fast lookup
    # sets_with_pair[e1][e2] = list of 4-set indices containing both e1 and e2
    # sets_with_single[e] = list of 4-set indices containing e
    # sets_without_pair[e1][e2] = ... (complement)

    # Actually, let's use a completely different strategy:
    # Instead of building the full conflict graph, use a greedy algorithm that
    # dynamically checks conflicts as we build the family.

    # Greedy minimum-degree approach:
    # 1. Start with all n sets as candidates
    # 2. For each candidate, compute its "degree" = number of conflicts with other candidates
    # 3. Pick the candidate with minimum degree, add to family
    # 4. Remove it and all its neighbors from candidates
    # 5. Repeat until no candidates remain

    # But computing degree for all candidates at each step is expensive.
    # Alternative: use a priority queue with lazy deletion.

    # Even better: since the conflict graph is defined by sunflower triples,
    # let's use the following insight:
    # Two sets a,b are in conflict (can't both be in family) iff there exists c
    # such that inter[a][b] == inter[a][c] == inter[b][c].
    #
    # For a fixed pair (a,b) with intersection v = inter[a][b]:
    # c must satisfy: c ∩ a = v and c ∩ b = v
    # This means c ⊇ v, c ∩ (a\v) = ∅, c ∩ (b\v) = ∅, and |c|=4.
    # So c = v ∪ S where S ⊆ [16]\(a∪b), |S| = 4-|v|.
    #
    # Let's precompute: for each v (bitmask with 0,1,2 bits), and each pair of
    # 4-sets a,b with a∩b=v, determine if there exists a valid c.

    # Precompute all 4-sets indexed by mask
    mask_to_idx = {}
    for i, m in enumerate(masks):
        mask_to_idx[m] = i

    # For each possible v (0 to 2 bits from [16]), precompute the list of 4-sets
    # that contain exactly v as intersection with... no, this is getting complex.

    # Let's just do the direct approach but with numpy for speed.
    import numpy as np

    # Convert masks to numpy array
    masks_np = np.array(masks, dtype=np.uint32)

    # For the greedy algorithm, let's use a simpler but effective approach:
    # Multiple random restarts with different orderings, using fast conflict checking.

    def can_add_fast(c_idx, family_mask):
        """Check if set c_idx can be added. family_mask is a set of indices."""
        cm = masks[c_idx]
        # Group family members by intersection with c
        groups = {}
        for a_idx in family_mask:
            v = inter[a_idx][c_idx]
            if v not in groups:
                groups[v] = []
            groups[v].append(a_idx)

        for v, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                a = group[i]
                for j in range(i + 1, len(group)):
                    b = group[j]
                    if inter[a][b] == v:
                        return False
        return True

    # Build conflict graph more efficiently using the enumeration approach
    # For each v (0,1,2-bit mask), and each pair of disjoint "extensions"
    # Let's enumerate all possible sunflower configurations.

    # A sunflower with core v (|v|<=2) and petals: three 4-sets each containing v
    # and two additional elements, with all three pairs having intersection exactly v.
    #
    # For core v with |v|=0: three 4-sets, pairwise disjoint. 
    # For core v with |v|=1: three 4-sets, each containing the one core element,
    #   and the other 3 elements are pairwise disjoint between sets.
    # For core v with |v|=2: three 4-sets, each containing the two core elements,
    #   and the other 2 elements are pairwise disjoint between sets.

    # Let's build the conflict graph by enumerating all sunflower triples.
    # Number of sunflower triples:
    # |v|=0: C(16,4)*C(12,4)*C(8,4) / 6 ≈ 1820*495*70/6 ≈ 10.3M
    # |v|=1: 16 * C(15,3)^3 / 6 ≈ 16*455^3/6 ≈ 250M  -- too many
    #
    # This is too slow. Let's use a different strategy.

    # Strategy: Use the greedy algorithm with dynamic conflict checking.
    # To speed up, precompute for each set c and each possible intersection value v,
    # the list of sets that have intersection v with c.

    # inter_lists[c][v] = list of set indices a (a != c) where inter[a][c] == v
    # v ranges over all possible 4-bit... no, v is a subset of [16] with |v|<=2 (since |a|=|c|=4)
    # Actually v can be any subset of [16] with |v| <= 2 (intersection of two 4-sets has size 0,1,2)

    # Let's precompute this.
    # For each c, for each v (0,1,2-bit mask), store which sets have that intersection with c.
    # There are C(16,0)+C(16,1)+C(16,2) = 1+16+120 = 137 possible v values.

    # Precompute all possible v masks (0, 1, 2 bits)
    all_v_masks = set()
    all_v_masks.add(0)
    for e in range(1, 17):
        all_v_masks.add(1 << (e-1))
    for e1 in range(1, 17):
        for e2 in range(e1+1, 17):
            all_v_masks.add((1 << (e1-1)) | (1 << (e2-1)))

    v_list = list(all_v_masks)
    v_to_pos = {v: i for i, v in enumerate(v_list)}
    num_v = len(v_list)

    # inter_lists[c][v_pos] = list of a indices where inter[a][c] == v
    # This is 1820 * 137 lists. Memory is fine.
    # But building it: for each c, for each a, compute v = inter[a][c], append a to inter_lists[c][v_pos]
    # That's 1820*1820 = 3.3M operations. Fine.

    inter_lists = [[[] for _ in range(num_v)] for _ in range(n)]
    for c in range(n):
        for a in range(n):
            if a == c:
                continue
            v = inter[a][c]
            vp = v_to_pos[v]
            inter_lists[c][vp].append(a)

    # Now for conflict checking: c conflicts with pair (a,b) in family if
    # there exists v such that a,b are both in inter_lists[c][v] AND inter[a][b] == v.
    #
    # For the greedy algorithm, let's compute for each candidate c, the number of
    # "conflict pairs" in the current candidate pool.
    #
    # Actually, let's just do a well-optimized greedy with multiple strategies.

    # Precompute for fast popcount
    pc = [bin(i).count('1') for i in range(16)]

    # Strategy 1: Greedy by minimum conflict degree
    # For each candidate, degree = number of other candidates that conflict with it.
    # Two candidates a,b conflict if there exists c with inter[a][b]==inter[a][c]==inter[b][c].
    #
    # To compute if a and b conflict: let v = inter[a][b].
    # Need c with inter[a][c]==v and inter[b][c]==v.
    # c must be in inter_lists[a][v] ∩ inter_lists[b][v].
    # And c != a, c != b.
    #
    # So a and b conflict iff (inter_lists[a][v_to_pos[v]] ∩ inter_lists[b][v_to_pos[v]]) is non-empty
    # (excluding a and b themselves, but they're not in their own lists).

    # Precompute intersection of inter_lists for fast conflict checking
    # conflict[a][b] = True if a and b conflict
    # This is 1820^2 = 3.3M entries. Let's compute it.

    # For each pair (a,b), v = inter[a][b], check if inter_lists[a][vp] and inter_lists[b][vp] share an element.
    # To do this fast, use sets.

    # Convert inter_lists to sets for fast intersection
    inter_sets = [[set(lst) for lst in row] for row in inter_lists]

    # Build conflict matrix
    conflict = [[False] * n for _ in range(n)]
    for a in range(n):
        for b in range(a+1, n):
            v = inter[a][b]
            vp = v_to_pos[v]
            if inter_sets[a][vp] & inter_sets[b][vp]:
                conflict[a][b] = True
                conflict[b][a] = True

    # Now degree of a in candidate set S = |{b in S : conflict[a][b]}|
    # Greedy: repeatedly pick min-degree vertex, add to family, remove it and its neighbors.

    def greedy_min_degree(candidate_order=None):
        """Greedy minimum-degree independent set on conflict graph."""
        if candidate_order is None:
            candidate_order = list(range(n))
        
        candidates = set(candidate_order)
        family = []
        
        while candidates:
            if elapsed() > DEADLINE - 15:
                break
            
            # Find minimum degree vertex
            min_deg = float('inf')
            min_vertex = None
            for v in candidates:
                deg = 0
                for u in candidates:
                    if u != v and conflict[v][u]:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_vertex = v
                if min_deg == 0:
                    break
            
            if min_vertex is None:
                break
            
            family.append(min_vertex)
            # Remove min_vertex and all its neighbors from candidates
            to_remove = {min_vertex}
            for u in candidates:
                if conflict[min_vertex][u]:
                    to_remove.add(u)
            candidates -= to_remove
        
        return family

    # The above is O(n^2) per iteration, and n iterations, so O(n^3) = 6B. Too slow.
    # Need a faster approach.

    # Faster: maintain degree counts incrementally.
    def greedy_min_degree_fast(initial_candidates=None):
        """Faster greedy min-degree with incremental degree updates."""
        if initial_candidates is None:
            initial_candidates = set(range(n))
        
        candidates = set(initial_candidates)
        # degree[v] = number of neighbors of v in candidates
        degree = {}
        for v in candidates:
            deg = 0
            for u in candidates:
                if u != v and conflict[v][u]:
                    deg += 1
            degree[v] = deg
        
        family = []
        
        while candidates:
            if elapsed() > DEADLINE - 10:
                break
            
            # Find min degree
            min_deg = float('inf')
            min_vertex = None
            for v in candidates:
                if degree[v] < min_deg:
                    min_deg = degree[v]
                    min_vertex = v
                if min_deg == 0:
                    break
            
            if min_vertex is None:
                break
            
            family.append(min_vertex)
            # Remove min_vertex and its neighbors
            neighbors = [u for u in candidates if u != min_vertex and conflict[min_vertex][u]]
            to_remove = {min_vertex} | set(neighbors)
            
            # Update degrees
            for u in to_remove:
                for w in candidates:
                    if w != u and conflict[u][w]:
                        degree[w] -= 1
                del degree[u]
            
            candidates -= to_remove
        
        return family

    # This is still O(n^2) for initial degree computation and O(n^2) per removal in worst case.
    # Let's try a different approach: use the fact that we just need a good independent set.

    # Alternative: Randomized greedy with multiple passes
    # 1. Randomly order all sets
    # 2. Greedily add sets that don't conflict with already-added sets
    # 3. Repeat with different orderings, keep best

    # For checking if a set conflicts with the family, use the inter_lists approach:
    def can_add_to_family(c, family_set):
        """Check if c can be added to family_set without creating a sunflower."""
        cm = masks[c]
        # For each v, get family members with intersection v with c
        # Check if any pair in that group has mutual intersection v
        for vp in range(num_v):
            v = v_list[vp]
            # Get family members with inter[a][c] == v
            group = [a for a in family_set if inter[a][c] == v]
            if len(group) < 2:
                continue
            # Check pairs
            for i in range(len(group)):
                a = group[i]
                for j in range(i+1, len(group)):
                    b = group[j]
                    if inter[a][b] == v:
                        return False
        return True

    # This is O(|family| * num_v + sum |group|^2) per check.
    # For |family| ~ 100 and groups of size ~50, this is ~100*137 + 137*2500 ≈ 350K per check.
    # Times 1820 checks = 637M. Too slow.

    # Let's use the precomputed inter_lists for faster checking:
    def can_add_fast2(c, family_set):
        """Fast check using precomputed inter_lists."""
        for vp in range(num_v):
            v = v_list[vp]
            # inter_lists[c][vp] contains all a with inter[a][c] == v
            # We need to check if any two in family_set ∩ inter_lists[c][vp] have inter[a][b] == v
            group = [a for a in inter_lists[c][vp] if a in family_set]
            if len(group) < 2:
                continue
            for i in range(len(group)):
                a = group[i]
                for j in range(i+1, len(group)):
                    b = group[j]
                    if inter[a][b] == v:
                        return False
        return True

    # Still slow due to the family_set membership test. Let's use a different approach.

    # Best approach: Precompute the full conflict graph, then use a fast
    # maximum independent set heuristic.

    # Let's use a different strategy entirely: 
    # Since the problem is to find a large independent set in a 3-uniform hypergraph
    # (or equivalently, in the conflict graph), let's use:
    # 1. Randomized greedy with multiple restarts
    # 2. Local search (swap improvements)

    # For speed, let's precompute the neighbor list of each vertex in the conflict graph.
    neighbors = [set() for _ in range(n)]
    for a in range(n):
        for b in range(a+1, n):
            if conflict[a][b]:
                neighbors[a].add(b)
                neighbors[b].add(a)

    def greedy_random(order=None):
        """Greedy: process in given order, add if no conflict with family."""
        if order is None:
            order = list(range(n))
        family = []
        family_set = set()
        for c in order:
            if elapsed() > DEADLINE - 10:
                break
            # Check if c conflicts with any family member
            if not (set(family) & neighbors[c]):
                family.append(c)
                family_set.add(c)
        return family

    # The set intersection check is O(|family|) which is fine.
    # But set(family) creates a new set each time. Let's maintain family_set.

    def greedy_random2(order=None):
        """Greedy with maintained family set."""
        if order is None:
            order = list(range(n))
        family = []
        family_set = set()
        for c in order:
            if elapsed() > DEADLINE - 10:
                break
            # Check if c conflicts with any family member
            conflict_found = False
            for f in family:
                if conflict[c][f]:
                    conflict_found = True
                    break
            if not conflict_found:
                family.append(c)
                family_set.add(c)
        return family

    # This is O(n * |family|) per run. For n=1820, |family|~100, that's 182K per run.
    # We can do many runs.

    # Let's try multiple strategies:
    best_family = []
    best_size = 0
    rng = random.Random(12345)

    # Strategy 1: Natural order
    fam = greedy_random2(list(range(n)))
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = list(fam)

    # Strategy 2: Reverse order
    fam = greedy_random2(list(range(n-1, -1, -1)))
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = list(fam)

    # Strategy 3: Random orders
    for trial in range(50):
        if elapsed() > DEADLINE - 20:
            break
        order = list(range(n))
        rng.shuffle(order)
        fam = greedy_random2(order)
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = list(fam)

    # Strategy 4: Order by degree (low degree first)
    deg_order = sorted(range(n), key=lambda x: len(neighbors[x]))
    fam = greedy_random2(deg_order)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = list(fam)

    # Strategy 5: Order by reverse degree
    deg_order_rev = sorted(range(n), key=lambda x: -len(neighbors[x]))
    fam = greedy_random2(deg_order_rev)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = list(fam)

    # Local search: try to improve by removing one and adding two
    def try_improve(family):
        """Try to improve family by removing one set and adding up to two."""
        if elapsed() > DEADLINE - 5:
            return family
        family_set = set(family)
        improved = False
        best = list(family)
        best_size_local = len(family)
        
        for pos in range(len(family)):
            if elapsed() > DEADLINE - 5:
                break
            removed = family[pos]
            temp = [family[i] for i in range(len(family)) if i != pos]
            temp_set = set(temp)
            
            # Try to add one set
            for c in range(n):
                if c in temp_set or c == removed:
                    continue
                if not (set(temp) & neighbors[c]):
                    new_fam = temp + [c]
                    if len(new_fam) > best_size_local:
                        best = list(new_fam)
                        best_size_local = len(new_fam)
                        improved = True
                        break
            
            # Try to add two sets
            if elapsed() > DEADLINE - 5:
                break
            for c1 in range(n):
                if c1 in temp_set or c1 == removed:
                    continue
                if set(temp) & neighbors[c1]:
                    continue
                new_temp = temp + [c1]
                new_temp_set = set(new_temp)
                for c2 in range(n):
                    if c2 in new_temp_set or c2 == removed or c2 == c1:
                        continue
                    if not (set(new_temp) & neighbors[c2]):
                        new_fam = new_temp + [c2]
                        if len(new_fam) > best_size_local:
                            best = list(new_fam)
                            best_size_local = len(new_fam)
                            improved = True
                        break
                if improved and best_size_local > len(family) + 1:
                    break
            if best_size_local > len(family) + 1:
                break
        
        return best

    # Apply local search
    if elapsed() < DEADLINE - 5:
        improved_family = try_improve(best_family)
        if len(improved_family) > best_size:
            best_family = improved_family
            best_size = len(best_family)

    # Additional: try to add any remaining sets
    if elapsed() < DEADLINE - 3:
        family_set = set(best_family)
        for c in range(n):
            if elapsed() > DEADLINE - 3:
                break
            if c in family_set:
                continue
            if not (set(best_family) & neighbors[c]):
                best_family.append(c)
                family_set.add(c)

    return [sorted(all_sets[i]) for i in best_family]
