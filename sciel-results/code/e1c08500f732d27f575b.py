import time
import random
import numpy as np

def has_k33_fast(adj_bitmasks):
    """
    Check if graph is K3,3-free (no triple has >= 3 common neighbors).
    Returns True if valid.
    """
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

def find_k33(adj_bitmasks):
    """Return (i, j, k, mask) where mask has >=3 bits (a K3,3 witness), or None."""
    n = len(adj_bitmasks)
    for i in range(n):
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            cij = ai & adj_bitmasks[j]
            if cij.bit_count() < 3:
                continue
            for k in range(j + 1, n):
                c = cij & adj_bitmasks[k]
                if c.bit_count() >= 3:
                    return (i, j, k, c)
    return None

def is_add_valid_fast(adj_bitmasks, u, v):
    """
    Fast incremental check: adding edge (u,v) keeps graph K3,3-free.
    Only triples containing u or v can become invalid.
    
    Key insight: For a triple {a,b,c}, the common neighbor set changes only if
    one of a,b,c is u or v. Since the old graph was valid, we only need to check
    whether the NEW common neighbor (v for triples containing u, u for triples 
    containing v) causes the count to reach 3.
    
    For triple {u, i, j}: new common = old_common ∪ {v if v∈N(i)∩N(j)}
    Invalid iff v∈N(i)∩N(j) AND (old_u ∩ N(i) ∩ N(j)).bit_count() >= 2
    
    For triple {v, i, j}: new common = old_common ∪ {u if u∈N(i)∩N(j)}  
    Invalid iff u∈N(i)∩N(j) AND (old_v ∩ N(i) ∩ N(j)).bit_count() >= 2
    
    For triple {u, v, i}: new common = old_common ∪ {v if i∈N(v)} ∪ {u if i∈N(u)}
    Invalid iff (old_u ∩ old_v ∩ N(i)).bit_count() + [i∈old_v] + [i∈old_u] >= 3
    """
    n = len(adj_bitmasks)
    old_u = adj_bitmasks[u]
    old_v = adj_bitmasks[v]
    
    # Check triples {u, i, j} where i,j != u,v, i < j
    # Need: v ∈ N(i) ∩ N(j) AND (old_u ∩ N(i) ∩ N(j)).bit_count() >= 2
    # v ∈ N(i) means i is a neighbor of v, i.e., i ∈ old_v
    # So i,j must be in old_v (neighbors of v), and i != u, j != u, i != v, j != v
    # Since u-v is not an edge, u ∉ old_v and v ∉ old_v automatically.
    
    # Get neighbors of v (excluding u and v themselves, which aren't in old_v anyway)
    neighbors_of_v = old_v  # bitmask of vertices adjacent to v
    
    # For each pair (i, j) both in neighbors_of_v, i < j, i != u, j != u:
    # check if (old_u & adj_bitmasks[i] & adj_bitmasks[j]).bit_count() >= 2
    # But we also need v ∈ N(i) ∩ N(j), which is guaranteed since i,j ∈ old_v means
    # i and j are both neighbors of v, so v ∈ N(i) and v ∈ N(j).
    
    # Efficient: iterate over pairs in neighbors_of_v
    # But that could be O(d(v)^2) which is fine for d(v) <= 19.
    
    # Actually, let's just iterate i from 0 to n-1, j from i+1 to n-1,
    # skipping u and v, and check if both are in neighbors_of_v.
    
    for i in range(n):
        if i == u or i == v:
            continue
        if not (neighbors_of_v >> i) & 1:
            continue
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            if j == u or j == v:
                continue
            if not (neighbors_of_v >> j) & 1:
                continue
            common = old_u & ai & adj_bitmasks[j]
            if common.bit_count() >= 2:
                return False
    
    # Check triples {v, i, j} where i,j != u,v, i < j
    # Need: u ∈ N(i) ∩ N(j) AND (old_v ∩ N(i) ∩ N(j)).bit_count() >= 2
    # u ∈ N(i) means i is a neighbor of u, i.e., i ∈ old_u
    neighbors_of_u = old_u
    
    for i in range(n):
        if i == u or i == v:
            continue
        if not (neighbors_of_u >> i) & 1:
            continue
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            if j == u or j == v:
                continue
            if not (neighbors_of_u >> j) & 1:
                continue
            common = old_v & ai & adj_bitmasks[j]
            if common.bit_count() >= 2:
                return False
    
    # Check triples {u, v, i} where i != u, i != v
    # New common = (old_u ∩ old_v ∩ N(i)) ∪ {v if i∈old_v} ∪ {u if i∈old_u}
    # Count = (old_u ∩ old_v ∩ N(i)).bit_count() + (1 if i∈old_v) + (1 if i∈old_u)
    # Note: v ∉ old_u, u ∉ old_v, so {v} and {u} are disjoint from old_u ∩ old_v.
    # Also i ∉ old_u ∪ old_v would give count = 0 + 0 + 0 = 0, fine.
    
    for i in range(n):
        if i == u or i == v:
            continue
        base = (old_u & old_v & adj_bitmasks[i]).bit_count()
        extra = 0
        if (old_v >> i) & 1:
            extra += 1
        if (old_u >> i) & 1:
            extra += 1
        if base + extra >= 3:
            return False
    
    return True

def is_add_valid(adj_bitmasks, u, v):
    """
    Check if adding edge (u,v) keeps the graph K3,3-free.
    Only triples containing u or v can become invalid.
    Uses the optimized check from is_add_valid_fast.
    """
    return is_add_valid_fast(adj_bitmasks, u, v)

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s

    best_edges = 0
    best_adj_bitmasks = None

    def bitmasks_to_edges(bitmasks):
        edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    def count_edges(bitmasks):
        return sum(b.bit_count() for b in bitmasks) // 2

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
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    all_edges.append((i, j))
                m -= lsb
        return all_edges

    def greedy_add_exhaustive(bitmasks, time_limit=None):
        """Add any valid non-edge repeatedly until none remain (exhaustive)."""
        if time_limit is None:
            time_limit = deadline
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            non_edges = get_non_edges(bitmasks)
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() > time_limit:
                    break
                if is_add_valid(bitmasks, u, v):
                    bitmasks[u] |= (1 << v)
                    bitmasks[v] |= (1 << u)
                    changed = True
        return bitmasks

    def local_search(initial_bitmasks):
        """Local search: greedy add + targeted swap moves to escape local optima."""
        bitmasks = initial_bitmasks[:]
        current_edges = count_edges(bitmasks)
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]

        # First, greedily add valid edges
        bitmasks = greedy_add_exhaustive(bitmasks)
        current_edges = count_edges(bitmasks)
        if current_edges > best_local_edges:
            best_local_edges = current_edges
            best_local_bitmasks = bitmasks[:]

        no_improve = 0
        while time.time() < deadline:
            moved = False

            # Try to add edges: exhaustive pass
            non_edges = get_non_edges(bitmasks)
            if non_edges:
                random.shuffle(non_edges)
                for u, v in non_edges:
                    if time.time() > deadline:
                        break
                    if is_add_valid(bitmasks, u, v):
                        bitmasks[u] |= (1 << v)
                        bitmasks[v] |= (1 << u)
                        current_edges += 1
                        moved = True
                        if current_edges > best_local_edges:
                            best_local_edges = current_edges
                            best_local_bitmasks = bitmasks[:]

            if not moved:
                # Targeted 1-for-1 swap: find a K3,3 witness, remove an edge from it,
                # then try to add a non-edge that was blocked.
                witness = find_k33(bitmasks)
                if witness is None:
                    # Already K3,3-free (shouldn't happen if we maintain validity)
                    break

                i, j, k, mask = witness
                # The K3,3 has parts {i,j,k} and 3 neighbors in mask.
                # Remove one of the 9 edges incident to this K3,3.
                # Candidate edges to remove: (i,x), (j,x), (k,x) for x in mask.
                mask_bits = []
                m = mask
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    mask_bits.append(x)
                    m -= lsb

                remove_candidates = []
                for part in (i, j, k):
                    for x in mask_bits:
                        if bitmasks[part] & (1 << x):
                            remove_candidates.append((part, x))

                if remove_candidates:
                    random.shuffle(remove_candidates)
                    non_edges = get_non_edges(bitmasks)
                    random.shuffle(non_edges)

                    for a, b in remove_candidates:
                        if time.time() > deadline:
                            break
                        # Try removing (a,b) and adding various non-edges
                        for c, d in non_edges[:20]:
                            if time.time() > deadline:
                                break
                            bm2 = bitmasks[:]
                            bm2[a] &= ~(1 << b)
                            bm2[b] &= ~(1 << a)
                            if is_add_valid(bm2, c, d):
                                bm2[c] |= (1 << d)
                                bm2[d] |= (1 << c)
                                # Verify full validity
                                if has_k33_fast(bm2):
                                    bitmasks = bm2
                                    current_edges = count_edges(bitmasks)
                                    moved = True
                                    if current_edges > best_local_edges:
                                        best_local_edges = current_edges
                                        best_local_bitmasks = bitmasks[:]
                                    break
                        if moved:
                            break

            if not moved:
                # Random 1-for-1 swap as fallback
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if all_edges and non_edges:
                    for _ in range(30):
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
                # Try 2-edge swap: remove 2 edges, add 2 different edges
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if len(all_edges) >= 2 and len(non_edges) >= 2:
                    for attempt in range(50):
                        if time.time() > deadline:
                            break
                        # Pick two edges to remove
                        e1 = random.choice(all_edges)
                        e2 = random.choice(all_edges)
                        if e1 == e2:
                            continue
                        # Pick two non-edges to add
                        ne1 = random.choice(non_edges)
                        ne2 = random.choice(non_edges)
                        if ne1 == ne2:
                            continue
                        # Ensure no conflicts (no shared vertices between removals and additions)
                        removed_verts = set()
                        for a, b in (e1, e2):
                            removed_verts.add(a)
                            removed_verts.add(b)
                        added_verts = set()
                        for a, b in (ne1, ne2):
                            added_verts.add(a)
                            added_verts.add(b)
                        if removed_verts & added_verts:
                            continue
                        
                        bm2 = bitmasks[:]
                        # Remove e1
                        a, b = e1
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        # Remove e2
                        a, b = e2
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        # Try adding ne1
                        c, d = ne1
                        if is_add_valid(bm2, c, d):
                            bm2[c] |= (1 << d)
                            bm2[d] |= (1 << c)
                            # Try adding ne2
                            c, d = ne2
                            if is_add_valid(bm2, c, d):
                                bm2[c] |= (1 << d)
                                bm2[d] |= (1 << c)
                                # Verify full validity
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
                if no_improve > 60:
                    break
            else:
                no_improve = 0

        return best_local_bitmasks, best_local_edges

    # Multiple restarts with varied initializations
    num_restarts = 150
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break

        if r == 0:
            # Start empty, greedy add
            init_bitmasks = [0] * n
        elif r == 1:
            # Start with a random moderate graph, repair, then greedy
            init_bitmasks = [0] * n
            p = 0.35
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
            # Repair until valid using targeted removal
            attempts = 0
            while not has_k33_fast(init_bitmasks) and attempts < 5000:
                witness = find_k33(init_bitmasks)
                if witness is None:
                    break
                i, j, k, mask = witness
                # Remove a random edge from this K3,3
                mask_bits = []
                m = mask
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    mask_bits.append(x)
                    m -= lsb
                part = random.choice((i, j, k))
                x = random.choice(mask_bits)
                init_bitmasks[part] &= ~(1 << x)
                init_bitmasks[x] &= ~(1 << part)
                attempts += 1
        else:
            # Random sparse start
            init_bitmasks = [0] * n
            p = 0.15 + 0.015 * (r % 10)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
            # Repair until valid using targeted removal
            attempts = 0
            while not has_k33_fast(init_bitmasks) and attempts < 5000:
                witness = find_k33(init_bitmasks)
                if witness is None:
                    break
                i, j, k, mask = witness
                mask_bits = []
                m = mask
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    mask_bits.append(x)
                    m -= lsb
                part = random.choice((i, j, k))
                x = random.choice(mask_bits)
                init_bitmasks[part] &= ~(1 << x)
                init_bitmasks[x] &= ~(1 << part)
                attempts += 1

        if not has_k33_fast(init_bitmasks):
            continue

        # Quick greedy pass first to get a baseline
        quick_bm = init_bitmasks[:]
        quick_bm = greedy_add_exhaustive(quick_bm, time_limit=min(deadline - 5, time.time() + 2))
        quick_edges = count_edges(quick_bm)
        
        if quick_edges >= best_edges:
            # Worth doing full local search
            local_best_bitmasks, local_best_edges = local_search(init_bitmasks)
        else:
            # Skip full local search if quick greedy doesn't beat best
            local_best_bitmasks, local_best_edges = quick_bm, quick_edges

        if local_best_edges > best_edges:
            best_edges = local_best_edges
            best_adj_bitmasks = local_best_bitmasks[:]

    # Final: one more greedy pass on the best found, then a short local search polish
    if best_adj_bitmasks is not None:
        improved = greedy_add_exhaustive(best_adj_bitmasks[:], time_limit=min(deadline - 5, time.time() + 10))
        improved_edges = count_edges(improved)
        if improved_edges > best_edges:
            best_edges = improved_edges
            best_adj_bitmasks = improved
        
        # Short local search polish on the best
        if time.time() < deadline - 10:
            polished_bm, polished_edges = local_search(best_adj_bitmasks[:])
            if polished_edges > best_edges:
                best_edges = polished_edges
                best_adj_bitmasks = polished_bm

    if best_adj_bitmasks is None:
        return []

    return bitmasks_to_edges(best_adj_bitmasks)

# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Found {len(edges)} edges in {t1-t0:.2f}s")
    # Verify K3,3-free
    n = 20
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                common = adj[i] & adj[j] & adj[k]
                if np.sum(common) >= 3:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid K3,3-free: {valid}")
