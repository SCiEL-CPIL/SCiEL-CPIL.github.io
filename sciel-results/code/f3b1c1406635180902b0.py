"""Greedy maximal biclique partition of a random graph."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 50.0

    # Build adjacency bitsets (undirected)
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Remaining edges as a set of (min, max) tuples
    rem_edges = set()
    for u, v in edges:
        if u == v:
            continue
        rem_edges.add((min(u, v), max(u, v)))

    # Adjacency restricted to remaining edges
    rem_adj = [0] * n
    for (a, b) in rem_edges:
        rem_adj[a] |= (1 << b)
        rem_adj[b] |= (1 << a)

    def popcount(x):
        return bin(x).count('1')

    def iter_bits(x):
        while x:
            lsb = x & (-x)
            yield lsb.bit_length() - 1
            x ^= lsb

    def grow_biclique(seed_l, seed_r):
        """Grow a maximal biclique containing seed_l and seed_r within remaining edges."""
        # Use bitmask representation throughout
        Lmask = 1 << seed_l
        Rmask = rem_adj[seed_l]
        if not (Rmask & (1 << seed_r)):
            return [], []
        # Ensure seed_r is in Rmask
        Rmask |= (1 << seed_r)
        
        changed = True
        while changed:
            changed = False
            # Common neighborhood of current L in remaining graph
            common = (1 << n) - 1
            temp = Lmask
            while temp:
                lsb = temp & (-temp)
                l = lsb.bit_length() - 1
                common &= rem_adj[l]
                temp ^= lsb
            new_R = common
            if new_R != Rmask:
                Rmask = new_R
                changed = True
            # Now L can be any vertex adjacent to all of Rmask
            if Rmask:
                commonL = (1 << n) - 1
                temp = Rmask
                while temp:
                    lsb = temp & (-temp)
                    r = lsb.bit_length() - 1
                    commonL &= rem_adj[r]
                    temp ^= lsb
                new_L = commonL
                if new_L != Lmask:
                    Lmask = new_L
                    changed = True
            else:
                break
        
        L = list(iter_bits(Lmask))
        R = list(iter_bits(Rmask))
        return L, R

    bicliques = []
    biclique_masks = []  # Store (Lmask, Rmask) for each biclique

    while rem_edges:
        if time.time() > deadline:
            break

        # Find best seed: try each remaining edge, grow, pick max coverage
        best = None
        best_count = -1
        best_Lmask = 0
        best_Rmask = 0
        seed_list = list(rem_edges)

        for (a, b) in seed_list:
            if time.time() > deadline:
                break
            # Try both orientations
            for (sl, sr) in ((a, b), (b, a)):
                L, R = grow_biclique(sl, sr)
                if not L or not R:
                    continue
                # Build masks
                Lmask = 0
                for l in L:
                    Lmask |= (1 << l)
                Rmask = 0
                for r in R:
                    Rmask |= (1 << r)
                
                # Count remaining edges covered by this biclique using bitmasks
                cnt = 0
                temp = Lmask
                while temp:
                    lsb = temp & (-temp)
                    l = lsb.bit_length() - 1
                    # Count neighbors of l in R that are in rem_adj[l]
                    # Only count edges where l < r to avoid double counting
                    common_with_rem = Rmask & rem_adj[l]
                    # Mask out edges where l >= r
                    lt_mask = (1 << l) - 1  # vertices less than l
                    cnt += popcount(common_with_rem & lt_mask)
                    temp ^= lsb
                
                if cnt > best_count:
                    best_count = cnt
                    best = (L, R)
                    best_Lmask = Lmask
                    best_Rmask = Rmask
            # Early stop heuristic
            if best_count >= len(rem_edges):
                break

        if best is None or best_count <= 0:
            # Fallback: single edge
            (a, b) = next(iter(rem_edges))
            best = ([a], [b])
            best_count = 1
            best_Lmask = (1 << a)
            best_Rmask = (1 << b)

        L, R = best
        # Remove covered remaining edges
        Lmask = best_Lmask
        Rmask = best_Rmask
        temp_L = Lmask
        while temp_L:
            lsb_L = temp_L & (-temp_L)
            l = lsb_L.bit_length() - 1
            temp_R = Rmask
            while temp_R:
                lsb_R = temp_R & (-temp_R)
                r = lsb_R.bit_length() - 1
                if l != r:
                    key = (min(l, r), max(l, r))
                    if key in rem_edges:
                        rem_edges.discard(key)
                        rem_adj[l] &= ~(1 << r)
                        rem_adj[r] &= ~(1 << l)
                temp_R ^= lsb_R
            temp_L ^= lsb_L
        bicliques.append([sorted(L), sorted(R)])
        biclique_masks.append((Lmask, Rmask))

    # Fallback: if deadline broke early, cover remaining edges one by one
    if rem_edges:
        for (a, b) in list(rem_edges):
            bicliques.append([[a], [b]])
            rem_edges.discard((a, b))

    # Merge pass: try to merge compatible bicliques to reduce count
    # Two bicliques [L1,R1] and [L2,R2] can be merged if:
    # - L1 and L2 are disjoint, and R1 and R2 are disjoint
    # - All cross edges between (L1∪L2) and (R1∪R2) exist in original graph
    # We'll do a simple greedy merge
    merged = True
    while merged and time.time() < deadline - 5:
        merged = False
        n_bicliques = len(bicliques)
        for i in range(n_bicliques):
            if time.time() > deadline - 5:
                break
            if i >= len(bicliques):
                continue
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline - 5:
                    break
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                # Check disjointness
                set_L1 = set(L1)
                set_R1 = set(R1)
                set_L2 = set(L2)
                set_R2 = set(R2)
                if set_L1 & set_L2 or set_R1 & set_R2:
                    continue
                # Check all cross edges exist in original graph
                adj_set = set()
                for u, v in edges:
                    adj_set.add((min(u, v), max(u, v)))
                
                all_edges_exist = True
                for l in L1:
                    for r in R2:
                        if l != r and (min(l, r), max(l, r)) not in adj_set:
                            all_edges_exist = False
                            break
                    if not all_edges_exist:
                        break
                if all_edges_exist:
                    for l in L2:
                        for r in R1:
                            if l != r and (min(l, r), max(l, r)) not in adj_set:
                                all_edges_exist = False
                                break
                        if not all_edges_exist:
                            break
                
                if all_edges_exist:
                    # Merge: new biclique is [L1∪L2, R1∪R2]
                    new_L = sorted(set_L1 | set_L2)
                    new_R = sorted(set_R1 | set_R2)
                    bicliques[i] = [new_L, new_R]
                    bicliques.pop(j)
                    merged = True
                    break
            if merged:
                break

    # Ensure types: lists of ints
    out = []
    for L, R in bicliques:
        out.append([[int(x) for x in L], [int(y) for y in R]])
    return out
# EVOLVE-BLOCK-END
