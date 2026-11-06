"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    # Generate all 4-subsets of [1,16] as bitmasks
    all_masks = []
    for row in combinations(range(1, 17), 4):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_masks.append(mask)

    n = len(all_masks)  # C(16,4) = 1820

    # Precompute pairwise intersections as bitmasks
    inter = [[0] * n for _ in range(n)]
    for i in range(n):
        mi = all_masks[i]
        for j in range(i + 1, n):
            ij = mi & all_masks[j]
            inter[i][j] = ij
            inter[j][i] = ij

    # Precompute sunflower triple membership:
    # For each pair (i,j), precompute the set of k such that (i,j,k) is a sunflower
    # sunflower_partner[i][j] = frozenset of k != i,j where inter[i][j]==inter[i][k]==inter[j][k]
    # This is O(n^3) = ~6 billion... too slow.
    
    # Alternative: For each index i, build a dict: intersection_value -> list of j's
    # Then for pair (i,j) with val=inter[i][j], partners are in dict_i[val] ∩ dict_j[val]
    
    # Build per-index intersection dictionaries
    inter_dict = [{} for _ in range(n)]
    for i in range(n):
        d = inter_dict[i]
        for j in range(n):
            if i == j:
                continue
            v = inter[i][j]
            if v not in d:
                d[v] = []
            d[v].append(j)
    
    # Now for a pair (i,j), sunflower partners k are in inter_dict[i][inter[i][j]] ∩ inter_dict[j][inter[i][j]]
    # minus {i,j}
    
    # For the greedy, maintain:
    # - chosen: list of chosen indices
    # - chosen_set: set of chosen indices
    # - For each candidate x, conflict_count[x] = number of pairs (i,j) in chosen such that (x,i,j) is sunflower
    #   Actually we need: when adding x, check if there exist ANY pair (i,j) in chosen with (x,i,j) sunflower
    #   So we just need a boolean: is_conflicting[x] = True if x conflicts with chosen
    
    # When we add new element c to chosen:
    # For each remaining candidate x, x becomes conflicting if there exists i in chosen_before_c
    # such that (x, i, c) is a sunflower, i.e., inter[x][i] == inter[x][c] == inter[i][c]
    
    # Let's precompute for each pair (i,c), the set of x such that (x,i,c) is sunflower
    # That's: inter[x][i] == inter[i][c] AND inter[x][c] == inter[i][c]
    # So x must be in inter_dict[i][inter[i][c]] ∩ inter_dict[c][inter[i][c]] minus {i,c}
    
    # Precompute sunflower_pairs[i][c] = frozenset of x (x != i, x != c) where (x,i,c) is sunflower
    # This is O(n^2) computations, each involving intersection of two lists.
    # Total work: sum over all pairs of |list_i| * |list_c| which could be large but let's try.
    
    # Actually, let's just compute on the fly during greedy but cache results.
    
    def get_sunflower_partners(i, j):
        """Return set of k (k!=i, k!=j) such that (i,j,k) is a sunflower triple."""
        v = inter[i][j]
        li = inter_dict[i].get(v, [])
        lj = inter_dict[j].get(v, [])
        # Find intersection
        if len(li) < len(lj):
            common = [x for x in li if x in set(lj)]
        else:
            common = [x for x in lj if x in set(li)]
        common_set = set(common)
        common_set.discard(i)
        common_set.discard(j)
        return common_set
    
    # Cache for sunflower partner lookups
    sf_partner_cache = {}
    
    def get_sf_partners_cached(i, j):
        key = (min(i,j), max(i,j))
        if key not in sf_partner_cache:
            sf_partner_cache[key] = get_sunflower_partners(i, j)
        return sf_partner_cache[key]
    
    def can_add_fast(chosen_list, x):
        """Check if x can be added to chosen_list without forming a sunflower."""
        m = len(chosen_list)
        for idx1 in range(m):
            i = chosen_list[idx1]
            for idx2 in range(idx1 + 1, m):
                j = chosen_list[idx2]
                if is_sunflower_fast(x, i, j):
                    return False
        return True
    
    def is_sunflower_fast(i, j, k):
        """Check if i, j, k form a sunflower triple."""
        return inter[i][j] == inter[i][k] == inter[j][k]
    
    # Greedy strategy: pick elements in order of increasing "conflict potential"
    # Conflict potential of x = number of pairs (i,j) in all elements such that (x,i,j) is sunflower
    # But computing this for all x is O(n^3). Instead, use random orders with smart selection.
    
    def greedy_random_order(order, deadline):
        """Greedy: try to add elements in the given order."""
        chosen = []
        chosen_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in chosen_set:
                continue
            # Check if x can be added
            m = len(chosen)
            valid = True
            for idx1 in range(m):
                i = chosen[idx1]
                for idx2 in range(idx1 + 1, m):
                    j = chosen[idx2]
                    if inter[x][i] == inter[x][j] == inter[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                chosen.append(x)
                chosen_set.add(x)
        return chosen
    
    # Run multiple greedy passes with different random orders
    best = []
    best_size = 0
    
    # Natural order first
    if time.time() < deadline:
        fam = greedy_random_order(list(range(n)), deadline)
        if len(fam) > best_size:
            best = fam
            best_size = len(fam)
    
    # Multiple random orders
    rng = random.Random(12345)
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(range(n))
        rng.shuffle(order)
        fam = greedy_random_order(order, deadline)
        if len(fam) > best_size:
            best = fam
            best_size = len(fam)
    
    # Local search: try to improve by swapping
    if time.time() < deadline:
        current = list(best)
        current_set = set(best)
        
        # Precompute for current set: for each pair in current, the sunflower partners
        # Then for each candidate x not in current, count how many pairs in current it conflicts with
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            m = len(current)
            
            # For each candidate x not in current, compute conflict count
            # conflict_count[x] = number of pairs (i,j) in current where (x,i,j) is sunflower
            conflict_counts = {}
            for x in range(n):
                if x in current_set:
                    continue
                cc = 0
                for idx1 in range(m):
                    i = current[idx1]
                    for idx2 in range(idx1 + 1, m):
                        j = current[idx2]
                        if inter[x][i] == inter[x][j] == inter[i][j]:
                            cc += 1
                if cc > 0:
                    conflict_counts[x] = cc
            
            # Try swaps: for each r in current, for each x not in current with conflict_counts[x] small
            # Try removing r and adding x if valid
            # To speed up, only try x with conflict_counts[x] <= 1 (or some threshold)
            
            # Sort candidates by conflict count
            candidates = sorted(conflict_counts.items(), key=lambda t: t[1])
            
            for r_idx in range(m):
                if time.time() > deadline:
                    break
                r = current[r_idx]
                remaining = [current[k] for k in range(m) if k != r_idx]
                remaining_set = set(remaining)
                
                for x, cc in candidates:
                    if x in current_set:
                        continue
                    # Check if (remaining + [x]) is valid
                    # Need: no sunflower triple in remaining ∪ {x}
                    # Since remaining is valid (subset of valid current), only need to check triples involving x
                    valid = True
                    for idx1 in range(len(remaining)):
                        i = remaining[idx1]
                        for idx2 in range(idx1 + 1, len(remaining)):
                            j = remaining[idx2]
                            if inter[x][i] == inter[x][j] == inter[i][j]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Perform swap
                        current[r_idx] = x
                        current_set.discard(r)
                        current_set.add(x)
                        improved = True
                        break
                if improved:
                    break
        
        if len(current) > best_size:
            best = current
            best_size = len(current)
    
    # Convert back to sorted lists of elements
    result = []
    for x in best:
        m = all_masks[x]
        elems = []
        for i in range(1, 17):
            if (m >> i) & 1:
                elems.append(i)
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
