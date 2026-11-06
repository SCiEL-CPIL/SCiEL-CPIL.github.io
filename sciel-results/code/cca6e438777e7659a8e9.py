import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill

    # Generate all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    n_total = len(all_masks)  # 1820

    # Precompute pairwise intersections
    # inter[i][j] = all_masks[i] & all_masks[j]
    # Store as a list of lists for fast access
    inter = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        for j in range(i + 1, n_total):
            ij = mi & all_masks[j]
            inter[i][j] = ij
            inter[j][i] = ij

    # A triple (i, j, k) is a sunflower if inter[i][j] == inter[i][k] == inter[j][k]
    # We want to find a large subset with no such triples.
    # This is equivalent to finding a maximum clique in a graph where vertices are
    # masks and edges connect pairs that "don't conflict" - but the conflict is
    # transitive (depends on 3 vertices). So it's not a standard graph problem.
    
    # Instead, use a greedy randomized approach with local search.
    # Key insight: for a given candidate mask m, we can track which pairs in the
    # current family would form a sunflower with m.

    def can_add(family, m_idx, family_set):
        """Check if mask m_idx can be added to family without forming a sunflower."""
        m = all_masks[m_idx]
        # For each pair (a, b) in family, check if inter[a][b] == inter[a][m] == inter[b][m]
        # This is O(|F|^2) per check, but we can optimize.
        
        # Alternative: group by intersection with m.
        # If two sets a, b in family have the same intersection I with m,
        # then we need inter[a][b] != I.
        groups = {}
        for a_idx in family_set:
            a = all_masks[a_idx]
            inter_a_m = a & m
            if inter_a_m in groups:
                # Check all existing b in this group
                for b_idx in groups[inter_a_m]:
                    b = all_masks[b_idx]
                    if inter[a_idx][b_idx] == inter_a_m:
                        return False
                groups[inter_a_m].append(a_idx)
            else:
                groups[inter_a_m] = [a_idx]
        return True

    rng = random.Random(12345)
    best_family = []
    best_size = 0

    # Multiple random greedy passes
    for iteration in range(500):
        if time.time() > deadline - 5:
            break

        shuffled = list(range(n_total))
        rng.shuffle(shuffled)

        family = []
        family_set = set()
        for idx in shuffled:
            if can_add(family, idx, family_set):
                family.append(idx)
                family_set.add(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)

        if best_size >= 250:
            break

    # Local search: try to add more sets to the best family
    if time.time() < deadline - 3 and best_family:
        family = list(best_family)
        family_set = set(family)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(n_total):
                if time.time() > deadline - 2:
                    break
                if idx in family_set:
                    continue
                if can_add(family, idx, family_set):
                    family.append(idx)
                    family_set.add(idx)
                    improved = True

        if len(family) > best_size:
            best_family = family
            best_size = len(family)

    # Convert bitmasks back to sorted lists of 1-indexed elements
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
