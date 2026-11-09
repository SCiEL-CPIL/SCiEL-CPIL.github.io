"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Generate all 4-element subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n = len(all_sets)  # 1820
    
    # Precompute intersection sizes between all pairs
    # For each pair (i, j), store the intersection size
    # We need to check if three sets have all pairwise intersections equal
    
    # Convert to frozensets for faster intersection
    set_list = [frozenset(s) for s in all_sets]
    
    # Precompute intersection sizes as a matrix
    # inter[i][j] = len(set_i & set_j)
    inter = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = len(set_list[i] & set_list[j])
            inter[i][j] = val
            inter[j][i] = val
    
    # A sunflower of three sets A, B, C exists if:
    # A∩B = A∩C = B∩C (as sets)
    # For 4-element sets, this means all three pairwise intersections are equal as sets.
    # Since we're dealing with 4-element sets, the intersection can have size 0, 1, 2, or 3.
    # If all three pairwise intersections are equal, they must all be the same set.
    
    # Key insight: For three 4-element sets, if A∩B = A∩C = B∩C = S, then:
    # - |S| can be 0, 1, 2, or 3
    # - If |S| = 3, then all three sets share 3 elements, meaning they differ by only 1 element each
    # - We need to check if the actual intersection sets are equal, not just sizes
    
    # For efficiency, let's precompute for each pair the actual intersection
    # But that's expensive. Instead, let's use a different approach.
    
    # Alternative: For each potential "core" S (subset of [1,16]), 
    # find all 4-sets that contain S. If more than 2 such sets exist and 
    # their pairwise intersections are all equal to S, we have a sunflower.
    
    # Actually, let's think more carefully. A sunflower with core S means:
    # A = S ∪ {a}, B = S ∪ {b}, C = S ∪ {c} where a, b, c are distinct and not in S
    # Then A∩B = S, A∩C = S, B∩C = S. So all pairwise intersections equal S.
    
    # So for each subset S of [1,16] with |S| ≤ 3 (since sets are size 4),
    # we can form sunflowers by choosing 3 distinct elements a, b, c not in S
    # and creating sets S∪{a}, S∪{b}, S∪{c}.
    
    # Wait, that's not quite right. The sets don't have to be of the form S∪{x}.
    # They just need to have pairwise intersections equal to S.
    
    # Let me reconsider. If A∩B = A∩C = B∩C = S, then:
    # - S ⊆ A, S ⊆ B, S ⊆ C
    # - A\S, B\S, C\S are disjoint (otherwise there would be additional common elements)
    # - So A = S ∪ A', B = S ∪ B', C = S ∪ C' where A', B', C' are disjoint non-empty sets
    # - |A| = |S| + |A'| = 4, so |A'| = 4 - |S|
    # - Similarly for B' and C'
    # - A', B', C' are pairwise disjoint
    
    # So for a given core S with |S| = k, we need to choose three disjoint subsets 
    # A', B', C' of [1,16]\S, each of size 4-k, such that:
    # - A', B', C' are pairwise disjoint
    # - The sets S∪A', S∪B', S∪C' are in our family
    
    # This is getting complex. Let me use a simpler approach for the conflict detection.
    
    # For the simulated annealing, I need to efficiently check if adding a set creates a sunflower.
    # When adding set T to a family F, a sunflower is created if there exist A, B in F 
    # such that A∩B = A∩T = B∩T.
    
    # Let me precompute for each pair of sets their intersection.
    # Then when checking if T can be added, for each pair (A, B) in F,
    # check if A∩B == A∩T == B∩T.
    
    # This is O(|F|^2) per addition, which might be slow for large families.
    
    # Let me try a different approach: use bitmasks for efficiency.
    
    # Represent each 4-element subset as a 16-bit integer bitmask.
    bitmask_list = []
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        bitmask_list.append(mask)
    
    # Precompute intersection masks
    # inter_mask[i][j] = bitmask_list[i] & bitmask_list[j]
    # But this is 1820*1820 = 3.3 million entries, each 16 bits. That's manageable.
    
    # Actually, let's just compute on the fly with bit operations, which are fast.
    
    # Simulated annealing approach:
    # Start with an empty family, try to add sets one by one.
    # Or start with a random subset and try to flip elements.
    
    # Let's use a greedy + local search approach:
    # 1. Greedily build a family
    # 2. Try to improve by removing and re-adding sets
    
    # First, let's implement a fast conflict check.
    # Given a family F (list of indices into all_sets) and a candidate set T (index),
    # check if adding T creates a sunflower.
    
    def creates_sunflower(family, t_idx):
        """Check if adding set t_idx to family creates a sunflower."""
        t_mask = bitmask_list[t_idx]
        # For each pair (a, b) in family, check if a∩b == a∩t == b∩t
        for i in range(len(family)):
            a_idx = family[i]
            a_mask = bitmask_list[a_idx]
            a_t_inter = a_mask & t_mask
            for j in range(i + 1, len(family)):
                b_idx = family[j]
                b_mask = bitmask_list[b_idx]
                b_t_inter = b_mask & t_mask
                a_b_inter = a_mask & b_mask
                if a_b_inter == a_t_inter == b_t_inter:
                    return True
        return False
    
    # Greedy construction
    def greedy_construct():
        family = []
        # Try adding sets in a random order
        indices = list(range(n))
        random.shuffle(indices)
        for idx in indices:
            if not creates_sunflower(family, idx):
                family.append(idx)
        return family
    
    # Local search: try to improve by removing one set and adding another
    def local_search(family, max_iterations=1000):
        family = list(family)
        in_family = [False] * n
        for idx in family:
            in_family[idx] = True
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            # Try removing each set and see if we can add a better one
            for i in range(len(family)):
                if time.time() >= deadline:
                    break
                # Remove family[i]
                removed = family[i]
                in_family[removed] = False
                new_family = [x for j, x in enumerate(family) if j != i]
                
                # Try to add a set not in the family
                # Find candidates that don't create sunflowers
                best_add = -1
                for idx in range(n):
                    if in_family[idx]:
                        continue
                    if not creates_sunflower(new_family, idx):
                        best_add = idx
                        break
                
                if best_add != -1:
                    # Check if this swap improves something (for now, just accept)
                    # Actually, for maximization, we want to keep the size the same or larger
                    # Since we're swapping one for one, size stays the same
                    # Let's try to find a case where we can add without removing
                    pass
                
                # Restore
                in_family[removed] = True
                family = list(family)  # This is getting complicated
        
        return family
    
    # Let's use a simpler approach: multiple random greedy runs
    best_family = []
    num_runs = 50
    for run in range(num_runs):
        if time.time() >= deadline:
            break
        family = greedy_construct()
        if len(family) > len(best_family):
            best_family = family
    
    # Try to improve best_family by adding more sets
    if best_family:
        in_best = [False] * n
        for idx in best_family:
            in_best[idx] = True
        
        # Try to add more sets
        for idx in range(n):
            if time.time() >= deadline:
                break
            if not in_best[idx]:
                if not creates_sunflower(best_family, idx):
                    best_family.append(idx)
                    in_best[idx] = True
    
    # Convert back to sorted lists
    result = [sorted(all_sets[idx]) for idx in best_family]
    
    return result

# EVOLVE-BLOCK-END
