import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit

    # Generate all 4-element subsets of [16] as bitmasks (0-indexed bits for 0..15)
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    rng = random.Random(42)
    
    # Precompute pairwise intersections for all mask pairs
    inter_matrix = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        for j in range(i + 1, n_total):
            ij = mi & all_masks[j]
            inter_matrix[i][j] = ij
            inter_matrix[j][i] = ij
    
    # For each pair (i,j), precompute whether they form a "conflict pattern"
    # A sunflower triple (a,b,c) has inter(a,b)==inter(a,c)==inter(b,c)
    # We'll use a different approach: for each new candidate, check conflicts efficiently

    def is_valid_addition_fast(fam_indices, new_idx):
        """Check if all_masks[new_idx] can be added to family without creating a sunflower."""
        n = len(fam_indices)
        if n < 2:
            return True
        
        # Group existing masks by their intersection with new_mask
        groups = defaultdict(list)
        for fam_idx in fam_indices:
            inter = inter_matrix[fam_idx][new_idx]
            groups[inter].append(fam_idx)
        
        # For each group with >= 2 elements, check if any pair has mutual intersection == group key
        for inter, indices in groups.items():
            if len(indices) < 2:
                continue
            for idx_a in range(len(indices)):
                a = indices[idx_a]
                for idx_b in range(idx_a + 1, len(indices)):
                    b = indices[idx_b]
                    if inter_matrix[a][b] == inter:
                        return False
        return True

    best_family = []
    best_size = 0
    best_indices = []

    # Phase 1: Multiple greedy trials with different orderings
    # Use more trials and also try reverse-order and sorted-by-popularity
    num_greedy_trials = 30
    for trial in range(num_greedy_trials):
        if time.time() > deadline - 35:
            break
        
        order = list(range(n_total))
        if trial == 0:
            # First trial: natural order
            pass
        elif trial == 1:
            # Second trial: reverse order
            order.reverse()
        else:
            rng.shuffle(order)
        
        family_indices = []
        for idx in order:
            if is_valid_addition_fast(family_indices, idx):
                family_indices.append(idx)
        
        if len(family_indices) > best_size:
            best_size = len(family_indices)
            best_family = [all_masks[i] for i in family_indices]
            best_indices = list(family_indices)

    family_indices = list(best_indices)
    family_set = set(best_indices)  # Use indices for set membership

    # Phase 2: Fill-in - try to add any remaining sets
    improved = True
    while improved and time.time() < deadline - 12:
        improved = False
        for idx in range(n_total):
            if idx in family_set:
                continue
            if is_valid_addition_fast(family_indices, idx):
                family_indices.append(idx)
                family_set.add(idx)
                improved = True
                if len(family_indices) > best_size:
                    best_size = len(family_indices)
                    best_family = [all_masks[i] for i in family_indices]
                    best_indices = list(family_indices)
                if len(family_indices) >= 250:
                    break
        if len(family_indices) >= 250:
            break

    # Phase 3: Iterated Local Search
    current_indices = list(family_indices)
    current_set = set(family_indices)

    no_improve_count = 0
    max_ils_iters = 100000
    
    for it in range(max_ils_iters):
        if time.time() > deadline - 5:
            break
        if not current_indices:
            break

        # Pick a random set to remove
        remove_pos = rng.randrange(len(current_indices))
        removed_idx = current_indices[remove_pos]
        current_indices.pop(remove_pos)
        current_set.discard(removed_idx)

        # Try to add a different set from a random sample
        added = False
        # Sample more candidates for better exploration
        n_sample = min(800, n_total)
        candidates = rng.sample(range(n_total), n_sample)
        for new_idx in candidates:
            if new_idx in current_set:
                continue
            if is_valid_addition_fast(current_indices, new_idx):
                current_indices.append(new_idx)
                current_set.add(new_idx)
                added = True
                break

        if added:
            if len(current_indices) > best_size:
                best_size = len(current_indices)
                best_family = [all_masks[i] for i in current_indices]
                best_indices = list(current_indices)
            no_improve_count = 0
        else:
            # Restore original
            current_indices.insert(remove_pos, removed_idx)
            current_set.add(removed_idx)
            no_improve_count += 1
            
            # If stuck for too long, restart with a fresh greedy
            if no_improve_count > 3000:
                no_improve_count = 0
                order = list(range(n_total))
                rng.shuffle(order)
                current_indices = []
                current_set = set()
                for idx in order:
                    if is_valid_addition_fast(current_indices, idx):
                        current_indices.append(idx)
                        current_set.add(idx)
                if len(current_indices) > best_size:
                    best_size = len(current_indices)
                    best_family = [all_masks[i] for i in current_indices]
                    best_indices = list(current_indices)

        if best_size >= 250:
            break

    # Phase 4: Final polish - try to add any remaining sets
    if time.time() < deadline - 2:
        final_indices = list(best_indices)
        final_set = set(final_indices)
        for idx in range(n_total):
            if idx in final_set:
                continue
            if is_valid_addition_fast(final_indices, idx):
                final_indices.append(idx)
                final_set.add(idx)
                if len(final_indices) > best_size:
                    best_size = len(final_indices)
                    best_family = [all_masks[i] for i in final_indices]
                    best_indices = list(final_indices)
                if len(final_indices) >= 250:
                    break

    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
