import itertools
import random


def construct(v, k, t):
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # For each k-subset, compute the list of t-subset indices it covers
    k_to_t_indices = []
    for k_sub in all_k_subsets:
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            t_indices.append(t_subset_to_idx[t_sub])
        k_to_t_indices.append(t_indices)
        
    # Reverse mapping: for each t-subset index, which k-subsets cover it
    t_to_k_indices = [[] for _ in range(num_t)]
    for ki, t_indices in enumerate(k_to_t_indices):
        for ti in t_indices:
            t_to_k_indices[ti].append(ki)
            
    max_cov = len(k_to_t_indices[0]) if num_k > 0 else 0

    def run_greedy(order):
        coverage = [max_cov] * num_k
        uncovered = [True] * num_t
        selected = [False] * num_k
        uncovered_count = num_t
        cover_count = [0] * num_t
        
        # Initialize buckets
        buckets = [[] for _ in range(max_cov + 1)]
        for ki in order:
            buckets[max_cov].append(ki)
            
        current_max = max_cov
        selected_indices = []
        
        while uncovered_count > 0:
            # Find the highest non-empty bucket
            while current_max >= 0 and not buckets[current_max]:
                current_max -= 1
            
            if current_max < 0:
                break
                
            ki = buckets[current_max].pop()
            
            if selected[ki]:
                continue
                
            if coverage[ki] != current_max:
                # Stale entry, move to correct bucket
                buckets[coverage[ki]].append(ki)
                continue
                
            if coverage[ki] == 0:
                continue
            
            # Select this k-subset
            selected[ki] = True
            selected_indices.append(ki)
            
            # Update coverage for neighbors and cover_count
            for ti in k_to_t_indices[ki]:
                cover_count[ti] += 1
                if uncovered[ti]:
                    uncovered[ti] = False
                    uncovered_count -= 1
                    for other_ki in t_to_k_indices[ti]:
                        if not selected[other_ki]:
                            coverage[other_ki] -= 1
                            
        return selected_indices, cover_count

    def reduce_cover(selected_indices, cover_count):
        # selected_indices is a list of ki
        # cover_count is the count for this specific set
        # We modify selected_indices and cover_count to remove redundant blocks
        changed = True
        while changed:
            changed = False
            to_remove = []
            for idx, ki in enumerate(selected_indices):
                # Check if ki is redundant
                # Redundant if for all ti in ki, cover_count[ti] > 1
                redundant = True
                for ti in k_to_t_indices[ki]:
                    if cover_count[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(idx)
            
            if not to_remove:
                break
                
            changed = True
            # Remove them in reverse order to keep indices valid
            for idx in reversed(to_remove):
                ki = selected_indices[idx]
                for ti in k_to_t_indices[ki]:
                    cover_count[ti] -= 1
                selected_indices.pop(idx)
                
        return selected_indices

    # Run multiple times with different orders
    best_blocks = None
    best_len = float('inf')
    
    # Deterministic run
    order = list(range(num_k))
    indices, counts = run_greedy(order)
    reduced_indices = reduce_cover(indices, counts)
    if len(reduced_indices) < best_len:
        best_len = len(reduced_indices)
        best_blocks = [list(all_k_subsets[i]) for i in reduced_indices]
        
    # Randomized runs
    num_restarts = 50 # Increase restarts as reduction is fast
    for _ in range(num_restarts):
        random_order = list(range(num_k))
        random.shuffle(random_order)
        indices, counts = run_greedy(random_order)
        reduced_indices = reduce_cover(indices, counts)
        if len(reduced_indices) < best_len:
            best_len = len(reduced_indices)
            best_blocks = [list(all_k_subsets[i]) for i in reduced_indices]
            
    return best_blocks
