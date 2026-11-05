import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # 150s kill limit. Use 135s as deadline to be safe (15s margin).
    deadline = start_time + 135.0

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    # Precompute all k-blocks and their covered triplets
    # This is C(24,7) = 346,104 blocks. Each covers C(7,3) = 35 triplets.
    # Total entries: 346104 * 35 = ~12M. This is manageable.
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_all_blocks = len(all_k_blocks)
    
    # Map each block index to the list of triplet indices it covers
    block_triplets = []
    for b in all_k_blocks:
        tids = [get_id(triplet) for triplet in itertools.combinations(b, t)]
        block_triplets.append(tids)
        
    # Map each triplet index to the list of block indices covering it
    # This is needed for efficient gain updates (reverse index)
    triple_blocks = [[] for _ in range(n_triplets)]
    for b_idx, tids in enumerate(block_triplets):
        for t_idx in tids:
            triple_blocks[t_idx].append(b_idx)

    def greedy_construct_ratio(v, k, t, seed=0, bias=0.0, use_ratio=True):
        """
        Greedy construction using precomputed blocks.
        Selects block maximizing gain (or gain/k if use_ratio) among uncovered triplets.
        bias: 0.0 means pure greedy (max gain). 1.0 means random.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_triplets, dtype=np.int32)
        # gain[b] is the number of currently uncovered triplets covered by block b
        gain = np.array([len(ts) for ts in block_triplets], dtype=np.int32)
        # If using ratio, we might want to normalize by k, but k is constant, 
        # so max(gain) == max(gain/k). However, the strategy suggests "ratio of newly covered 
        # to number of points". Since all blocks have size k, this is equivalent to max gain.
        # BUT, the inspiration mentions "penalizes redundancy". 
        # A more subtle ratio might be gain / (number of points in block already covered?) 
        # No, standard greedy set cover uses max gain.
        # Let's stick to max gain but with randomized tie-breaking or noise.
        
        selected_blocks = []
        
        while uncovered.sum() > 0:
            if time.time() > deadline - 10:
                # Fallback: just pick random blocks to finish
                while uncovered.sum() > 0:
                    b_idx = rng.randint(0, n_all_blocks - 1)
                    selected_blocks.append(b_idx)
                    for t_idx in block_triplets[b_idx]:
                        if uncovered[t_idx] > 0:
                            uncovered[t_idx] = 0
                            for other_b_idx in triple_blocks[t_idx]:
                                gain[other_b_idx] -= 1
                break

            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                # Should not happen if uncovered > 0 and we have valid blocks
                break
            
            # Find indices with max gain
            candidates = np.where(gain == max_gain)[0]
            
            if bias == 0.0 or len(candidates) == 1:
                b_idx = candidates[0] if len(candidates) == 1 else rng.choice(candidates.tolist())
            else:
                # With bias, we might pick from top K or add noise
                # For simplicity, just random pick from top candidates
                b_idx = rng.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)
            
            # Update gain and uncovered
            for t_idx in block_triplets[b_idx]:
                if uncovered[t_idx] > 0:
                    uncovered[t_idx] -= 1
                    # For every block that covers this triplet, decrease its gain
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1
                        
        return [all_k_blocks[b_idx] for b_idx in selected_blocks]

    def greedy_construct_pointwise(v, k, t, seed=0, bias=0.0):
        """
        Original pointwise greedy construction.
        """
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        while uncovered:
            # Pick a starting triplet
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                best_point = -1
                max_gain = -1
                second_best_gain = -1
                second_best_point = -1
                
                # Pairs in current block (for t=3)
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                gains = {}
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    gains[p] = gain
                    if gain > max_gain:
                        second_best_gain = max_gain
                        second_best_point = best_point
                        max_gain = gain
                        best_point = p
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = p

                if max_gain == 0:
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    r = rng.random()
                    if r < (1.0 - bias):
                        pass
                    elif second_best_point != -1 and r < (1.0 - bias) + bias * 0.5:
                        best_point = second_best_point
                    else:
                        candidates = [p for p in range(v) if p not in block_set]
                        best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
        
        return blocks

    # 1. Initial Construction
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Precomputed Greedy with different seeds
    # This is likely faster and finds good covers.
    for seed in range(20):
        if time.time() > deadline - 60:
            break
        blocks = greedy_construct_ratio(v, k, t, seed, bias=0.0)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    # Phase 2: Pointwise Greedy with different seeds
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        blocks = greedy_construct_pointwise(v, k, t, seed, bias=0.0)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    # Phase 3: Randomized variants
    for seed in range(5):
        if time.time() > deadline - 15:
            break
        blocks = greedy_construct_ratio(v, k, t, seed, bias=0.3)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct_ratio(v, k, t, 0, bias=0.0)

    # 2. Local Search Optimization
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks, block_weights=None):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        # Precompute triplet sets for all blocks
        block_triplet_ids = []
        for b in blocks:
            ids = set()
            for triplet in itertools.combinations(b, t):
                ids.add(get_id(triplet))
            block_triplet_ids.append(ids)
        
        # If block_weights provided, use them to prioritize swaps
        if block_weights is not None:
            pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
            pair_indices.sort(key=lambda ij: block_weights[ij[0]] + block_weights[ij[1]])
        else:
            pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for i, j in pair_indices:
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = block_triplet_ids[i]
            b2 = blocks[j]
            ids_b2 = block_triplet_ids[j]
            
            critical_points = set()
            
            for tid in ids_b1:
                c = count[tid]
                covered_by_b1 = 1
                covered_by_b2 = 1 if tid in ids_b2 else 0
                if c == covered_by_b1 + covered_by_b2:
                    critical_points.update(all_triplets[tid])
            
            for tid in ids_b2:
                if tid in ids_b1:
                    continue
                c = count[tid]
                covered_by_b1 = 0
                covered_by_b2 = 1
                if c == covered_by_b1 + covered_by_b2:
                    critical_points.update(all_triplets[tid])
            
            if not critical_points:
                update_counts([b1, b2], -1)
                return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
            
            if len(critical_points) <= k:
                new_block = list(critical_points)
                while len(new_block) < k:
                    p = random.randint(0, v - 1)
                    if p not in new_block:
                        new_block.append(p)
                new_block.sort()
                
                update_counts([b1, b2], -1)
                update_counts([new_block], 1)
                
                new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                new_blocks.append(new_block)
                return new_blocks, True

        return blocks, False

    # Main Optimization Loop
    progress_made = True
    while progress_made and time.time() < deadline - 2:
        progress_made = False
        
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True
            
        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True

    # Final pass to ensure no redundants remain
    current_blocks, _ = try_remove_redundant(current_blocks)
    
    # Phase 4: Continuous Relaxation with Projected Gradient Descent
    # Relax binary block selection variables to [0,1] and minimize:
    # f(x) = sum(x_i) + lambda * sum_j max(0, 1 - sum_{i: j in B_i} x_i)^2
    # with L1 regularization to encourage sparsity
    
    if time.time() < deadline - 5:
        n_blocks = len(current_blocks)
        
        # Build incidence matrix A: A[j, i] = 1 if triplet j is in block i
        # Shape: (n_triplets, n_blocks)
        A = np.zeros((n_triplets, n_blocks), dtype=np.float64)
        for i, block in enumerate(current_blocks):
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                A[tid, i] = 1.0
        
        # Initialize x to 1 for all blocks (all selected)
        x = np.ones(n_blocks, dtype=np.float64)
        
        # Parameters for gradient descent
        lambda_cov = 100.0  # Coverage penalty weight
        lambda_l1 = 0.1     # L1 regularization weight
        lr = 0.05           # Learning rate
        n_iters = 200       # Number of iterations
        
        # Projected gradient descent
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute coverage: cov[j] = sum_i A[j,i] * x[i]
            cov = A @ x
            
            # Gradient of coverage penalty: lambda * sum_j 2 * max(0, 1 - cov[j]) * (-A[j,:])
            # = -2 * lambda * A.T @ max(0, 1 - cov)
            deficit = np.maximum(0.0, 1.0 - cov)
            grad_cov = -2.0 * lambda_cov * (A.T @ deficit)
            
            # Gradient of sum(x_i): ones
            grad_sum = np.ones(n_blocks)
            
            # Gradient of L1: sign(x) (subgradient)
            grad_l1 = lambda_l1 * np.sign(x)
            
            # Total gradient
            grad = grad_sum + grad_cov + grad_l1
            
            # Gradient step
            x = x - lr * grad
            
            # Project to [0, 1]
            x = np.clip(x, 0.0, 1.0)
        
        # Use continuous weights to guide local search
        # Blocks with higher x values are more "important"
        block_weights = x.copy()
        
        # Try swaps prioritized by block weights
        if time.time() < deadline - 3:
            progress = True
            while progress and time.time() < deadline - 3:
                progress = False
                current_blocks, swapped = try_swap_2_for_1(current_blocks, block_weights)
                if swapped:
                    progress = True
                    # Recompute weights after swap
                    n_blocks_new = len(current_blocks)
                    if n_blocks_new != n_blocks:
                        # Rebuild incidence and re-run gradient descent
                        A = np.zeros((n_triplets, n_blocks_new), dtype=np.float64)
                        for i, block in enumerate(current_blocks):
                            for triplet in itertools.combinations(block, t):
                                tid = get_id(triplet)
                                A[tid, i] = 1.0
                        x = np.ones(n_blocks_new, dtype=np.float64)
                        for it in range(50):
                            if time.time() > deadline - 3:
                                break
                            cov = A @ x
                            deficit = np.maximum(0.0, 1.0 - cov)
                            grad_cov = -2.0 * lambda_cov * (A.T @ deficit)
                            grad_sum = np.ones(n_blocks_new)
                            grad_l1 = lambda_l1 * np.sign(x)
                            grad = grad_sum + grad_cov + grad_l1
                            x = x - lr * grad
                            x = np.clip(x, 0.0, 1.0)
                        block_weights = x.copy()
                    else:
                        # Same number of blocks, just update weights
                        block_weights = block_weights[:n_blocks_new]
        
        # Final removal pass
        current_blocks, _ = try_remove_redundant(current_blocks)
    
    return current_blocks

# Test/Entry
if __name__ == "__main__":
    # Test with small parameters to ensure no crash
    # blocks = construct(8, 4, 2)
    # print(len(blocks))
    
    # Run for the actual task parameters if needed by evaluator
    # The evaluator calls construct(24, 7, 3)
    pass
