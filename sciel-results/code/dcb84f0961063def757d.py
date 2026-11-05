import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin: evaluator kills at 150s
    
    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Build coverage matrix: n_k_blocks x n_t_subs, entry is 1 if block covers t-sub
    # Use uint8 to save memory
    coverage_matrix = np.zeros((n_k_blocks, n_t_subs), dtype=np.uint8)
    for i, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            coverage_matrix[i, idx] = 1
    
    # 3. Greedy construction with NumPy vectorization
    def greedy_construct(seed_order=None, use_randomization=False):
        """Greedy set cover using NumPy for speed."""
        if time.time() > deadline - 5:
            return None
        
        # Build ordered list of block indices
        if seed_order is not None:
            block_order = list(seed_order)
        else:
            block_order = list(range(n_k_blocks))
            if use_randomization:
                random.shuffle(block_order)
        
        uncovered = np.ones(n_t_subs, dtype=np.uint8)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline - 5:
                break
            
            # Compute gain for each block: number of currently uncovered t-subs it covers
            gains = (coverage_matrix[uncovered_indices] @ uncovered).astype(np.int32) if False else None
            
            # Fast: for each candidate block, count how many uncovered t-subs it covers
            # gains[b] = sum(coverage_matrix[b] & uncovered)
            # Using matrix-vector product: coverage_matrix @ uncovered gives gains for all blocks
            gains = (coverage_matrix @ uncovered).astype(np.int32)
            
            # Only consider blocks not yet selected
            selected_set = set(selected)
            best_gain = 0
            best_idx = -1
            best_candidates = []
            
            for bi in block_order:
                if bi in selected_set:
                    continue
                g = gains[bi]
                if g > best_gain:
                    best_gain = g
                    best_idx = bi
                    best_candidates = [bi]
                elif g == best_gain and g > 0:
                    best_candidates.append(bi)
            
            if best_idx == -1 or best_gain == 0:
                break
            
            if use_randomization and len(best_candidates) > 1:
                best_idx = random.choice(best_candidates)
            
            selected.append(best_idx)
            uncovered = (uncovered & ~coverage_matrix[best_idx]).astype(np.uint8)
        
        return selected
    
    # 4. Redundancy removal (vectorized)
    def remove_redundant(selected, deadline_margin=10):
        """Remove blocks that don't contribute to covering any t-subset uniquely."""
        if not selected:
            return selected
        
        current = list(selected)
        
        # Build cover count
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for bi in current:
            cover_count += coverage_matrix[bi]
        
        improved = True
        while improved and time.time() < deadline - deadline_margin:
            improved = False
            new_current = []
            for i in range(len(current)):
                bi = current[i]
                # Check if removing bi leaves any t-sub with count 0
                if (cover_count[coverage_matrix[bi] == 1] == 1).any():
                    # Critical: at least one t-sub is only covered by bi
                    new_current.append(bi)
                else:
                    # Can remove
                    cover_count -= coverage_matrix[bi]
                    improved = True
            current = new_current
        
        return current
    
    # 5. Local search: swap to improve
    def local_search(selected, max_iters=50, deadline_margin=10):
        """Try to swap blocks to reduce count or enable further pruning."""
        if not selected:
            return selected
        
        current = list(selected)
        current_set = set(current)
        
        # Build cover count
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for bi in current:
            cover_count += coverage_matrix[bi]
        
        improved = True
        while improved and time.time() < deadline - deadline_margin:
            improved = False
            n = len(current)
            
            for i in range(n):
                if time.time() > deadline - deadline_margin:
                    break
                bi = current[i]
                
                # Try removing bi
                cover_count -= coverage_matrix[bi]
                uncovered = (cover_count == 0).astype(np.uint8)
                
                if not uncovered.any():
                    # Can remove bi entirely
                    current.pop(i)
                    current_set.discard(bi)
                    improved = True
                    break
                else:
                    # Try to find a replacement block that covers all uncovered
                    n_uncovered = uncovered.sum()
                    best_gain = 0
                    best_block = -1
                    
                    # Scan all blocks not in current
                    for bi2 in range(n_k_blocks):
                        if bi2 in current_set:
                            continue
                        gain = int((coverage_matrix[bi2] & uncovered).sum())
                        if gain > best_gain:
                            best_gain = gain
                            best_block = bi2
                        if best_gain >= n_uncovered:
                            break
                    
                    if best_block != -1 and best_gain == n_uncovered:
                        # Replace bi with best_block
                        current[i] = best_block
                        current_set.discard(bi)
                        current_set.add(best_block)
                        cover_count += coverage_matrix[best_block]
                        improved = True
                        break
                    else:
                        # Revert
                        cover_count += coverage_matrix[bi]
        
        return current
    
    # 6. Multiple restarts
    best_blocks = None
    best_count = float('inf')
    
    # Pre-generate some random orderings for diversity
    np.random.seed(42)
    
    max_restarts = 12
    time_per_restart = (deadline - time.time() - 20) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        t_start = time.time()
        
        # Create a random ordering for greedy
        order = list(range(n_k_blocks))
        random.shuffle(order)
        
        # Greedy with this ordering
        selected = greedy_construct(seed_order=order, use_randomization=True)
        
        if selected is None:
            continue
        
        # Verify coverage
        cov = np.zeros(n_t_subs, dtype=np.uint8)
        for bi in selected:
            cov |= coverage_matrix[bi]
        if not cov.all():
            continue
        
        # Local search
        selected = remove_redundant(selected, deadline_margin=15)
        selected = local_search(selected, max_iters=30, deadline_margin=15)
        
        # Final redundancy check
        selected = remove_redundant(selected, deadline_margin=10)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = list(selected)
        
        # If we've hit the known best, try harder local search
        if best_count <= 151:
            # Try more aggressive local search
            selected = local_search(selected, max_iters=100, deadline_margin=10)
            selected = remove_redundant(selected, deadline_margin=5)
            if len(selected) < best_count:
                best_count = len(selected)
                best_blocks = list(selected)
        
        # Check time
        if time.time() - t_start > time_per_restart * 1.5:
            # Slow restart, reduce remaining
            max_restarts = restart + 3
    
    # Fallback: simple greedy if nothing found
    if best_blocks is None:
        selected = greedy_construct(use_randomization=False)
        if selected is None:
            # Very basic fallback
            uncovered = set(all_t_subs)
            selected = []
            for bi, block in enumerate(all_k_blocks):
                if not uncovered:
                    break
                if time.time() > deadline - 5:
                    break
                block_t_subs = set(itertools.combinations(block, t))
                if uncovered & block_t_subs:
                    selected.append(bi)
                    uncovered -= block_t_subs
        best_blocks = selected
    
    return [list(all_k_blocks[i]) for i in best_blocks]
