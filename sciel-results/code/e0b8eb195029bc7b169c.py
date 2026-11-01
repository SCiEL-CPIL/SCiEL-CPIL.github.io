import time
import random
import itertools
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Precompute t-subsets and their indices
    tsubs = list(itertools.combinations(range(v), t))
    n_tsub = len(tsubs)
    tsub_to_bit = {sub: i for i, sub in enumerate(tsubs)}
    
    # Precompute all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    # For each block, store the list of t-subset indices it covers
    block_tsub_indices = []
    for comb in all_k_subsets:
        indices = []
        for sub in itertools.combinations(comb, t):
            idx = tsub_to_bit[sub]
            indices.append(idx)
        block_tsub_indices.append(indices)
    
    # Build incidence matrix: M[block][tsub] = 1 if block covers tsub
    # Using numpy for efficient operations
    M = np.zeros((n_ksub, n_tsub), dtype=np.uint8)
    for b_idx, indices in enumerate(block_tsub_indices):
        for idx in indices:
            M[b_idx, idx] = 1
    
    # Greedy construction
    def greedy_construct():
        covered = np.zeros(n_tsub, dtype=np.uint8)
        blocks = []
        remaining = np.ones(n_tsub, dtype=bool)
        
        while remaining.any():
            if time.time() > deadline:
                break
            # Find block with max coverage of remaining
            gains = M @ remaining.astype(np.uint8)
            # Only consider blocks not already selected
            for b in blocks:
                gains[b] = 0
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            blocks.append(best_idx)
            remaining &= ~M[best_idx].astype(bool)
        
        return blocks
    
    # Reduce redundant blocks
    def reduce_blocks(blocks):
        if not blocks:
            return blocks
        # Count coverage
        cover_count = np.zeros(n_tsub, dtype=np.int32)
        for b in blocks:
            for idx in block_tsub_indices[b]:
                cover_count[idx] += 1
        
        new_blocks = []
        for b in blocks:
            # Check if any t-subset in this block is covered only by this block
            redundant = True
            for idx in block_tsub_indices[b]:
                if cover_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_tsub_indices[b]:
                    cover_count[idx] -= 1
            else:
                new_blocks.append(b)
        
        return new_blocks
    
    # 2-for-1 swap: try replacing two blocks with one
    def try_2for1_swap(blocks):
        """Try to find two blocks that can be replaced by a single block."""
        n = len(blocks)
        if n < 2:
            return blocks
        
        # For efficiency, only try a subset of pairs
        # Build coverage sets for each block
        block_sets = [set(block_tsub_indices[b]) for b in blocks]
        
        # Count how many blocks cover each t-subset
        tsub_count = {}
        for i, bs in enumerate(block_sets):
            for tsub in bs:
                tsub_count[tsub] = tsub_count.get(tsub, 0) + 1
        
        # Try random pairs
        indices = list(range(n))
        random.shuffle(indices)
        tried = 0
        max_tries = min(200, n * (n - 1) // 2)
        
        for i in range(len(indices)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(indices)):
                if tried >= max_tries:
                    break
                tried += 1
                bi = indices[i]
                bj = indices[j]
                
                # Union of t-subsets covered by blocks bi and bj
                union = block_sets[bi] | block_sets[bj]
                
                # Check if there exists a block that covers all of union
                # and doesn't require removing coverage of other t-subsets
                # Actually, we need a block that covers all t-subsets in union
                # that are currently covered ONLY by bi or bj
                critical = set()
                for tsub in union:
                    if tsub_count[tsub] == 1:
                        critical.add(tsub)
                
                if not critical:
                    continue
                
                # Find a block that covers all critical t-subsets
                # For efficiency, check if any block covers all of critical
                # This is expensive, so we limit
                found = False
                for b in range(n_ksub):
                    if b == blocks[bi] or b == blocks[bj]:
                        continue
                    # Check if block b covers all critical t-subsets
                    covers_all = True
                    for tsub in critical:
                        if M[b, tsub] == 0:
                            covers_all = False
                            break
                    if covers_all:
                        # Replace bi and bj with b
                        new_blocks = [blocks[x] for x in range(n) if x != bi and x != bj]
                        new_blocks.append(b)
                        return new_blocks
                        found = True
                        break
                if found:
                    break
        
        return blocks
    
    # Local search: destroy and repair
    def local_search(current_blocks, max_iters=50):
        best = list(current_blocks)
        best_size = len(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Perturb: remove 1-3 random blocks
            n_remove = random.randint(1, 3)
            if len(best) <= n_remove:
                break
            
            items = list(range(len(best)))
            random.shuffle(items)
            removed_indices = set(items[:n_remove])
            
            remaining = [best[i] for i in range(len(best)) if i not in removed_indices]
            
            # Calculate uncovered
            covered = np.zeros(n_tsub, dtype=np.uint8)
            for b in remaining:
                covered |= M[b]
            uncovered = ~covered.astype(bool)
            
            if not uncovered.any():
                continue
            
            # Repair greedily
            new_blocks = list(remaining)
            while uncovered.any():
                if time.time() > deadline:
                    break
                gains = M @ uncovered.astype(np.uint8)
                for b in new_blocks:
                    gains[b] = 0
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                new_blocks.append(best_idx)
                uncovered &= ~M[best_idx].astype(bool)
            
            if uncovered.any():
                continue
            
            # Reduce
            new_blocks = reduce_blocks(new_blocks)
            
            # Try 2-for-1 swap
            new_blocks = try_2for1_swap(new_blocks)
            
            if len(new_blocks) < best_size:
                best = new_blocks
                best_size = len(best)
        
        return best
    
    # Main algorithm
    # Initial greedy construction
    current_blocks = greedy_construct()
    current_blocks = reduce_blocks(current_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # Local search iterations
    while time.time() < deadline:
        # Run local search from current
        current_blocks = local_search(current_blocks, max_iters=20)
        
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = list(current_blocks)
        
        # Try 2-for-1 on best
        best_blocks = try_2for1_swap(best_blocks)
        best_blocks = reduce_blocks(best_blocks)
        best_size = len(best_blocks)
        
        # Perturb best to escape local minima
        if len(best_blocks) > 5:
            n_perturb = random.randint(1, 2)
            items = list(range(len(best_blocks)))
            random.shuffle(items)
            removed = set(items[:n_perturb])
            perturbed = [best_blocks[i] for i in range(len(best_blocks)) if i not in removed]
            
            # Repair
            covered = np.zeros(n_tsub, dtype=np.uint8)
            for b in perturbed:
                covered |= M[b]
            uncovered = ~covered.astype(bool)
            
            new_blocks = list(perturbed)
            while uncovered.any() and time.time() < deadline:
                gains = M @ uncovered.astype(np.uint8)
                for b in new_blocks:
                    gains[b] = 0
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                new_blocks.append(best_idx)
                uncovered &= ~M[best_idx].astype(bool)
            
            if not uncovered.any():
                new_blocks = reduce_blocks(new_blocks)
                new_blocks = try_2for1_swap(new_blocks)
                if len(new_blocks) < best_size:
                    best_size = len(new_blocks)
                    best_blocks = new_blocks
                current_blocks = new_blocks
            else:
                current_blocks = best_blocks
    
    # Final output
    result = [list(all_k_subsets[idx]) for idx in best_blocks]
    return result
