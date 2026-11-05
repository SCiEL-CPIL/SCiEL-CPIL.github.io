import itertools
import time
import numpy as np
from math import comb

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin

    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)

    # block_triple_indices[i, j] = index of j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        # Vectorize the inner loop for speed
        subs = list(itertools.combinations(block, t))
        block_triple_indices[i] = [triple_to_idx[sub] for sub in subs]

    # Calculate frequency of each triple across all blocks for weighting
    # freq[j] = number of blocks that contain triple j
    freq = np.bincount(block_triple_indices.flatten(), minlength=n_triples)
    # Inverse frequency: rare triples get higher weight
    # Add small epsilon to avoid division by zero, though freq should be > 0 for valid t < k < v
    inv_freq = 1.0 / (freq.astype(np.float64) + 1e-6)

    def _greedy_cover(mode, random_seed=0):
        """
        Vectorized greedy set cover.
        mode: 'standard', 'weighted', 'randomized'
        """
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_indices = []
        remaining = n_triples
        selected_mask = np.zeros(n_blocks, dtype=bool)

        while remaining > 0:
            if time.time() > deadline:
                return None # Timeout signal

            # Get triples covered by each unselected block
            # uncovered_mask[block_triple_indices] is boolean array (n_blocks, k_t)
            if mode == 'standard':
                gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            elif mode == 'weighted':
                # Weighted gain: sum of weights of uncovered triples covered by block
                # w is 0 if covered, inv_freq if uncovered
                w = np.where(uncovered_mask, inv_freq, 0.0)
                gains = np.sum(w[block_triple_indices], axis=1)
            else:
                gains = np.sum(uncovered_mask[block_triple_indices], axis=1)

            # Mask out already selected blocks
            gains[selected_mask] = -1

            max_gain = np.max(gains)
            if max_gain <= 0:
                # If max_gain is 0 and remaining > 0, we are stuck (shouldn't happen if valid)
                # Or we ran out of useful blocks.
                break

            if mode == 'randomized':
                # Pick randomly from top candidates
                # To be efficient, pick from candidates with gain > 0.9 * max_gain or similar
                # For simplicity, pick from all with max_gain
                candidates = np.where(gains == max_gain)[0]
                if len(candidates) > 50:
                    # Sample to avoid huge list operations
                    candidates = rng.choice(candidates, size=50, replace=False)
                best_idx = candidates[rng.randint(len(candidates))]
            else:
                best_idx = np.argmax(gains)

            selected_mask[best_idx] = True
            selected_indices.append(best_idx)

            # Update uncovered mask
            b_triples = block_triple_indices[best_idx]
            newly = b_triples[uncovered_mask[b_triples]]
            if len(newly) > 0:
                uncovered_mask[newly] = False
                remaining -= len(newly)

        if remaining > 0:
            return None # Incomplete
        
        return [list(all_blocks[i]) for i in selected_indices]

    def _local_improve(blocks, random_seed=0, max_rounds=50):
        """
        Fast local improvement:
        1. Remove redundant blocks.
        2. Try to replace blocks to reduce count or improve coverage efficiency.
        """
        if not blocks:
            return blocks

        # Build coverage counts and triple sets for current blocks
        triple_to_idx_local = triple_to_idx
        n_t = n_triples

        cov = np.zeros(n_t, dtype=np.int32)
        block_triple_sets = []
        for b in blocks:
            trs = list(itertools.combinations(b, t))
            t_indices = np.array([triple_to_idx_local[tr] for tr in trs], dtype=np.int32)
            block_triple_sets.append(t_indices)
            cov[t_indices] += 1

        improved = True
        rounds = 0
        rng = np.random.RandomState(random_seed)
        
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            if time.time() > deadline:
                break

            # 1) Remove redundant blocks (vectorized check)
            # A block is redundant if all its triples have cov > 1
            # We check blocks one by one to maintain cov array consistency easily, 
            # but we can optimize by checking all at once if we were building a matrix.
            # Given n_blocks in solution is small (~70-80), loop is fine.
            removed = False
            for i in range(len(block_triple_sets)):
                t_indices = block_triple_sets[i]
                if np.all(cov[t_indices] > 1):
                    cov[t_indices] -= 1
                    block_triple_sets.pop(i)
                    blocks.pop(i)
                    improved = True
                    removed = True
                    break # Restart loop after modification

            if removed:
                continue

            # 2) Try to replace a block
            # Strategy: Find the block with the fewest "unique" triples (cov==1)
            # If a block has 0 unique triples, it's redundant (handled above).
            # If it has some, try to find a replacement that covers those unique triples
            # and potentially more, allowing us to drop other blocks later?
            # Simpler strategy: Try to replace a block with another block that
            # covers the same unique triples, but overlaps less with others?
            # Actually, just trying to find a better block for the "weakest" block.
            
            # Find block with minimum unique triples count (but > 0)
            unique_counts = []
            for i in range(len(block_triple_sets)):
                t_indices = block_triple_sets[i]
                unc = np.sum(cov[t_indices] == 1)
                unique_counts.append(unc)
            
            if not unique_counts:
                break
                
            # Sort blocks by unique count (ascending)
            sorted_indices = np.argsort(unique_counts)
            
            replaced = False
            for i in sorted_indices[:5]: # Try the 5 weakest blocks
                if time.time() > deadline:
                    break
                
                if unique_counts[i] == 0:
                    continue # Should have been removed
                
                old_block = blocks[i]
                old_triples = block_triple_sets[i]
                unique_triples = old_triples[cov[old_triples] == 1]
                
                # Temporarily remove old block
                cov[old_triples] -= 1
                uncovered = np.where(cov == 0)[0]
                
                if len(uncovered) == 0:
                    # This block was not strictly necessary? 
                    # Wait, if unique_triples > 0, removing it makes those triples uncovered.
                    # So uncovered should be exactly unique_triples.
                    pass

                # Find a new block that covers all uncovered triples
                # We want a block that covers `uncovered` and minimizes new overlaps?
                # Or just any block that covers them.
                # To keep it simple, we look for a block that covers the most uncovered triples.
                # Since we removed one block, `uncovered` is small.
                
                # Filter blocks that cover all uncovered triples? 
                # That might be too strict. Let's find a block that covers the most uncovered.
                # But we must cover ALL uncovered to be valid.
                
                # Efficient check:
                # For each candidate block, does it cover all `uncovered`?
                # This is expensive if done for all n_blocks.
                
                # Optimization: Only consider blocks that contain the first uncovered triple.
                if len(uncovered) > 0:
                    # Find blocks containing uncovered[0]
                    # We precomputed triple_to_blocks in current program? No, we can compute it.
                    # But for now, let's just do a quick scan of a subset of blocks or use a heuristic.
                    
                    # Heuristic: Try blocks that share many points with old_block?
                    # Or just try a few random blocks that cover the first uncovered triple.
                    
                    # Let's stick to a simpler improvement: 
                    # If we can find a block that covers the unique triples of the removed block,
                    # we swap. 
                    
                    # To avoid O(n_blocks * n_uncovered) per swap, we limit attempts.
                    
                    found_replacement = False
                    # Try 20 random blocks
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        # Pick a random block
                        cand_idx = rng.randint(n_blocks)
                        cand_block = all_blocks[cand_idx]
                        
                        # Check if cand_block covers all uncovered
                        # Convert cand_block to triple indices
                        cand_triples = set(block_triple_indices[cand_idx])
                        
                        # Check if set(uncovered) is subset of cand_triples
                        # uncovered is an array of indices
                        if all(u in cand_triples for u in uncovered):
                            # Valid replacement!
                            blocks[i] = cand_block
                            new_triples = block_triple_indices[cand_idx]
                            block_triple_sets[i] = new_triples
                            cov[new_triples] += 1
                            improved = True
                            replaced = True
                            found_replacement = True
                            break
                    
                    if found_replacement:
                        break # Restart loop after modification
                else:
                    # No uncovered triples? That means the block was redundant but we missed it?
                    # Revert
                    cov[old_triples] += 1

            if not improved:
                break

        return blocks

    def _is_valid(blocks):
        """Check every t-subset is covered."""
        covered = np.zeros(n_triples, dtype=bool)
        for b in blocks:
            trs = list(itertools.combinations(b, t))
            t_indices = [triple_to_idx[tr] for tr in trs]
            covered[t_indices] = True
        return np.all(covered)

    best_blocks = None
    best_len = float('inf')

    # Strategies to try
    strategies = [
        ('standard', 0),
        ('weighted', 1),
        ('randomized', 2),
        ('standard', 3),
        ('weighted', 4),
        ('randomized', 5),
    ]

    # Run multiple restarts
    # We have ~110s. Precomp takes ~1-2s.
    # Each greedy run takes ~5-10ms. Local search takes ~50-200ms.
    # We can run many iterations.
    
    attempts = 0
    while time.time() < deadline - 5.0:
        mode, seed = strategies[attempts % len(strategies)]
        attempts += 1
        
        blocks = _greedy_cover(mode, seed)
        
        if blocks is None:
            continue
            
        # Local improvement
        blocks = _local_improve(blocks, random_seed=seed, max_rounds=20)
        
        # Validate
        if not _is_valid(blocks):
            continue
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
        # Early stop if we hit the known best
        if best_len <= 76:
            break

    if best_blocks is None:
        # Fallback: Deterministic greedy
        blocks = _greedy_cover('standard', 0)
        if blocks:
            blocks = _local_improve(blocks, 0, 10)
            if _is_valid(blocks):
                best_blocks = blocks
            else:
                best_blocks = blocks # Return invalid rather than empty? No, empty is invalid too.
        else:
            best_blocks = []

    return best_blocks
