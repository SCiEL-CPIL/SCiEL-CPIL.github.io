"""Covering design C(24,7,3): minimize number of 7-subsets of 24 points
so that every 3-subset is contained in some block.
Entry point: construct(v, k, t) -> list of blocks (sorted k-subsets)."""

import itertools
import time
import numpy as np
from math import comb, log, exp
import random


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # safety margin before 150s kill

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
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]

    # triple_to_blocks: for each triple, which blocks contain it
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for j in range(k_t):
            t_idx = block_triple_indices[i, j]
            triple_to_blocks[t_idx].append(i)

    def _weighted_greedy(random_seed, weight_decay=0.9):
        """Weighted greedy: use Chvátal-McGill weights to break ties."""
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        selected_mask = np.zeros(n_blocks, dtype=bool)
        weights = np.ones(n_triples, dtype=np.float64)

        while remaining > 0:
            if time.time() > deadline:
                break

            # Compute weighted gains for all unselected blocks
            gains = np.sum(weights[block_triple_indices] * uncovered_mask[block_triple_indices], axis=1)
            gains[selected_mask] = -1

            max_gain = np.max(gains)
            if max_gain <= 0:
                break

            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]

            selected_mask[best_idx] = True
            selected_blocks.append(list(all_blocks[best_idx]))

            # Update: mark newly covered triples and decay their weights
            b_triples = block_triple_indices[best_idx]
            newly = b_triples[uncovered_mask[b_triples]]
            if len(newly) > 0:
                uncovered_mask[newly] = False
                weights[newly] *= weight_decay
                remaining -= len(newly)

        return selected_blocks

    def _simple_greedy(random_seed):
        """Standard greedy without weights."""
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        selected_mask = np.zeros(n_blocks, dtype=bool)

        while remaining > 0:
            if time.time() > deadline:
                break

            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            gains[selected_mask] = -1

            max_gain = np.max(gains)
            if max_gain <= 0:
                break

            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]

            selected_mask[best_idx] = True
            selected_blocks.append(list(all_blocks[best_idx]))

            b_triples = block_triple_indices[best_idx]
            newly = b_triples[uncovered_mask[b_triples]]
            if len(newly) > 0:
                uncovered_mask[newly] = False
                remaining -= len(newly)

        return selected_blocks

    def _build_coverage(blocks):
        """Build coverage data structure for a list of blocks."""
        cov = np.zeros(n_triples, dtype=np.int32)
        block_triple_sets = []
        for b in blocks:
            trs = list(itertools.combinations(b, t))
            t_indices = np.array([triple_to_idx[tr] for tr in trs], dtype=np.int32)
            block_triple_sets.append(t_indices)
            cov[t_indices] += 1
        return cov, block_triple_sets

    def _remove_redundant(blocks, cov, block_triple_sets, max_rounds=100):
        """Remove any block whose removal doesn't uncover any triple."""
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            if time.time() > deadline:
                break
            for i in range(len(block_triple_sets)):
                t_indices = block_triple_sets[i]
                if np.all(cov[t_indices] > 1):
                    cov[t_indices] -= 1
                    block_triple_sets.pop(i)
                    blocks.pop(i)
                    improved = True
                    break
        return blocks, cov, block_triple_sets

    def _try_replace_block(blocks, cov, block_triple_sets, block_idx, random_seed=0):
        """Try to replace a block with a better one that covers more uncovered triples."""
        rng = np.random.RandomState(random_seed)
        old_block = blocks[block_idx]
        old_triples = block_triple_sets[block_idx]
        
        # Temporarily remove the block
        cov[old_triples] -= 1
        uncovered = np.where(cov == 0)[0]
        
        if len(uncovered) == 0:
            # Block was redundant, just remove it
            blocks.pop(block_idx)
            block_triple_sets.pop(block_idx)
            return True, blocks, cov, block_triple_sets
        
        # Try to build a replacement block
        # Start with the triple that is most uncovered
        seed_triple = all_triples[uncovered[0]]
        new_block = list(seed_triple)
        in_b = set(new_block)
        
        # Greedily add points
        while len(new_block) < k:
            best_x = None
            best_cnt = -1
            for x in range(v):
                if x in in_b:
                    continue
                # Count how many uncovered triples this x would help cover
                cnt = 0
                for rest in itertools.combinations(new_block, t - 1):
                    tr = tuple(sorted(rest + (x,)))
                    if tr in triple_to_idx:
                        idx = triple_to_idx[tr]
                        if cov[idx] == 0:
                            cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_x = x
            if best_x is None:
                best_x = next(p for p in range(v) if p not in in_b)
            new_block.append(best_x)
            in_b.add(best_x)
        
        new_block.sort()
        new_triples = np.array(
            [triple_to_idx[tr] for tr in itertools.combinations(new_block, t)],
            dtype=np.int32
        )
        
        # Check if this replacement is beneficial
        covered_new = np.sum(cov[new_triples] == 0)
        if covered_new >= len(uncovered):
            blocks[block_idx] = new_block
            block_triple_sets[block_idx] = new_triples
            cov[new_triples] += 1
            return True, blocks, cov, block_triple_sets
        else:
            # Revert
            cov[old_triples] += 1
            return False, blocks, cov, block_triple_sets

    def _simulated_annealing(blocks, cov, block_triple_sets, random_seed=0, 
                               max_iters=5000, T_start=1.0, T_end=0.01, alpha=0.999):
        """Simulated annealing to improve the covering."""
        rng = np.random.RandomState(random_seed)
        current_blocks = blocks[:]
        current_cov = cov.copy()
        current_bts = block_triple_sets[:]
        current_len = len(current_blocks)
        
        best_blocks = current_blocks[:]
        best_len = current_len
        
        T = T_start
        iters = 0
        
        while iters < max_iters:
            iters += 1
            if time.time() > deadline:
                break
            
            T *= alpha
            if T < T_end:
                break
            
            # Choose a move: swap, replace, or remove
            move_type = rng.randint(3)
            
            if move_type == 0 and len(current_blocks) > 1:
                # Swap two blocks
                i, j = rng.choice(len(current_blocks), size=2, replace=False)
                if i == j:
                    continue
                # Try swapping
                old_i = current_blocks[i]
                old_j = current_blocks[j]
                
                # Remove both
                current_cov[current_bts[i]] -= 1
                current_cov[current_bts[j]] -= 1
                
                # Find new blocks
                # For simplicity, try random blocks that cover uncovered triples
                uncovered = np.where(current_cov == 0)[0]
                if len(uncovered) == 0:
                    # Both were redundant
                    new_i = None
                    new_j = None
                else:
                    # Pick random blocks from triple_to_blocks
                    new_i = None
                    new_j = None
                    for _ in range(10):
                        tr_idx = uncovered[rng.randint(len(uncovered))]
                        candidates = triple_to_blocks[tr_idx]
                        if candidates:
                            new_i = rng.choice(candidates)
                            break
                    if new_i is not None and new_i not in [i, j]:
                        current_blocks[i] = list(all_blocks[new_i])
                        current_bts[i] = block_triple_indices[new_i]
                        current_cov[current_bts[i]] += 1
                    
                    uncovered = np.where(current_cov == 0)[0]
                    for _ in range(10):
                        if len(uncovered) == 0:
                            break
                        tr_idx = uncovered[rng.randint(len(uncovered))]
                        candidates = triple_to_blocks[tr_idx]
                        if candidates:
                            new_j = rng.choice(candidates)
                            break
                    if new_j is not None and new_j not in [i, j] and new_j != new_i:
                        current_blocks[j] = list(all_blocks[new_j])
                        current_bts[j] = block_triple_indices[new_j]
                        current_cov[current_bts[j]] += 1
                    else:
                        # Revert
                        current_blocks[i] = old_i
                        current_bts[i] = block_triple_indices[old_i]
                        current_cov[current_bts[i]] += 1
                        continue
                
                # Check validity
                if np.any(current_cov == 0):
                    # Revert
                    current_blocks[i] = old_i
                    current_blocks[j] = old_j
                    current_bts[i] = block_triple_indices[old_i]
                    current_bts[j] = block_triple_indices[old_j]
                    current_cov[current_bts[i]] += 1
                    current_cov[current_bts[j]] += 1
                    continue
                
                # Accept or reject based on energy change
                new_len = len(current_blocks)
                delta = new_len - current_len
                if delta <= 0 or rng.random() < exp(-delta / max(T, 1e-10)):
                    current_len = new_len
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = current_blocks[:]
                else:
                    # Revert
                    current_blocks[i] = old_i
                    current_blocks[j] = old_j
                    current_bts[i] = block_triple_indices[old_i]
                    current_bts[j] = block_triple_indices[old_j]
                    current_cov[current_bts[i]] += 1
                    current_cov[current_bts[j]] += 1
            
            elif move_type == 1:
                # Replace one block
                i = rng.randint(len(current_blocks))
                old_block = current_blocks[i]
                old_triples = current_bts[i]
                
                current_cov[old_triples] -= 1
                uncovered = np.where(current_cov == 0)[0]
                
                if len(uncovered) == 0:
                    # Remove the block
                    current_blocks.pop(i)
                    current_bts.pop(i)
                    new_len = len(current_blocks)
                    if new_len < best_len:
                        best_len = new_len
                        best_blocks = current_blocks[:]
                    continue
                
                # Try to find a replacement
                replacement_found = False
                for _ in range(5):
                    tr_idx = uncovered[rng.randint(len(uncovered))]
                    candidates = triple_to_blocks[tr_idx]
                    if candidates:
                        new_idx = rng.choice(candidates)
                        new_block = list(all_blocks[new_idx])
                        new_triples = block_triple_indices[new_idx]
                        
                        # Check if this covers all uncovered
                        current_cov[new_triples] += 1
                        if np.all(current_cov >= 0) and np.sum(current_cov == 0) == 0:
                            current_blocks[i] = new_block
                            current_bts[i] = new_triples
                            replacement_found = True
                            break
                        else:
                            current_cov[new_triples] -= 1
                
                if not replacement_found:
                    # Revert
                    current_cov[old_triples] += 1
                    continue
            
            elif move_type == 2 and len(current_blocks) > 1:
                # Try to remove a block if redundant
                i = rng.randint(len(current_blocks))
                t_indices = current_bts[i]
                if np.all(current_cov[t_indices] > 1):
                    current_cov[t_indices] -= 1
                    current_blocks.pop(i)
                    current_bts.pop(i)
                    new_len = len(current_blocks)
                    if new_len < best_len:
                        best_len = new_len
                        best_blocks = current_blocks[:]
        
        return best_blocks, best_len

    def _is_valid(blocks):
        """Check every t-subset is covered."""
        covered = set()
        for b in blocks:
            for tr in itertools.combinations(b, t):
                covered.add(tr)
        return len(covered) == n_triples

    # 2. Run multiple strategies
    best_blocks = None
    best_len = float('inf')
    
    # Strategy 1: Weighted greedy with different decays
    for seed in range(5):
        if time.time() > deadline:
            break
        for decay in [0.8, 0.9, 0.95]:
            if time.time() > deadline:
                break
            blocks = _weighted_greedy(seed * 100 + int(decay * 100), weight_decay=decay)
            if not blocks or not _is_valid(blocks):
                continue
            cov, bts = _build_coverage(blocks)
            blocks, cov, bts = _remove_redundant(blocks, cov, bts)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
    
    # Strategy 2: Simple greedy with local improvement
    for seed in range(5):
        if time.time() > deadline:
            break
        blocks = _simple_greedy(seed)
        if not blocks or not _is_valid(blocks):
            continue
        cov, bts = _build_coverage(blocks)
        blocks, cov, bts = _remove_redundant(blocks, cov, bts)
        
        # Try to improve by replacing blocks
        for _ in range(3):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                success, blocks, cov, bts = _try_replace_block(blocks, cov, bts, i, random_seed=seed + _ + i)
                if success:
                    improved = True
                    break
            if not improved:
                break
        
        blocks, cov, bts = _remove_redundant(blocks, cov, bts)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    # Strategy 3: Simulated annealing from best solution
    if best_blocks is not None and time.time() < deadline - 20:
        cov, bts = _build_coverage(best_blocks)
        best_blocks, best_len = _simulated_annealing(
            best_blocks, cov, bts, random_seed=42,
            max_iters=20000, T_start=1.0, T_end=0.01, alpha=0.9995
        )
    
    # 3. Final cleanup
    if best_blocks is None:
        # Fallback
        blocks = _simple_greedy(0)
        if blocks and _is_valid(blocks):
            best_blocks = blocks
    
    return best_blocks if best_blocks else []
