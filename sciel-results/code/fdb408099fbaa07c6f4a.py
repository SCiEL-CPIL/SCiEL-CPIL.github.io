"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np
from collections import defaultdict


def _build_tsub_index(v, t):
    """Build a mapping from t-subset to index, and return the list of all t-subsets."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    return all_tsubs, tsub_to_idx


def _block_tsub_indices(block, t, tsub_to_idx):
    """Return list of t-subset indices covered by a block."""
    return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]


def _greedy_cover_fast(v, k, t, rng, deadline, all_tsubs, tsub_to_idx):
    """Fast greedy covering using vectorized gain computation."""
    n_tsubs = len(all_tsubs)
    
    uncovered_mask = np.ones(n_tsubs, dtype=bool)
    n_uncovered = n_tsubs
    
    blocks = []
    
    # Precompute: for each (t-1)-subset, the list of t-subset indices that extend it
    # tsub_index_by_prefix[(t-1)-tuple] -> list of (point, tsub_idx)
    tsub_index_by_prefix = {}
    for ts, idx in tsub_to_idx.items():
        for i in range(t):
            prefix = ts[:i] + ts[i+1:]
            pt = ts[i]
            if prefix not in tsub_index_by_prefix:
                tsub_index_by_prefix[prefix] = []
            tsub_index_by_prefix[prefix].append((pt, idx))
    
    while n_uncovered > 0:
        if time.time() > deadline:
            return None
        
        uncovered_indices = np.where(uncovered_mask)[0]
        start_idx = int(rng.choice(uncovered_indices))
        start_tsub = all_tsubs[start_idx]
        block = list(start_tsub)
        block_set = set(block)
        
        while len(block) < k:
            if t - 1 <= len(block):
                block_combos = list(itertools.combinations(block, t - 1))
            else:
                block_combos = []
            
            if not block_combos:
                candidates = [x for x in range(v) if x not in block_set]
                best_point = rng.choice(candidates)
                block.append(best_point)
                block_set.add(best_point)
                continue
            
            # Vectorized gain computation
            gains = np.zeros(v, dtype=np.int32)
            for rest in block_combos:
                prefix = tuple(sorted(rest))
                entries = tsub_index_by_prefix.get(prefix)
                if not entries:
                    continue
                for pt, idx in entries:
                    if pt not in block_set and uncovered_mask[idx]:
                        gains[pt] += 1
            
            # Mask out points already in block
            for p in block_set:
                gains[p] = -1
            
            max_gain = gains.max()
            if max_gain <= 0:
                candidates = [x for x in range(v) if x not in block_set]
                best_point = rng.choice(candidates)
            else:
                best_points = np.where(gains == max_gain)[0]
                best_point = int(rng.choice(best_points))
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts, -1)
            if idx >= 0 and uncovered_mask[idx]:
                uncovered_mask[idx] = False
                n_uncovered -= 1
    
    return blocks


def _compute_cover_count(blocks, v, t, tsub_to_idx):
    """Compute coverage count for each t-subset."""
    cover_count = np.zeros(len(tsub_to_idx), dtype=np.int32)
    for block in blocks:
        for idx in _block_tsub_indices(block, t, tsub_to_idx):
            cover_count[idx] += 1
    return cover_count


def _fitness(blocks, v, t, tsub_to_idx):
    """Fitness: number of blocks + penalty for uncovered t-subsets."""
    if not blocks:
        return float('inf')
    
    cover_count = _compute_cover_count(blocks, v, t, tsub_to_idx)
    n_uncovered = np.sum(cover_count == 0)
    
    # Primary objective: minimize block count
    # Secondary: minimize uncovered t-subsets (penalty)
    return len(blocks) + 0.01 * n_uncovered


def _crossover(parent1, parent2, v, k, t, rng, tsub_to_idx):
    """Crossover: combine blocks from two parents, keeping only those that contribute."""
    # Union of all blocks from both parents
    all_blocks = list(parent1) + list(parent2)
    
    # Compute coverage
    cover_count = defaultdict(int)
    for block in all_blocks:
        for ts in itertools.combinations(block, t):
            cover_count[ts] += 1
    
    # Greedily select blocks that cover unique t-subsets
    selected = []
    selected_cover = defaultdict(int)
    
    # Sort by "uniqueness" - blocks with more unique coverage first
    def uniqueness_score(block):
        unique = 0
        for ts in itertools.combinations(block, t):
            if cover_count[ts] == 1:
                unique += 1
        return unique
    
    sorted_blocks = sorted(all_blocks, key=uniqueness_score, reverse=True)
    
    for block in sorted_blocks:
        # Check if this block adds any new coverage
        adds_new = False
        for ts in itertools.combinations(block, t):
            if selected_cover.get(ts, 0) == 0:
                adds_new = True
                break
        
        if adds_new:
            selected.append(block)
            for ts in itertools.combinations(block, t):
                selected_cover[ts] += 1
    
    # Remove redundant blocks from selected
    selected = _remove_redundant_blocks(selected, v, t)
    
    return selected


def _mutate_block(block, v, k, t, rng, cover_count, tsub_to_idx):
    """Mutate a block: replace 1-2 points with random ones."""
    new_block = list(block)
    n_mutate = rng.randint(1, 2)
    
    for _ in range(n_mutate):
        # Remove a random point
        idx = rng.randint(0, k - 1)
        removed = new_block.pop(idx)
        
        # Add a random new point
        candidates = [x for x in range(v) if x not in new_block]
        new_point = rng.choice(candidates)
        new_block.append(new_point)
        new_block.sort()
    
    return new_block


def _remove_redundant_blocks(blocks, v, t):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    cover_count = defaultdict(int)
    block_tsubs = []
    
    for block in blocks:
        tsubs = list(itertools.combinations(block, t))
        block_tsubs.append(tsubs)
        for ts in tsubs:
            cover_count[ts] += 1
    
    changed = True
    while changed:
        changed = False
        kept_blocks = []
        for i, block in enumerate(blocks):
            all_redundant = True
            for ts in block_tsubs[i]:
                if cover_count[ts] <= 1:
                    all_redundant = False
                    break
            if not all_redundant:
                kept_blocks.append(block)
            else:
                changed = True
                for ts in block_tsubs[i]:
                    cover_count[ts] -= 1
        
        if changed:
            blocks = kept_blocks
            block_tsubs = []
            for block in blocks:
                tsubs = list(itertools.combinations(block, t))
                block_tsubs.append(tsubs)
    
    return blocks


def _polish_covering(blocks, v, k, t, rng, deadline, all_tsubs, tsub_to_idx):
    """Final polish: aggressive block replacement, block-merge, and redundancy removal."""
    if not blocks:
        return blocks
    
    current_blocks = [list(b) for b in blocks]
    best_blocks = [list(b) for b in current_blocks]
    best_count = len(current_blocks)
    
    # Build cover count using index-based approach
    cover_count = np.zeros(len(tsub_to_idx), dtype=np.int32)
    block_indices = []
    for block in current_blocks:
        idxs = _block_tsub_indices(block, t, tsub_to_idx)
        block_indices.append(idxs)
        for idx in idxs:
            cover_count[idx] += 1
    
    # Precompute prefix index for fast gain
    tsub_index_by_prefix = {}
    for ts, idx in tsub_to_idx.items():
        for i in range(t):
            prefix = ts[:i] + ts[i+1:]
            pt = ts[i]
            if prefix not in tsub_index_by_prefix:
                tsub_index_by_prefix[prefix] = []
            tsub_index_by_prefix[prefix].append((pt, idx))
    
    max_passes = 15
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        
        # Pass 1: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break
                is_redundant = True
                for idx in block_indices[i]:
                    if cover_count[idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in block_indices[i]:
                        cover_count[idx] -= 1
                    current_blocks.pop(i)
                    block_indices.pop(i)
                    changed = True
                    if len(current_blocks) < best_count:
                        best_count = len(current_blocks)
                        best_blocks = [list(b) for b in current_blocks]
        
        # Pass 2: Try to merge pairs of blocks into one
        merge_improved = False
        n_blocks = len(current_blocks)
        for i in range(n_blocks):
            if time.time() > deadline:
                break
            unique_i = [idx for idx in block_indices[i] if cover_count[idx] == 1]
            if not unique_i:
                continue
            for j in range(i + 1, n_blocks):
                if time.time() > deadline:
                    break
                unique_j = [idx for idx in block_indices[j] if cover_count[idx] == 1]
                if not unique_j:
                    continue
                
                # Try to build a single block covering all unique t-subsets from both
                all_unique = unique_i + unique_j
                merged_block = None
                for _trial in range(8):
                    start_idx = int(rng.choice(all_unique))
                    candidate = list(all_tsubs[start_idx])
                    cand_set = set(candidate)
                    success = True
                    while len(candidate) < k:
                        block_combos = list(itertools.combinations(candidate, t - 1))
                        gains = np.zeros(v, dtype=np.int32)
                        for rest in block_combos:
                            prefix = tuple(sorted(rest))
                            entries = tsub_index_by_prefix.get(prefix)
                            if not entries:
                                continue
                            for pt, idx in entries:
                                if pt not in cand_set and cover_count[idx] <= 1:
                                    gains[pt] += 1
                        for p in cand_set:
                            gains[p] = -1
                        max_gain = gains.max()
                        if max_gain <= 0:
                            candidates = [x for x in range(v) if x not in cand_set]
                            best_point = int(rng.choice(candidates))
                        else:
                            best_points = np.where(gains == max_gain)[0]
                            best_point = int(rng.choice(best_points))
                        candidate.append(best_point)
                        cand_set.add(best_point)
                    candidate.sort()
                    # Verify covers all unique t-subsets
                    cand_tsub_set = set(itertools.combinations(candidate, t))
                    if all(all_tsubs[idx] in cand_tsub_set for idx in all_unique):
                        merged_block = candidate
                        break
                
                if merged_block is not None:
                    # Check no duplicates
                    if merged_block not in current_blocks:
                        # Remove blocks i and j
                        for idx in block_indices[j]:
                            cover_count[idx] -= 1
                        for idx in block_indices[i]:
                            cover_count[idx] -= 1
                        current_blocks.pop(j)
                        block_indices.pop(j)
                        current_blocks.pop(i)
                        block_indices.pop(i)
                        
                        # Add merged block
                        new_idxs = _block_tsub_indices(merged_block, t, tsub_to_idx)
                        for idx in new_idxs:
                            cover_count[idx] += 1
                        current_blocks.append(merged_block)
                        block_indices.append(new_idxs)
                        
                        if len(current_blocks) < best_count:
                            best_count = len(current_blocks)
                            best_blocks = [list(b) for b in current_blocks]
                        merge_improved = True
                        break
            if merge_improved:
                break
        
        if not merge_improved:
            # Pass 3: Try single-block replacement
            improved_any = False
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                unique_idxs = [idx for idx in block_indices[i] if cover_count[idx] == 1]
                if not unique_idxs:
                    continue
                
                best_new_block = None
                best_new_gain = 0
                for _trial in range(12):
                    start_idx = int(rng.choice(unique_idxs))
                    candidate = list(all_tsubs[start_idx])
                    cand_set = set(candidate)
                    while len(candidate) < k:
                        block_combos = list(itertools.combinations(candidate, t - 1))
                        gains = np.zeros(v, dtype=np.int32)
                        for rest in block_combos:
                            prefix = tuple(sorted(rest))
                            entries = tsub_index_by_prefix.get(prefix)
                            if not entries:
                                continue
                            for pt, idx in entries:
                                if pt not in cand_set and cover_count[idx] == 0:
                                    gains[pt] += 1
                        for p in cand_set:
                            gains[p] = -1
                        max_gain = gains.max()
                        if max_gain <= 0:
                            candidates = [x for x in range(v) if x not in cand_set]
                            best_point = int(rng.choice(candidates))
                        else:
                            best_points = np.where(gains == max_gain)[0]
                            best_point = int(rng.choice(best_points))
                        candidate.append(best_point)
                        cand_set.add(best_point)
                    candidate.sort()
                    cand_tsub_set = set(itertools.combinations(candidate, t))
                    if all(all_tsubs[idx] in cand_tsub_set for idx in unique_idxs):
                        add_gain = 0
                        for ts in cand_tsub_set:
                            idx = tsub_to_idx.get(ts, -1)
                            if idx >= 0 and cover_count[idx] == 0:
                                add_gain += 1
                        if add_gain > best_new_gain:
                            best_new_gain = add_gain
                            best_new_block = candidate
                
                if best_new_block is not None and best_new_block not in current_blocks:
                    for idx in block_indices[i]:
                        cover_count[idx] -= 1
                    new_idxs = _block_tsub_indices(best_new_block, t, tsub_to_idx)
                    for idx in new_idxs:
                        cover_count[idx] += 1
                    current_blocks[i] = best_new_block
                    block_indices[i] = new_idxs
                    improved_any = True
            
            if not improved_any and not merge_improved:
                break
    
    return best_blocks


def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    all_tsubs = set(itertools.combinations(range(v), t))
    return all_tsubs.issubset(covered)


def _local_search(blocks, v, k, t, rng, deadline, all_tsubs, tsub_to_idx):
    """Local search: try removing blocks and replacing blocks to reduce total block count."""
    if not blocks:
        return blocks
    
    current_blocks = [list(b) for b in blocks]
    best_blocks = [list(b) for b in current_blocks]
    best_count = len(current_blocks)
    
    # Use numpy-based cover count for speed
    cover_count = np.zeros(len(tsub_to_idx), dtype=np.int32)
    block_indices = []
    for block in current_blocks:
        idxs = _block_tsub_indices(block, t, tsub_to_idx)
        block_indices.append(idxs)
        for idx in idxs:
            cover_count[idx] += 1
    
    # Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            is_redundant = True
            for idx in block_indices[i]:
                if cover_count[idx] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for idx in block_indices[i]:
                    cover_count[idx] -= 1
                current_blocks.pop(i)
                block_indices.pop(i)
                changed = True
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = [list(b) for b in current_blocks]
    
    # Block replacement with more trials
    for _iteration in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(current_blocks)):
            if time.time() > deadline:
                break
            
            unique_idxs = [idx for idx in block_indices[i] if cover_count[idx] == 1]
            
            if not unique_idxs:
                continue
            
            best_new_block = None
            best_new_gain = 0
            
            for _trial in range(25):
                start_idx = rng.choice(unique_idxs)
                start_ts = all_tsubs[start_idx]
                candidate = list(start_ts)
                cand_set = set(candidate)
                
                while len(candidate) < k:
                    block_combos = list(itertools.combinations(candidate, t - 1))
                    best_gain = -1
                    best_point = -1
                    
                    for x in range(v):
                        if x in cand_set:
                            continue
                        gain = 0
                        for rest in block_combos:
                            ts = tuple(sorted(rest + (x,)))
                            idx = tsub_to_idx.get(ts, -1)
                            if idx >= 0 and cover_count[idx] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    
                    if best_point == -1:
                        candidates = [x for x in range(v) if x not in cand_set]
                        best_point = rng.choice(candidates)
                    
                    candidate.append(best_point)
                    cand_set.add(best_point)
                
                candidate.sort()
                
                cand_tsubs = set(itertools.combinations(candidate, t))
                if all(all_tsubs[idx] in cand_tsubs for idx in unique_idxs):
                    add_gain = 0
                    for ts in cand_tsubs:
                        idx = tsub_to_idx.get(ts, -1)
                        if idx >= 0 and cover_count[idx] == 0:
                            add_gain += 1
                    if add_gain > best_new_gain:
                        best_new_gain = add_gain
                        best_new_block = candidate
            
            if best_new_block is not None:
                old_block = current_blocks[i]
                for idx in block_indices[i]:
                    cover_count[idx] -= 1
                
                new_idxs = _block_tsub_indices(best_new_block, t, tsub_to_idx)
                for idx in new_idxs:
                    cover_count[idx] += 1
                
                current_blocks[i] = best_new_block
                block_indices[i] = new_idxs
                improved = True
        
        if not improved:
            break
    
    # Final redundancy removal
    changed = True
    while changed:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            is_redundant = True
            for idx in block_indices[i]:
                if cover_count[idx] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for idx in block_indices[i]:
                    cover_count[idx] -= 1
                current_blocks.pop(i)
                block_indices.pop(i)
                changed = True
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = [list(b) for b in current_blocks]
    
    return best_blocks


def _genetic_algorithm(v, k, t, rng, deadline, all_tsubs, tsub_to_idx):
    """Genetic algorithm that evolves a population of block sets."""
    population_size = 8
    max_generations = 30
    
    # Initialize population with diverse greedy solutions
    population = []
    for i in range(population_size):
        if time.time() > deadline:
            break
        rng_init = random.Random(1000 + i * 13)
        blocks = _greedy_cover_fast(v, k, t, rng_init, deadline, all_tsubs, tsub_to_idx)
        if blocks is None:
            continue
        blocks = _remove_redundant_blocks(blocks, v, t)
        population.append(blocks)
    
    if not population:
        return None
    
    # Track best solution
    best_solution = None
    best_fitness = float('inf')
    
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        # Evaluate fitness
        fitnesses = []
        for blocks in population:
            if _verify_covering(blocks, v, t):
                fit = len(blocks)
            else:
                # Penalty for invalid solutions
                cover_count = _compute_cover_count(blocks, v, t, tsub_to_idx)
                n_uncovered = np.sum(cover_count == 0)
                fit = len(blocks) + 0.1 * n_uncovered
            fitnesses.append(fit)
        
        # Update best
        min_fit_idx = np.argmin(fitnesses)
        if fitnesses[min_fit_idx] < best_fitness:
            best_fitness = fitnesses[min_fit_idx]
            best_solution = [list(b) for b in population[min_fit_idx]]
        
        # Selection: tournament selection
        def tournament_select():
            candidates = rng.sample(range(len(population)), min(3, len(population)))
            best_idx = min(candidates, key=lambda i: fitnesses[i])
            return population[best_idx]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individual
        new_population.append([list(b) for b in population[min_fit_idx]])
        
        # Fill rest with offspring
        while len(new_population) < population_size:
            if time.time() > deadline:
                break
            
            # Crossover
            if len(population) >= 2 and rng.random() < 0.7:
                p1 = tournament_select()
                p2 = tournament_select()
                child = _crossover(p1, p2, v, k, t, rng, tsub_to_idx)
            else:
                # Mutation only
                parent = tournament_select()
                child = [list(b) for b in parent]
            
            # Mutation: replace some blocks with mutated versions
            if child:
                n_mutate_blocks = max(1, len(child) // 8)
                cover_count = _compute_cover_count(child, v, t, tsub_to_idx)
                
                for _ in range(n_mutate_blocks):
                    if time.time() > deadline:
                        break
                    if not child:
                        break
                    
                    idx = rng.randint(0, len(child) - 1)
                    old_block = child[idx]
                    
                    # Try to find a better replacement
                    unique_tsubs = []
                    for ts in itertools.combinations(old_block, t):
                        idx_ts = tsub_to_idx.get(ts, -1)
                        if idx_ts >= 0 and cover_count[idx_ts] == 1:
                            unique_tsubs.append(ts)
                    
                    if unique_tsubs:
                        # Build a new block that covers these unique tsubs
                        start_ts = rng.choice(unique_tsubs)
                        new_block = list(start_ts)
                        cand_set = set(new_block)
                        
                        while len(new_block) < k:
                            block_combos = list(itertools.combinations(new_block, t - 1))
                            best_gain = -1
                            best_point = -1
                            
                            for x in range(v):
                                if x in cand_set:
                                    continue
                                gain = 0
                                for rest in block_combos:
                                    ts = tuple(sorted(rest + (x,)))
                                    idx_ts = tsub_to_idx.get(ts, -1)
                                    if idx_ts >= 0 and cover_count[idx_ts] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_point = x
                            
                            if best_point == -1:
                                candidates = [x for x in range(v) if x not in cand_set]
                                best_point = rng.choice(candidates)
                            
                            new_block.append(best_point)
                            cand_set.add(best_point)
                        
                        new_block.sort()
                        
                        # Verify it covers the unique tsubs
                        new_tsubs = set(itertools.combinations(new_block, t))
                        if all(ts in new_tsubs for ts in unique_tsubs):
                            # Update cover counts
                            for ts in itertools.combinations(old_block, t):
                                idx_ts = tsub_to_idx.get(ts, -1)
                                if idx_ts >= 0:
                                    cover_count[idx_ts] -= 1
                            for ts in new_tsubs:
                                idx_ts = tsub_to_idx.get(ts, -1)
                                if idx_ts >= 0:
                                    cover_count[idx_ts] += 1
                            
                            child[idx] = new_block
                    else:
                        # Just mutate
                        child[idx] = _mutate_block(old_block, v, k, t, rng, cover_count, tsub_to_idx)
            
            # Remove redundant blocks
            child = _remove_redundant_blocks(child, v, t)
            
            if child:
                new_population.append(child)
        
        population = new_population
    
    return best_solution


def construct(v, k, t):
    deadline = time.time() + 143  # Safety margin before 150s limit
    rng = random.Random(42)
    
    # Precompute t-subset indexing
    all_tsubs, tsub_to_idx = _build_tsub_index(v, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple greedy restarts with local search
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        rng_restart = random.Random(42 + restart * 7)
        
        blocks = _greedy_cover_fast(v, k, t, rng_restart, deadline - 20, all_tsubs, tsub_to_idx)
        if blocks is None:
            continue
        
        blocks = _remove_redundant_blocks(blocks, v, t)
        blocks = _local_search(blocks, v, k, t, rng_restart, deadline - 20, all_tsubs, tsub_to_idx)
        
        if _verify_covering(blocks, v, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
        
        if best_count <= 76:
            break
    
    # Phase 2: Final polish pass (most impactful)
    if best_blocks is not None and time.time() < deadline - 5:
        rng_polish = random.Random(777)
        polished = _polish_covering(best_blocks, v, k, t, rng_polish, deadline - 2, all_tsubs, tsub_to_idx)
        if _verify_covering(polished, v, t):
            if len(polished) < best_count:
                best_count = len(polished)
                best_blocks = [list(b) for b in polished]
    
    # Phase 3: Extra local search on best if time remains
    if best_blocks is not None and time.time() < deadline - 3:
        rng_ls = random.Random(555)
        ls_result = _local_search(best_blocks, v, k, t, rng_ls, deadline - 2, all_tsubs, tsub_to_idx)
        if _verify_covering(ls_result, v, t):
            if len(ls_result) < best_count:
                best_count = len(ls_result)
                best_blocks = [list(b) for b in ls_result]
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
