"""Covering design C(24,7,3) - Simulated Annealing over fixed-size block multiset."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin below 150s kill
    start_time = time.time()

    # Precompute all t-subsets and mapping
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute for each 7-subset (block), the list of t-subset indices it covers
    # There are C(24,7) = 346104 possible blocks. Precomputing all is expensive but
    # we can compute on-the-fly. Instead, let's precompute the mapping from a block
    # (as a tuple) to its covered t-subset indices.
    # Actually, 346104 blocks * 35 t-subsets each = ~12M entries. That's a lot of memory
    # but let's try a different approach: compute coverage on the fly but cache it.

    # Better: precompute for each point, the t-subsets it's part of? No, let's just
    # compute coverage of a block quickly.
    # A block of size 7 has C(7,3)=35 t-subsets.
    # We can precompute a list of all (block_tuple, [t_idx1, t_idx2, ...]) for all blocks.
    # 346104 blocks. Each has 35 entries. Total 12.1M integers. That's ~48MB. Feasible.
    # But generating them takes time. Let's do it once.

    # Actually, let's be smarter. We don't need ALL blocks. We only need blocks that
    # appear in our search. Let's use a cache.
    block_cache = {}

    def get_block_t_indices(block):
        """Return list of t-subset indices covered by this block."""
        key = tuple(sorted(block))
        if key in block_cache:
            return block_cache[key]
        idxs = []
        for ts in itertools.combinations(block, t):
            idxs.append(t_subset_to_idx[ts])
        block_cache[key] = idxs
        return idxs

    def is_valid_covering(blocks):
        """Check if blocks form a valid covering."""
        covered = np.zeros(n_t, dtype=np.bool_)
        for b in blocks:
            for idx in get_block_t_indices(b):
                covered[idx] = True
        return covered.all()

    def coverage_count_array(blocks):
        """Return array of coverage counts for each t-subset."""
        cc = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for idx in get_block_t_indices(b):
                cc[idx] += 1
        return cc

    # --- Initial solution: greedy ---
    def greedy_initial(seed):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            start_idx = rng.choice(list(uncovered))
            block = list(all_t_subsets[start_idx])
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(t_subset_to_idx[ts])
        return blocks

    # --- Simulated Annealing ---
    def simulated_annealing(initial_blocks, seed, time_budget):
        """
        SA over a fixed-size block list. Moves:
        1. Swap a point in one block with a point from another block (or from outside)
        2. Replace an entire block with a random one
        3. Remove a block (if it makes the list shorter) -- but we need fixed size for SA.
        
        Actually, let's allow variable size. The "energy" is the number of blocks.
        We want to minimize it while maintaining validity.
        
        Alternative: work with a fixed-size list where some blocks can be "empty" (None).
        Energy = number of non-None blocks. Validity = all t-subsets covered.
        
        Let's use a simpler approach: 
        - Start with a valid covering of size N.
        - Try to reduce N by finding a block that can be removed and its t-subsets
          covered by modifying other blocks.
        
        This is complex. Let's use a different SA formulation:
        - State: list of blocks (variable length).
        - Energy: number of blocks + penalty * (number of uncovered t-subsets).
        - Moves: add a block, remove a block, modify a block (swap point).
        - Accept if new_energy < old_energy or with probability exp(-(new-old)/T).
        """
        rng = random.Random(seed)
        blocks = list(initial_blocks)
        cover_count = coverage_count_array(blocks)
        
        def energy(blocks, cover_count):
            n_blocks = len(blocks)
            n_uncovered = np.sum(cover_count == 0)
            return n_blocks + 100.0 * n_uncovered  # heavy penalty for uncovered
        
        current_energy = energy(blocks, cover_count)
        best_blocks = list(blocks)
        best_energy = current_energy
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.01
        cooling_rate = 0.995
        
        T = T_start
        step = 0
        
        while T > T_end and time.time() < deadline - 5.0:
            step += 1
            
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.3 and len(blocks) > 1:
                # Remove a random block
                i = rng.randint(0, len(blocks) - 1)
                removed_block = blocks.pop(i)
                for idx in get_block_t_indices(removed_block):
                    cover_count[idx] -= 1
                
                new_energy = energy(blocks, cover_count)
                
                if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = list(blocks)
                else:
                    # Revert
                    blocks.append(removed_block)
                    for idx in get_block_t_indices(removed_block):
                        cover_count[idx] += 1
                    
            elif move_type < 0.6:
                # Add a random block
                # Pick a random 7-subset
                new_block = sorted(rng.sample(range(v), k))
                blocks.append(new_block)
                for idx in get_block_t_indices(new_block):
                    cover_count[idx] += 1
                
                new_energy = energy(blocks, cover_count)
                
                if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = list(blocks)
                else:
                    # Revert
                    blocks.pop()
                    for idx in get_block_t_indices(new_block):
                        cover_count[idx] -= 1
                    
            elif move_type < 0.9 and len(blocks) > 0:
                # Modify a random block: swap one point
                i = rng.randint(0, len(blocks) - 1)
                block_i = blocks[i]
                if len(block_i) != k:
                    continue
                    
                # Pick a point to remove
                x_idx = rng.randint(0, k - 1)
                x = block_i[x_idx]
                
                # Pick a point to add (not in block)
                candidates = [y for y in range(v) if y not in block_i]
                if not candidates:
                    continue
                y = rng.choice(candidates)
                
                # Compute change in coverage
                rest = [p for p in block_i if p != x]
                lost = []
                gained = []
                for a, b in itertools.combinations(rest, t - 1):
                    ts_lost = tuple(sorted((x, a, b)))
                    idx_lost = t_subset_to_idx.get(ts_lost)
                    if idx_lost is not None:
                        lost.append(idx_lost)
                    ts_gained = tuple(sorted((y, a, b)))
                    idx_gained = t_subset_to_idx.get(ts_gained)
                    if idx_gained is not None:
                        gained.append(idx_gained)
                
                # Apply
                for idx in lost:
                    cover_count[idx] -= 1
                for idx in gained:
                    cover_count[idx] += 1
                
                new_block_i = list(block_i)
                new_block_i[x_idx] = y
                new_block_i.sort()
                blocks[i] = new_block_i
                
                new_energy = energy(blocks, cover_count)
                
                if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = list(blocks)
                else:
                    # Revert
                    for idx in gained:
                        cover_count[idx] -= 1
                    for idx in lost:
                        cover_count[idx] += 1
                    blocks[i] = block_i
                    
            else:
                # Swap a point between two blocks
                if len(blocks) < 2:
                    continue
                i, j = rng.sample(range(len(blocks)), 2)
                block_i = blocks[i]
                block_j = blocks[j]
                
                if len(block_i) != k or len(block_j) != k:
                    continue
                
                # Pick point from block_i to move to block_j
                x_idx = rng.randint(0, k - 1)
                x = block_i[x_idx]
                
                # Pick point from block_j to move to block_i
                y_idx = rng.randint(0, k - 1)
                y = block_j[y_idx]
                
                if x == y:
                    continue
                
                # Check that x is not in block_j and y is not in block_i
                if x in block_j or y in block_i:
                    continue
                
                # Compute changes
                rest_i = [p for p in block_i if p != x]
                rest_j = [p for p in block_j if p != y]
                
                # Block i loses x, gains y
                lost_i = []
                gained_i = []
                for a, b in itertools.combinations(rest_i, t - 1):
                    ts_l = tuple(sorted((x, a, b)))
                    idx_l = t_subset_to_idx.get(ts_l)
                    if idx_l is not None:
                        lost_i.append(idx_l)
                    ts_g = tuple(sorted((y, a, b)))
                    idx_g = t_subset_to_idx.get(ts_g)
                    if idx_g is not None:
                        gained_i.append(idx_g)
                
                # Block j loses y, gains x
                lost_j = []
                gained_j = []
                for a, b in itertools.combinations(rest_j, t - 1):
                    ts_l = tuple(sorted((y, a, b)))
                    idx_l = t_subset_to_idx.get(ts_l)
                    if idx_l is not None:
                        lost_j.append(idx_l)
                    ts_g = tuple(sorted((x, a, b)))
                    idx_g = t_subset_to_idx.get(ts_g)
                    if idx_g is not None:
                        gained_j.append(idx_g)
                
                # Apply
                for idx in lost_i + lost_j:
                    cover_count[idx] -= 1
                for idx in gained_i + gained_j:
                    cover_count[idx] += 1
                
                new_block_i = list(block_i)
                new_block_i[x_idx] = y
                new_block_i.sort()
                new_block_j = list(block_j)
                new_block_j[y_idx] = x
                new_block_j.sort()
                blocks[i] = new_block_i
                blocks[j] = new_block_j
                
                new_energy = energy(blocks, cover_count)
                
                if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = list(blocks)
                else:
                    # Revert
                    for idx in gained_i + gained_j:
                        cover_count[idx] -= 1
                    for idx in lost_i + lost_j:
                        cover_count[idx] += 1
                    blocks[i] = block_i
                    blocks[j] = block_j
            
            T *= cooling_rate
        
        return best_blocks

    # --- Main search: multiple SA runs with different initial solutions ---
    best_blocks = None
    best_count = float('inf')
    
    # Generate a few diverse initial solutions
    num_initials = 5
    for i in range(num_initials):
        if time.time() > deadline - 10.0:
            break
        init_blocks = greedy_initial(i)
        if len(init_blocks) < best_count:
            best_count = len(init_blocks)
            best_blocks = init_blocks
    
    # Run SA from the best initial solution and a few others
    sa_budget_per_run = 25.0  # seconds per SA run
    num_sa_runs = 4
    
    for i in range(num_sa_runs):
        if time.time() > deadline - 10.0:
            break
        
        # Use the best initial or a different one
        if i == 0:
            init_blocks = best_blocks
        else:
            init_blocks = greedy_initial(i + 10)
            if len(init_blocks) >= best_count:
                continue
        
        sa_result = simulated_annealing(init_blocks, i + 50, sa_budget_per_run)
        
        # Verify
        if is_valid_covering(sa_result):
            if len(sa_result) < best_count:
                best_count = len(sa_result)
                best_blocks = sa_result
    
    # Final local search: try to remove redundant blocks
    if best_blocks is not None:
        # Try removing blocks one by one
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(best_blocks)):
                if time.time() > deadline - 5.0:
                    break
                # Try removing block i
                test_blocks = best_blocks[:i] + best_blocks[i+1:]
                if is_valid_covering(test_blocks):
                    best_blocks = test_blocks
                    best_count = len(best_blocks)
                    improved = True
                    break  # Restart with new list
    
    # Ensure validity
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_initial(0)
    
    if not is_valid_covering(best_blocks):
        # Shouldn't happen, but fallback
        best_blocks = greedy_initial(0)
    
    return best_blocks
