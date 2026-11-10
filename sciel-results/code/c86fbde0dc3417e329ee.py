import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute block masks as numpy arrays of indices for fast counting
    # We store the list of t-subset indices covered by each block
    block_indices = []
    for b in all_blocks:
        idxs = [t_sub_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_indices.append(np.array(idxs, dtype=np.int32))
    
    # Full mask is just the set of all indices, but we track counts
    # For "bottleneck" heuristic, we need to know the count of coverage for each t-subset
    # counts[i] = number of selected blocks covering t_subsets[i]
    
    def bottleneck_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        selected = []
        
        # We need to find the block that maximizes the minimum count among its covered subsets?
        # Or maximizes the sum of 1/(counts+1)?
        # The prompt suggests: "maximizing the minimum degree of uncovered t-subsets"
        # This is slightly ambiguous. Usually "bottleneck" means addressing the most constrained element.
        # Let's interpret as: Find the t-subset with the LOWEST count (currently 0 for all).
        # Then pick a block that covers the MAXIMUM number of these low-count subsets.
        
        # Since all start at 0, initially it's standard greedy.
        # As we add blocks, some subsets get count 1, some 0.
        # We want to cover subsets with count 0 first.
        # If multiple blocks cover the same number of 0-count subsets, tie-break by total new coverage?
        
        # Efficient implementation:
        # Maintain a list of uncovered subsets (count == 0).
        # For each candidate block, calculate how many subsets from the 'uncovered' list it covers.
        # Select the block with max such count.
        
        uncovered_set = set(range(n_t_subsets))
        
        while uncovered_set:
            best_block_idx = -1
            best_gain = 0
            
            # Optimization: If we have many blocks, scanning all is slow.
            # But n_blocks is ~75k. n_t_subsets is ~11628.
            # Scanning 75k blocks * 56 elements = 4.2M ops per iteration.
            # If we need ~300 iterations, that's 1.2B ops. Too slow in Python.
            
            # We need a faster heuristic or vectorization.
            # Let's try a random subset of blocks if the full scan is too slow?
            # Or use the "standard greedy" (max total new coverage) which is easier to vectorize?
            # The prompt specifically asks for "bottleneck" (min-degree).
            
            # Let's approximate bottleneck by prioritizing blocks that cover subsets with count 0.
            # We can precompute for each t-subset, the list of blocks that cover it.
            # t_to_blocks = defaultdict(list)
            # for b_idx, idxs in enumerate(block_indices):
            #     for idx in idxs:
            #         t_to_blocks[idx].append(b_idx)
            
            # This is memory heavy but doable. 11628 * ~1000 blocks?
            # Total incidences = 75582 * 56 = 4.2M.
            # 4.2M integers is ~32MB. Acceptable.
            
            # However, building this takes time. Let's do it once.
            pass # Placeholder, see below for actual implementation using precomputed t_to_blocks
            
        return selected

    # Precompute t_to_blocks for bottleneck heuristic
    t_to_blocks = defaultdict(list)
    for b_idx, idxs in enumerate(block_indices):
        for idx in idxs:
            t_to_blocks[idx].append(b_idx)

    def bottleneck_greedy_impl(seed=None):
        if seed is not None:
            random.seed(seed)
        
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        selected = []
        uncovered_set = set(range(n_t_subsets))
        
        # Cache for gains? No, counts change.
        
        while uncovered_set:
            if time.time() > deadline:
                return None
            
            best_block_idx = -1
            best_gain = 0
            
            # To speed up, we can sample blocks or use a heuristic.
            # But for correctness, we should scan.
            # Let's scan all blocks but optimize the inner loop using numpy if possible?
            # No, the gain depends on 'uncovered_set' which is a sparse set.
            
            # Alternative:
            # Pick a random uncovered subset s.
            # Look at all blocks covering s.
            # Among those, pick the one that covers the most OTHER uncovered subsets.
            # This is a "candidate pool" approach, much faster than scanning all 75k blocks.
            
            if uncovered_set:
                # Pick a random uncovered subset
                s = random.choice(list(uncovered_set))
                candidates = t_to_blocks[s]
                
                # Shuffle candidates to break symmetry
                random.shuffle(candidates)
                
                # Evaluate top candidates? Or all?
                # Candidates size is roughly n_blocks * t / v * ... ~ 1000-2000
                # Evaluating 2000 blocks * 56 checks = 112k ops.
                # This is fast enough.
                
                best_local_block = -1
                best_local_gain = 0
                
                for b_idx in candidates:
                    # Count how many uncovered subsets this block covers
                    gain = 0
                    for idx in block_indices[b_idx]:
                        if idx in uncovered_set:
                            gain += 1
                    
                    if gain > best_local_gain:
                        best_local_gain = gain
                        best_local_block = b_idx
                
                if best_local_block != -1:
                    selected.append(best_local_block)
                    for idx in block_indices[best_local_block]:
                        if idx in uncovered_set:
                            uncovered_set.remove(idx)
                            counts[idx] += 1 # Keep counts for potential future use, though we use set here
                else:
                    break
            else:
                break
                
        return selected

    def standard_greedy_impl(seed=None):
        """Faster standard greedy for comparison/fallback"""
        if seed is not None:
            random.seed(seed)
        
        uncovered_set = set(range(n_t_subsets))
        selected = []
        
        while uncovered_set:
            if time.time() > deadline:
                return None
            
            best_block_idx = -1
            best_gain = 0
            
            # Sample a subset of blocks to speed up?
            # Or scan all?
            # Scanning all 75k blocks with 56 checks each is 4M ops.
            # 300 iterations * 4M = 1.2B ops. Too slow in Python loop.
            
            # We MUST use the candidate pool approach or vectorization.
            # Let's stick to the bottleneck approach which uses candidate pool.
            # But standard greedy is usually better.
            # Can we do standard greedy with candidate pool?
            # Pick random uncovered s. Candidates = t_to_blocks[s].
            # Pick block in candidates with max coverage of uncovered_set.
            # This is biased but often works well for set cover.
            
            s = random.choice(list(uncovered_set))
            candidates = t_to_blocks[s]
            random.shuffle(candidates)
            
            best_local_block = -1
            best_local_gain = 0
            
            for b_idx in candidates:
                gain = 0
                for idx in block_indices[b_idx]:
                    if idx in uncovered_set:
                        gain += 1
                
                if gain > best_local_gain:
                    best_local_gain = gain
                    best_local_block = b_idx
            
            if best_local_block != -1:
                selected.append(best_local_block)
                for idx in block_indices[best_local_block]:
                    if idx in uncovered_set:
                        uncovered_set.remove(idx)
            else:
                break
                
        return selected

    def prune_redundant(blocks):
        if not blocks:
            return blocks
        
        # Count coverage
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            for idx in block_indices[b_idx]:
                counts[idx] += 1
        
        essential = [False] * len(blocks)
        for i, b_idx in enumerate(blocks):
            for idx in block_indices[b_idx]:
                if counts[idx] == 1:
                    essential[i] = True
                    break
        
        return [blocks[i] for i in range(len(blocks)) if essential[i]]

    def local_search_2for1(blocks):
        """
        Try to replace 2 blocks with 1.
        Optimization: Only consider pairs where the union of their unique coverage
        is small, or where a candidate block covers most of it.
        """
        if len(blocks) < 2:
            return None
        
        block_set = set(blocks)
        
        # Precompute coverage mask for each block in the current set?
        # Or just iterate.
        
        max_attempts = 100
        attempts = 0
        rng = random.Random(42)
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            i, j = rng.sample(range(len(blocks)), 2)
            b1 = blocks[i]
            b2 = blocks[j]
            
            # Get unique coverage of b1 and b2 relative to the rest
            # This is expensive.
            # Alternative: Just check if there exists a block b3 such that
            # (block_indices[b1] U block_indices[b2]) subset of block_indices[b3]
            # This is a sufficient condition for 2-for-1, but not necessary
            # (b3 might not cover ALL, but the rest of the blocks do).
            # However, checking subset is O(56).
            
            # Let's try the strict subset check first. It's fast.
            union_set = set(block_indices[b1]) | set(block_indices[b2])
            
            # Find candidates for b3
            # Pick an element from union_set, look at blocks covering it.
            if not union_set:
                continue
                
            elem = next(iter(union_set))
            candidates = t_to_blocks[elem]
            
            for b3 in candidates:
                if b3 in block_set and b3 != b1 and b3 != b2:
                    continue
                # Check if b3 covers all of union_set
                # Note: b3 might be in block_set or not.
                # If b3 is in block_set, we are replacing b1,b2 with b3 (which is already there? No, we remove b1,b2 and add b3).
                # If b3 is already in blocks, we just remove b1 and b2.
                
                covers_all = True
                for idx in union_set:
                    if idx not in block_indices[b3]:
                        covers_all = False
                        break
                
                if covers_all:
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    if b3 not in new_blocks:
                        new_blocks.append(b3)
                    return new_blocks
        
        return None

    best_blocks = None
    best_count = float('inf')

    # Run multiple restarts
    # The bottleneck greedy is the primary strategy.
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Alternate between bottleneck and standard (candidate pool) greedy
        if i % 2 == 0:
            blocks = bottleneck_greedy_impl(seed=i)
        else:
            blocks = standard_greedy_impl(seed=i)
        
        if blocks is None:
            continue
        
        blocks = prune_redundant(blocks)
        
        # Local search
        improved = True
        max_ls_iters = 10
        ls_iter = 0
        while improved and time.time() < deadline and ls_iter < max_ls_iters:
            improved = False
            ls_iter += 1
            
            new_blocks = local_search_2for1(blocks)
            if new_blocks is not None:
                blocks = new_blocks
                improved = True
                blocks = prune_redundant(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        if best_count <= 335:
            break

    if best_blocks is None:
        # Fallback
        blocks = standard_greedy_impl(seed=None)
        if blocks:
            return [list(all_blocks[i]) for i in blocks]
        return []

    return [list(all_blocks[i]) for i in best_blocks]
