import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, store the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        idxs = [t_sub_to_idx[t_sub] for t_sub in itertools.combinations(k_sub, t)]
        k_sub_t_idxs.append(idxs)

    # For each t-subset, store the list of k-subset indices that cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def time_ok():
        return time.time() < deadline

    def greedy_cover(uncovered_set, used_blocks_set):
        """Greedy set cover. Returns list of block indices."""
        blocks = []
        uncovered = set(uncovered_set)
        
        while uncovered and time_ok():
            # Pick an uncovered t-subset. 
            # Heuristic: pick one that is in the fewest remaining candidate blocks?
            # Or just pick the first one? 
            # Standard greedy: pick the set (block) that covers the most uncovered elements.
            # To speed up, we can iterate through uncovered t-subsets and for each, 
            # look at its candidate blocks and pick the best one globally? No, that's expensive.
            
            # Efficient Greedy:
            # 1. Find the uncovered t-subset with the fewest candidate blocks (most constrained).
            #    This is a common heuristic for exact cover / set cover.
            #    Alternatively, just pick the first uncovered t-subset and find the best block for it.
            #    The "best block for a specific uncovered element" strategy is O(n_k * t) per step.
            #    The "global best block" strategy is O(n_k * t) per step too, but scans all blocks.
            
            # Let's use the "most constrained element" heuristic.
            # Find s in uncovered with min len([b for b in t_sub_to_k_idxs[s] if b not in used_blocks_set])
            # Actually, used_blocks_set is not strictly necessary for validity, only for avoiding duplicates.
            # But we want distinct blocks.
            
            # Let's stick to the standard greedy:
            # For each candidate block not yet used, calculate gain.
            # This is slow if done naively.
            
            # Optimized Greedy:
            # Pick an arbitrary uncovered t-subset s.
            # Among all blocks containing s, pick the one with max gain.
            # This is often sufficient and fast.
            
            s_idx = next(iter(uncovered))
            candidates = t_sub_to_k_idxs[s_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                if b_idx in used_blocks_set:
                    continue
                gain = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                # Should not happen if covering number is finite and we haven't run out of blocks
                # But if all candidates are used, we might be stuck? No, we just need ANY block.
                # If all candidates for s_idx are used, we still need to cover s_idx.
                # This implies we need to allow re-using blocks? No, blocks must be distinct.
                # If all blocks containing s_idx are already used, then s_idx is already covered by those blocks.
                # So s_idx should have been removed from uncovered.
                # Thus, this case shouldn't occur if logic is correct.
                break

            blocks.append(best_block_idx)
            used_blocks_set.add(best_block_idx)
            
            for t_idx in k_sub_t_idxs[best_block_idx]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
        
        return blocks

    def reduce_blocks(block_indices):
        """Remove redundant blocks. Returns new list of block indices."""
        if not block_indices:
            return []
            
        # Count coverage for each t-subset
        cover_count = np.zeros(n_t, dtype=np.int32)
        # Map block index to its position in the list for quick removal? 
        # Or just iterate.
        
        for b_idx in block_indices:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        # Iteratively remove blocks that don't have any unique coverage
        # A block is redundant if for every t-subset it covers, cover_count > 1.
        # If we remove it, we must decrement counts.
        
        # To do this efficiently, we can keep a list of blocks and check each.
        # Since the number of blocks is ~300-400, and n_t is ~11k, this is manageable.
        
        current_blocks = list(block_indices)
        changed = True
        while changed and time_ok():
            changed = False
            new_blocks = []
            for b_idx in current_blocks:
                # Check if any t-subset in this block has cover_count == 1
                essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        essential = True
                        break
                
                if essential:
                    new_blocks.append(b_idx)
                else:
                    # Redundant
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            
            current_blocks = new_blocks
        
        return current_blocks

    def local_search(block_indices, max_iter=50):
        """Try to swap blocks to reduce size or improve coverage? 
        For covering design, we just want fewer blocks.
        Local search: try removing a block and replacing it with a different one that covers the lost t-subsets with fewer new blocks?
        Or simply: try to find a smaller set.
        
        A simple local search: 
        1. Pick a random block B.
        2. Let U be the set of t-subsets covered ONLY by B (i.e., cover_count == 1).
        3. If U is empty, B is redundant (handled by reduction).
        4. If U is not empty, try to find a set of blocks that cover U.
           If we can cover U with fewer than 1 block (impossible) or if we can replace B with a different block B' that covers U and possibly more, but we are trying to REDUCE count.
           So we can only remove B if U is empty.
           
        Alternative local search:
        Try to replace two blocks with one? Unlikely to find a single block covering all t-subsets of two blocks.
        
        Maybe a better local search is "swap":
        Remove B. Let U be uncovered t-subsets.
        Find a block B' that covers the most of U.
        If B' covers ALL of U, we can replace B with B'. Size remains same.
        But we want to REDUCE size.
        
        Actually, the reduction step handles all "free" removals.
        To reduce further, we need to find a structural improvement.
        
        Let's try a heuristic: 
        Restart greedy with different random seeds or ordering to find a better local minimum.
        """
        pass

    # Main strategy:
    # 1. Run greedy multiple times with different randomizations to find the best covering.
    # 2. Apply reduction to each.
    # 3. Keep the best (smallest) one.
    
    best_blocks = None
    best_size = float('inf')
    
    num_restarts = 20 # Adjust based on time
    
    for i in range(num_restarts):
        if not time_ok():
            break
            
        # Randomize the order of t-subsets or the selection of s_idx
        # In greedy_cover, we pick s_idx = next(iter(uncovered)).
        # To randomize, we can shuffle the uncovered set? Sets are unordered.
        # We can instead pick a random uncovered t-subset.
        
        # Let's modify greedy_cover to accept a random seed or use random.choice
        
        def greedy_cover_random(uncovered_set, used_blocks_set, rng):
            blocks = []
            uncovered = set(uncovered_set)
            
            while uncovered and time_ok():
                # Pick a random uncovered t-subset
                s_idx = rng.choice(list(uncovered))
                candidates = t_sub_to_k_idxs[s_idx]
                
                best_block_idx = -1
                best_gain = -1
                
                # Shuffle candidates to break ties randomly
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for b_idx in cand_list:
                    if b_idx in used_blocks_set:
                        continue
                    gain = 0
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if t_idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
                
                if best_block_idx == -1:
                    break

                blocks.append(best_block_idx)
                used_blocks_set.add(best_block_idx)
                
                for t_idx in k_sub_t_idxs[best_block_idx]:
                    if t_idx in uncovered:
                        uncovered.remove(t_idx)
            
            return blocks

        rng = random.Random(i)
        used = set()
        blocks = greedy_cover_random(set(range(n_t)), used, rng)
        
        # Reduce
        reduced = reduce_blocks(blocks)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_blocks = reduced
            
        # If we've hit the theoretical lower bound or a very good number, we could stop early?
        # Lower bound is 264. If we get close, keep going.
        
    if best_blocks is None:
        # Fallback to a single deterministic run if all else fails (shouldn't happen)
        used = set()
        blocks = greedy_cover(set(range(n_t)), used)
        best_blocks = reduce_blocks(blocks)
        
    # Convert indices back to lists of points
    return [list(all_k_subs[idx]) for idx in best_blocks]
