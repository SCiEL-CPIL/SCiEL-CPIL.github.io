import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute block_t_indices as a list of lists for easier manipulation in local search
    # Or keep numpy for greedy, convert for LS?
    # Numpy is better for greedy.
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    best_solution = None
    best_score = float('inf')
    
    def get_coverage_count(blocks_indices):
        """Returns coverage count array for a list of block indices."""
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in blocks_indices:
            coverage_count[block_t_indices[b_idx]] += 1
        return coverage_count

    def is_valid_solution(blocks_indices):
        """Check if solution covers all t-subsets."""
        if not blocks_indices:
            return n_ts == 0
        coverage_count = get_coverage_count(blocks_indices)
        return np.min(coverage_count) > 0

    def polish_solution(blocks_indices, max_iters=50):
        """
        Applies deletion and simple swaps to reduce block count.
        Returns the polished list of indices.
        """
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks
            
        coverage_count = get_coverage_count(current_blocks)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            improved = False
            
            # 1. Deletion Pass
            # Find blocks that are redundant (all their t-subsets are covered by other blocks)
            # A block is redundant if for all t in block, coverage_count[t] > 1
            # We can check this efficiently.
            
            # Recompute coverage count if it got stale? 
            # In this loop, we only remove or swap. 
            # If we remove, coverage drops. If we swap, coverage changes.
            # Let's recompute at start of each iteration for safety, it's fast enough for <100 blocks.
            coverage_count = get_coverage_count(current_blocks)
            
            # Check for redundant blocks
            # For each block, check if min coverage of its t-subsets > 1
            redundant_indices = []
            for b_idx in current_blocks:
                subs = block_t_indices[b_idx]
                if np.min(coverage_count[subs]) > 1:
                    redundant_indices.append(b_idx)
                    
            if redundant_indices:
                # Remove all redundant blocks at once? 
                # Removing one might make another redundant.
                # Safer to remove one, update, and repeat.
                # But removing all is faster and still valid if they were all redundant w.r.t original set.
                # Actually, if A and B are both redundant, removing A might make B non-redundant? 
                # No, redundancy means "covered by others". If A is covered by others, and B is covered by others, 
                # removing A doesn't uncover B's subsets unless A was one of the "others" covering B.
                # But if A is redundant, its subsets are covered by blocks other than A.
                # So removing A is safe.
                # We can remove all identified redundant blocks.
                for r_idx in redundant_indices:
                    if r_idx in current_blocks:
                        current_blocks.remove(r_idx)
                improved = True
                continue # Restart loop to re-evaluate redundancy
            
            # 2. Swap Pass (1-for-1)
            # Try to replace a block with another that allows further deletion or is just a better fit.
            # To keep it fast, we try a few random swaps.
            if len(current_blocks) > 0:
                # Pick a random block to potentially swap
                b_idx = random.choice(current_blocks)
                # Remove it temporarily
                temp_blocks = [b for b in current_blocks if b != b_idx]
                
                # Find holes created by removing b_idx
                # Holes are t-subsets where coverage becomes 0
                # coverage_count is for current_blocks (with b_idx)
                # We need coverage_count for temp_blocks
                # Let's compute holes directly
                subs_in_block = block_t_indices[b_idx]
                # A t-subset is a hole if it was ONLY covered by b_idx
                # i.e., coverage_count[sub] == 1
                
                # Identify holes
                holes = []
                for sub in subs_in_block:
                    if coverage_count[sub] == 1:
                        holes.append(sub)
                        
                if not holes:
                    # If no holes, b_idx was redundant, which should have been caught above.
                    # So if we get here, there are holes.
                    pass
                else:
                    # Find best block to cover holes
                    # Score a block by how many holes it covers
                    # We want a block that covers ALL holes.
                    # If such a block exists, we can swap.
                    
                    # To speed up, we can filter candidates.
                    # A candidate must cover at least 1 hole.
                    # Let's just iterate through all blocks? n_ks is 24504. 
                    # Doing this for every block in solution is too slow.
                    # We only do this for ONE random block per iteration.
                    
                    best_candidate = -1
                    max_holes_covered = 0
                    covers_all = False
                    
                    # Optimization: Pre-compute which blocks cover which holes? No, too big.
                    # Just loop through all_ks.
                    # 24504 blocks * 35 t-subsets per block.
                    # 24504 * 35 = 857k operations.
                    # This is ~1ms in Python? Maybe 5-10ms.
                    # We do this maybe 100 times? 1s. Acceptable.
                    
                    for cand_idx in range(n_ks):
                        if cand_idx == b_idx:
                            continue
                        # Check if cand covers all holes
                        # Holes are specific t-subsets.
                        # We can check intersection of cand's t-subsets with holes.
                        cand_subs = block_t_indices[cand_idx]
                        
                        # Fast check: Does cand cover all holes?
                        # We can use a set of holes for O(1) lookup?
                        # Holes are small (<=35).
                        hole_set = set(holes)
                        
                        # Count how many holes cand covers
                        # If len(holes) is small, this is fast.
                        # If cand covers all, we accept.
                        
                        # Optimization: Early exit if cand doesn't cover first hole?
                        if not hole_set:
                            continue
                            
                        # Check if cand covers all holes
                        # Iterate through holes
                        all_covered = True
                        for h in hole_set:
                            # Check if h is in cand_subs
                            # cand_subs is a numpy array of ints.
                            # np.isin is slow for small arrays.
                            # Use a set for cand_subs?
                            # Pre-computing sets for all blocks is memory heavy but fast lookup.
                            # Let's use a global list of sets for block t-subsets?
                            # Or just check:
                            if h not in cand_subs:
                                all_covered = False
                                break
                                
                        if all_covered:
                            # Found a candidate that covers all holes
                            # We can swap
                            # Check if this candidate is already in temp_blocks?
                            # If so, we are just removing b_idx (which is good)
                            # If not, we add it.
                            
                            # Actually, if cand is in temp_blocks, we just remove b_idx.
                            # This is a pure deletion, which should have been caught?
                            # No, deletion check was "min coverage > 1".
                            # If cand covers all holes, and cand is in temp_blocks, 
                            # then holes were covered by cand (and others).
                            # So coverage_count[hole] >= 1 (from cand).
                            # Since coverage_count[hole] == 1 (from b_idx only), 
                            # and cand covers it, then coverage_count[hole] must be >= 2?
                            # Contradiction.
                            # If coverage_count[hole] == 1, only b_idx covers it.
                            # So no other block (including cand) covers it.
                            # So cand CANNOT cover the hole if coverage_count[hole] == 1.
                            # Therefore, a swap that covers ALL holes is IMPOSSIBLE if holes exist 
                            # and were uniquely covered by b_idx.
                            # Wait.
                            # If hole h is covered ONLY by b_idx, then no other block covers h.
                            # So no candidate can cover h.
                            # So we can never replace a block that uniquely covers a hole with a single other block.
                            # We would need to add a block that covers the hole, but that block doesn't exist?
                            # No, it might exist, but if it existed, it would have covered h, so coverage_count[h] would be > 1.
                            # So, if coverage_count[h] == 1, NO other block covers h.
                            # Thus, we cannot swap b_idx with any single block to maintain coverage of h.
                            # We must keep b_idx OR add a NEW block that covers h (which is impossible as no block covers h? 
                            # Wait. "No other block" means no block in the CURRENT solution.
                            # There might be a block NOT in the solution that covers h.
                            # Yes!
                            # So we CAN swap b_idx with a new block cand that covers h.
                            
                            # So the logic holds: Find cand NOT in current solution (or even if it is, but if it is, it covers h, contradiction) 
                            # that covers all holes.
                            
                            # If we find such a cand:
                            # Remove b_idx, Add cand.
                            # Size remains same.
                            # But now we have a different set.
                            # This might allow further deletions.
                            
                            # Accept swap
                            current_blocks.remove(b_idx)
                            if cand_idx not in current_blocks:
                                current_blocks.append(cand_idx)
                            improved = True
                            break # Restart polish loop
                    
                    if improved:
                        continue
                        
            if not improved:
                break
                
        return current_blocks

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Randomized selection from top gains
            # Use a soft-max or just top-K random
            if n_ks > 20:
                # Get top 20
                top_indices = np.argpartition(gains, -20)[-20:]
            else:
                top_indices = np.arange(n_ks)
                
            # Weighted random choice?
            # Just random for now
            best_idx = top_indices[random.randint(0, len(top_indices) - 1)]
            
            blocks_indices.append(best_idx)
            
            subs_to_cover = block_t_indices[best_idx]
            prev_state = covered[subs_to_cover]
            new_covers = np.sum(~prev_state)
            remaining -= new_covers
            covered[subs_to_cover] = True
            
        current_blocks = list(blocks_indices)
        
        # Polish
        polished = polish_solution(current_blocks)
        
        return polished, len(polished)

    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
        
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
