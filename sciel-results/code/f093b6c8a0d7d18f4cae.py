import time
import random
import itertools
import math
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 118.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets and their indices
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    B = len(all_blocks)
    
    # block_cover[bi] = list of t-subset indices covered by block bi
    # t_to_blocks[ti] = list of block indices that cover t-subset ti
    block_cover = []
    t_to_blocks = [[] for _ in range(N)]
    
    for bi, b in enumerate(all_blocks):
        covered = [idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        for ti in covered:
            t_to_blocks[ti].append(bi)
            
    # Convert to numpy arrays for faster counting? 
    # Actually, lists are often faster for small iterations in Python if we don't do vectorized ops on the whole array frequently.
    # But we need to count coverage. Let's stick to lists for block_cover and use a numpy array for counts.
    
    def greedy_cover():
        """Standard greedy set cover."""
        uncovered = set(range(N))
        blocks = []
        while uncovered:
            seed_idx = next(iter(uncovered))
            candidates = t_to_blocks[seed_idx]
            best_b = -1
            max_cov = -1
            # Optimization: If we have a time budget, we might restrict candidates, 
            # but for initialization, we want a good start.
            # B is ~75k. N is ~11628.
            # Iterating 75k items * 11k steps is too slow if done naively.
            # Greedy picks the best among candidates of a seed.
            for bi in candidates:
                cov = 0
                # Count how many of this block's covered subsets are currently uncovered
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
            if best_b == -1:
                break
            blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
        return blocks

    def local_search(blocks, max_iters=5000):
        """
        Improve a valid covering using 1-exchange (drop 1, add 1) and 2-exchange (drop 2, add 2).
        We maintain counts of how many times each t-subset is covered.
        """
        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks
            
        # Initialize counts
        cnt = np.zeros(N, dtype=np.int32)
        in_set = np.zeros(B, dtype=bool)
        for bi in current_blocks:
            in_set[bi] = True
            for ti in block_cover[bi]:
                cnt[ti] += 1
                
        # Verify validity (should be valid if input is valid)
        if np.min(cnt) < 1:
             return current_blocks

        best_blocks = list(current_blocks)
        best_cost = len(best_blocks)
        
        rng = random.Random(42)
        
        # Precompute "critical" sets? 
        # A block is "removable" if it doesn't uniquely cover any t-subset.
        # i.e., for all ti in block_cover[bi], cnt[ti] > 1.
        
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline - 5.0:
            improved = False
            iter_count += 1
            
            # 1. Try to remove a block (1-exchange, add 0)
            # Pick a random block to try removing
            if current_blocks:
                idx_drop = rng.randrange(len(current_blocks))
                b_drop = current_blocks[idx_drop]
                
                # Check if removal is safe (no t-subset becomes uncovered)
                safe = True
                for ti in block_cover[b_drop]:
                    if cnt[ti] == 1:
                        safe = False
                        break
                        
                if safe:
                    # Remove it
                    current_blocks.pop(idx_drop)
                    in_set[b_drop] = False
                    for ti in block_cover[b_drop]:
                        cnt[ti] -= 1
                    best_cost -= 1
                    best_blocks = list(current_blocks)
                    improved = True
                    continue
            
            # 2. Try 1-exchange (remove 1, add 1)
            # This is harder. We need to find a block to remove that makes some sets uncovered,
            # and a block to add that covers them, resulting in net 0 change in cost? 
            # Actually, if we remove 1 and add 1, cost is same. We only do this if it helps 
            # future removals? Or if it reduces "redundancy"?
            # Standard local search for set cover often focuses on cost reduction.
            # Since cost is fixed at 1-exchange, we might skip it unless we define a secondary objective (e.g., minimize total coverage count).
            
            # 3. Try 2-exchange (remove 2, add 1) -> Cost -1
            # This is the most powerful move for reduction.
            # Pick 2 blocks to remove.
            # If len(current_blocks) >= 2:
            #    Pick b1, b2.
            #    Calculate U = (Cover(b1) U Cover(b2)) \ (Cover(b1) \cap Cover(b2) covered by others)
            #    Actually, U = { ti | cnt[ti] == 1 and ti in Cover(b1) } U { ti | cnt[ti] == 1 and ti in Cover(b2) }
            #    Wait, if ti is covered by both b1 and b2, and no one else, cnt[ti]=2. Removing both makes cnt[ti]=0.
            #    So U = { ti | cnt[ti] <= 2 and (ti in Cover(b1) or ti in Cover(b2)) and (cnt[ti] - (1 if ti in Cover(b1) else 0) - (1 if ti in Cover(b2) else 0) == 0) }
            
            if len(current_blocks) >= 2:
                # To speed up, we can just pick 2 random blocks
                idx1, idx2 = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[idx1]
                b2 = current_blocks[idx2]
                
                # Calculate uncovered set U if we remove b1 and b2
                # We need to know for each ti in Cover(b1) U Cover(b2), how many other blocks cover it.
                # cnt[ti] is current count.
                # If we remove b1 and b2, new_cnt[ti] = cnt[ti] - (1 if ti in Cover(b1) else 0) - (1 if ti in Cover(b2) else 0)
                # We need new_cnt[ti] >= 1 for all ti.
                
                U = set()
                # Iterate over union of covers
                # Since covers are small (56), we can just iterate
                cover1 = set(block_cover[b1])
                cover2 = set(block_cover[b2])
                union_covers = cover1 | cover2
                
                for ti in union_covers:
                    c1 = 1 if ti in cover1 else 0
                    c2 = 1 if ti in cover2 else 0
                    if cnt[ti] - c1 - c2 == 0:
                        U.add(ti)
                        
                if not U:
                    # Safe to remove both!
                    # Remove b1 and b2
                    # Be careful with indices if we remove one then the other
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    # Remove higher index first to keep lower index valid
                    current_blocks.pop(idx1)
                    in_set[b1] = False
                    for ti in block_cover[b1]:
                        cnt[ti] -= 1
                    current_blocks.pop(idx2)
                    in_set[b2] = False
                    for ti in block_cover[b2]:
                        cnt[ti] -= 1
                    best_cost -= 2
                    best_blocks = list(current_blocks)
                    improved = True
                    continue
                else:
                    # Need to find 1 block to add that covers U
                    # Greedy: find block b_new not in set that covers most of U
                    best_b_new = -1
                    max_cov = -1
                    # Candidates: blocks covering any element in U
                    # To limit search, pick a seed from U
                    seed = next(iter(U))
                    for bi in t_to_blocks[seed]:
                        if in_set[bi]:
                            continue
                        cov = 0
                        for ti in block_cover[bi]:
                            if ti in U:
                                cov += 1
                        if cov > max_cov:
                            max_cov = cov
                            best_b_new = bi
                            
                    if best_b_new != -1 and max_cov == len(U):
                        # Found a block that covers all of U
                        # Apply move: remove b1, b2; add best_b_new
                        # Cost change: -2 + 1 = -1
                        # Update state
                        # Remove b1, b2
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        current_blocks.pop(idx1)
                        in_set[b1] = False
                        for ti in block_cover[b1]:
                            cnt[ti] -= 1
                        current_blocks.pop(idx2)
                        in_set[b2] = False
                        for ti in block_cover[b2]:
                            cnt[ti] -= 1
                        
                        # Add best_b_new
                        current_blocks.append(best_b_new)
                        in_set[best_b_new] = True
                        for ti in block_cover[best_b_new]:
                            cnt[ti] += 1
                            
                        best_cost -= 1
                        best_blocks = list(current_blocks)
                        improved = True
                        continue
                        
        return best_blocks

    # Main Loop: Iterated Local Search
    best_overall_blocks = []
    best_overall_cost = float('inf')
    
    rng = random.Random(42)
    
    # Initial solution
    init_blocks = greedy_cover()
    init_blocks = local_search(init_blocks)
    best_overall_blocks = list(init_blocks)
    best_overall_cost = len(best_overall_blocks)
    
    # ILS Loop
    while time.time() < deadline - 10.0:
        # Perturb the current best solution
        # Strategy: Randomly swap a few blocks
        perturbed_blocks = list(best_overall_blocks)
        
        # Number of swaps: proportional to size? Or fixed?
        # Let's try swapping 2-5 blocks
        num_swaps = rng.randint(2, 5)
        
        for _ in range(num_swaps):
            if len(perturbed_blocks) < 2:
                break
            # Pick a block to remove
            idx_rm = rng.randrange(len(perturbed_blocks))
            b_rm = perturbed_blocks[idx_rm]
            perturbed_blocks.pop(idx_rm)
            
            # Pick a random new block to add
            # To ensure validity, we might need to check, but ILS assumes we repair later.
            # However, if we just add a random block, the solution might become invalid.
            # local_search assumes a valid input.
            # So we must ensure the perturbed solution is valid, or use a repair step.
            # Better perturbation: Swap a block with another block that covers its unique parts?
            # That's complex.
            
            # Alternative: "Kick" by removing a block and adding a random one, then running a fast repair (greedy on uncovered).
            # Let's try: Remove b_rm. Find uncovered U. Add greedy blocks to cover U.
            # This is essentially a "destroy and repair" heuristic.
            
            # Let's implement a simple repair for the perturbed state.
            # But local_search already does improvement.
            # If we pass an invalid solution to local_search, it might fail or produce garbage.
            # So, let's make the perturbation valid.
            
            # Valid Perturbation:
            # 1. Pick a block b_rm.
            # 2. Identify U = subsets uniquely covered by b_rm.
            # 3. Pick a random block b_add that covers at least one element of U (or is random).
            # 4. If b_add doesn't cover all U, the solution is invalid.
            #    We need to add more blocks to cover the rest.
            #    This is getting complicated.
            
            # Simpler Valid Perturbation:
            # Swap two blocks? No, that keeps size same.
            # Let's try: Remove 1 block, add 1 block, such that coverage is maintained?
            # Hard to guarantee.
            
            # Let's go back to: Destroy (remove 1 block) and Repair (greedy add).
            # This might increase cost, but then local_search (which only reduces cost) will try to bring it back down,
            # potentially finding a different local minimum.
            
            # So:
            # 1. Remove b_rm from perturbed_blocks.
            # 2. Calculate U (uncovered).
            # 3. While U not empty:
            #       Pick best greedy block to cover U.
            #       Add it.
            # 4. Now perturbed_blocks is valid (and likely larger).
            # 5. Run local_search to reduce it.
            
            # Let's implement this repair inside the loop.
            
            # We need counts to do this fast.
            # This is expensive to recompute every time.
            # Given the time limit, maybe we should just do simple swaps that preserve validity?
            
            # Let's try a different perturbation: "Random Walk" in the space of valid solutions.
            # Move: Remove b1, Add b2.
            # Valid if: For all ti in Cover(b1), cnt[ti] - 1 + (1 if ti in Cover(b2) else 0) >= 1.
            # i.e., cnt[ti] + (1 if ti in Cover(b2) else 0) >= 2 for all ti in Cover(b1).
            
            # We can try this.
            
            found_valid_swap = False
            attempts = 0
            while attempts < 50 and not found_valid_swap:
                attempts += 1
                # Pick random block to remove
                if not perturbed_blocks:
                    break
                idx_rm = rng.randrange(len(perturbed_blocks))
                b_rm = perturbed_blocks[idx_rm]
                
                # Pick random block to add
                b_add = rng.randrange(B)
                if b_add in perturbed_blocks: # Check membership is O(N), use set?
                    continue
                # Check if b_add is already in set (using a set for membership is faster)
                # Let's maintain a set of current blocks for perturbation check?
                # Or just check list. List is small (~300).
                
                # Check validity
                valid = True
                # We need counts. We don't have them here.
                # This approach is too slow without counts.
                
                break # Abandon complex perturbation for now, stick to simple restarts or simpler kicks.
            
            # Let's just do a "Soft Restart": 
            # Take the current best, remove a random 10% of blocks, and repair greedily.
            # This is a common ILS technique.
            
            break # Exit inner loop, do the soft restart logic below.

        # Soft Restart Logic
        # 1. Take best_overall_blocks
        # 2. Remove random 10-20% of blocks
        # 3. Repair to valid using Greedy
        # 4. Local Search
        
        if len(best_overall_blocks) == 0:
            break
            
        remove_frac = rng.uniform(0.1, 0.2)
        num_remove = max(1, int(len(best_overall_blocks) * remove_frac))
        
        perturbed = list(best_overall_blocks)
        # Remove random blocks
        indices_to_remove = rng.sample(range(len(perturbed)), num_remove)
        for i in sorted(indices_to_remove, reverse=True):
            perturbed.pop(i)
            
        # Repair
        # We need to know what's uncovered.
        # Rebuild counts?
        # Since perturbed is small, and N is 11k, we can rebuild counts in O(|perturbed| * 56)
        cnt = np.zeros(N, dtype=np.int32)
        for bi in perturbed:
            for ti in block_cover[bi]:
                cnt[ti] += 1
                
        uncovered = set(np.where(cnt == 0)[0])
        
        # Greedy repair
        while uncovered and time.time() < deadline - 5.0:
            seed = next(iter(uncovered))
            best_b = -1
            max_cov = -1
            for bi in t_to_blocks[seed]:
                if bi in perturbed: # O(N) check, but perturbed is small
                    continue
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
            if best_b == -1:
                break
            perturbed.append(best_b)
            for ti in block_cover[best_b]:
                if ti in uncovered:
                    uncovered.discard(ti)
                    
        # Local Search on the repaired perturbed solution
        improved_perturbed = local_search(perturbed)
        
        if len(improved_perturbed) < best_overall_cost:
            best_overall_cost = len(improved_perturbed)
            best_overall_blocks = improved_perturbed

    # Final Prune (just in case)
    def prune_redundant(blocks):
        if not blocks:
            return blocks
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in blocks:
                essential = False
                for ti in block_cover[bi]:
                    if cnt[ti] == 1:
                        essential = True
                        break
                if not essential:
                    changed = True
                    for ti in block_cover[bi]:
                        cnt[ti] -= 1
                else:
                    new_blocks.append(bi)
            blocks = new_blocks
            if time.time() > deadline - 2.0:
                break
        return blocks

    best_overall_blocks = prune_redundant(best_overall_blocks)
    
    # Verify
    cnt_final = np.zeros(N, dtype=np.int32)
    for bi in best_overall_blocks:
        for ti in block_cover[bi]:
            cnt_final[ti] += 1
            
    if np.min(cnt_final) < 1:
        # Fallback
        return [list(all_blocks[bi]) for bi in greedy_cover()]
        
    return [list(all_blocks[bi]) for bi in best_overall_blocks]
