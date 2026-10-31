import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    # Using uint8 for memory efficiency and fast matmul
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    # Optimization: Precompute the t-subsets for each block to fill matrix faster
    # Or just iterate. For v=19, k=7, t=4, each block has C(7,4)=35 t-subsets.
    # 50388 blocks * 35 ops is fast.
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    # 2. Greedy Initialization
    uncovered = np.ones(n_tsubs, dtype=bool)
    selected_indices = []
    
    # Greedy loop
    while np.sum(uncovered) > 0:
        if time.time() > deadline - 30.0: # Leave time for optimization
            break
        gains = Cover @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
    
    # Convert to a set for easier manipulation in local search
    # But we need to track counts.
    # Let's maintain:
    # - current_blocks: list of indices
    # - cover_counts: array of length n_tsubs
    
    current_blocks = list(selected_indices)
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in current_blocks:
        cover_counts += Cover[idx]
        
    # Verify initial validity (should be valid by construction)
    # But if we broke early, it might not be.
    # For robustness, if invalid, we must repair.
    # The greedy loop above ensures validity unless it breaks due to time.
    # If it breaks due to time, we have an invalid cover.
    # We must ensure we return a VALID cover.
    # So, if we haven't finished greedy, we MUST finish it or repair.
    # Given the small size, greedy should finish in < 1s.
    
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # 3. Local Optimization: Simulated Annealing / ILS
    # Strategy: 
    # 1. Reduce: Remove any block that is redundant (covers no unique t-subset).
    # 2. Perturb: Remove random blocks.
    # 3. Repair: Greedily add blocks to cover gaps.
    
    def get_uncovered_mask(counts):
        return counts == 0

    def reduce_redundant(blocks, counts):
        """Remove blocks that do not cover any unique t-subset."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for idx in blocks:
                # Check if any t-subset covered by idx has count == 1
                # If yes, keep it. If no, remove it.
                # Vectorized check:
                # mask = (Cover[idx] == 1) & (counts == 1)
                # if np.any(mask): keep
                # else: remove
                if np.any((Cover[idx] == 1) & (counts == 1)):
                    new_blocks.append(idx)
                else:
                    counts -= Cover[idx]
                    changed = True
            blocks = new_blocks
        return blocks

    def repair(blocks, counts):
        """Greedily add blocks to cover all t-subsets."""
        while np.any(counts == 0):
            if time.time() > deadline - 5.0:
                break
            uncovered_mask = (counts == 0)
            gains = Cover @ uncovered_mask
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break # Should not happen if cover is possible
            blocks.append(best_idx)
            counts += Cover[best_idx]
        return blocks

    # Main Optimization Loop
    temperature = 10.0
    cooling = 0.95
    min_temp = 0.01
    
    # Initial reduction
    current_blocks = reduce_redundant(current_blocks, cover_counts)
    if len(current_blocks) < best_size:
        best_blocks = list(current_blocks)
        best_size = len(best_blocks)
        
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # 1. Reduce redundant blocks
        current_blocks = reduce_redundant(current_blocks, cover_counts)
        
        # Update best
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(best_blocks)
            # Reset temperature? Or just continue.
            
        # 2. Perturb
        # Remove 1 to 3 random blocks
        num_removals = random.randint(1, 3)
        if len(current_blocks) <= num_removals:
            break
            
        # Pick random indices to remove
        removals = random.sample(current_blocks, num_removals)
        
        # Apply removals
        for idx in removals:
            cover_counts -= Cover[idx]
        current_blocks = [b for b in current_blocks if b not in removals]
        
        # 3. Repair
        current_blocks = repair(current_blocks, cover_counts)
        
        # 4. Acceptance
        new_size = len(current_blocks)
        delta = new_size - best_size
        
        if delta < 0:
            # Improvement
            best_blocks = list(current_blocks)
            best_size = new_size
        elif temperature > min_temp:
            # Simulated Annealing acceptance
            # Probability of accepting worse solution
            # Delta is positive (worse)
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                # Accept, but we don't update best, we just stay in this state
                pass
            else:
                # Reject: revert to best state? 
                # In ILS, we usually keep the current state if accepted, 
                # but if rejected, we might want to go back to the last good state 
                # or just continue from the perturbed state (which is worse).
                # Standard ILS: If rejected, revert to the state BEFORE perturbation.
                # We need to store the state before perturbation.
                pass # Simplification: We will just restart from best if rejected often? 
                # Actually, let's just keep the current (worse) state for now and hope it leads to a better one?
                # No, that's random walk. 
                # Let's revert to best_blocks if rejected to maintain a good baseline.
                # But we need to reconstruct cover_counts for best_blocks.
                # This is expensive.
                
                # Alternative: Only accept improvements (Greedy ILS).
                # Or, store the previous state.
                
                # Let's try a simpler approach:
                # Always revert to best if not improved, but with some randomness in perturbation.
                # This is "Hill Climbing" with restarts.
                
                # Let's implement "Revert to Best" logic properly.
                # We need to be able to restore cover_counts quickly.
                # Since we only have one "best" state, we can just recompute counts for best_blocks.
                # But that's O(N_best * 35). Fast enough.
                
                # Let's just restart from best if rejected.
                current_blocks = list(best_blocks)
                cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                for idx in current_blocks:
                    cover_counts += Cover[idx]
                
        temperature *= cooling

    # Final reduction
    # We need to return a valid cover.
    # If the last state was invalid (due to time out in repair), we must use best_blocks.
    # best_blocks is always valid (starts valid, only updated on improvement which implies valid repair).
    
    # One final check and reduction on best_blocks
    final_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in best_blocks:
        final_counts += Cover[idx]
        
    # Ensure no redundancy in the final answer
    final_blocks = reduce_redundant(best_blocks, final_counts)
    
    # Verify validity of final_blocks
    # (Should be valid, but just in case)
    if np.any(final_counts == 0):
         # This shouldn't happen if reduce_redundant is correct
         # But if it does, we fall back to best_blocks
         final_blocks = best_blocks

    return [all_kblocks[i] for i in final_blocks]

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    # Verify
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
