import itertools
import time
import numpy as np

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Precompute the incidence matrix A: A[j, i] = 1 if block j covers tsub i
    # A is (num_kblocks, num_tsubs)
    A = np.zeros((num_kblocks, num_tsubs), dtype=np.int8)
    for j, block in enumerate(all_kblocks):
        for ts in itertools.combinations(block, t):
            A[j, tsub_to_idx[ts]] = 1
    
    # For each block, precompute the list of tsub indices it covers (for fast updates)
    block_tsub_idxs = []
    for j in range(num_kblocks):
        idxs = np.where(A[j] == 1)[0].tolist()
        block_tsub_idxs.append(idxs)
    
    def polish(selected):
        """Remove redundant blocks. Returns a list of block indices."""
        if not selected:
            return selected
        
        # Count coverage for each tsub
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in selected:
            counts[block_tsub_idxs[b_idx]] += 1
        
        changed = True
        while changed and time.monotonic() < deadline - 5.0:
            changed = False
            new_selected = []
            for b_idx in selected:
                # Check if any tsub in this block has count == 1
                can_remove = True
                for idx in block_tsub_idxs[b_idx]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    counts[block_tsub_idxs[b_idx]] -= 1
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected
        return selected

    def greedy_construct():
        """Standard greedy set cover with vectorized gain computation."""
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        
        while time.monotonic() < deadline - 10.0:
            if covered.all():
                break
            
            # Compute gains: for each block, count how many uncovered tsubs it covers
            # gain[j] = sum(A[j, i] for i where not covered[i])
            # We can do: gain = A @ (1 - covered)
            # But A is int8, (1-covered) is bool. Let's use A @ (~covered)
            # Actually, A is 0/1, and we want A[j] dot (not covered)
            # gain = A.dot(~covered)  -- this works if ~covered is treated as 0/1
            
            # Convert ~covered to int for dot product
            uncovered_mask = (~covered).astype(np.int8)
            gains = A.dot(uncovered_mask)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            best_block = int(np.argmax(gains))
            selected.append(best_block)
            
            # Update covered
            covered[block_tsub_idxs[best_block]] = True
        
        return selected

    def local_search(selected, max_iters=200):
        """
        Local search: try 1-for-1 swaps to enable deletions.
        For each block B1 in the set, compute U = tsubs only covered by B1.
        Find a block B2 that covers all of U. If found, swap B1 -> B2.
        Then try to delete.
        """
        best_selected = polish(selected[:])
        best_len = len(best_selected)
        
        if best_len == 0:
            return best_selected
        
        current_selected = best_selected[:]
        no_improve = 0
        max_no_improve = 30
        
        # Precompute for each block, the set of tsubs it covers as a frozenset for fast intersection
        # Actually, we can use the counts approach.
        
        while time.monotonic() < deadline - 2.0 and no_improve < max_no_improve:
            # Compute coverage counts
            counts = np.zeros(num_tsubs, dtype=np.int32)
            for b_idx in current_selected:
                counts[block_tsub_idxs[b_idx]] += 1
            
            improved = False
            # Try to find a swap that enables a deletion
            # For each block B1, U = {i : counts[i] == 1 and i in block_tsub_idxs[B1]}
            # If U is empty, B1 can be deleted (handled by polish).
            # If U is non-empty, find B2 such that U subset of block_tsub_idxs[B2].
            
            # To make this faster, we can iterate over blocks and check.
            # But this is O(num_selected * num_kblocks * |U|) which is expensive.
            
            # Alternative: For each tsub i with counts[i] == 1, find all blocks covering i.
            # Then for each such block B1, U is the set of such tsubs in B1.
            # This is complex. Let's just do a simple random swap attempt.
            
            # Randomly pick a block to remove
            if not current_selected:
                break
            
            rem_idx = np.random.randint(len(current_selected))
            rem_block = current_selected[rem_idx]
            
            # Find tsubs only covered by rem_block
            U = []
            for idx in block_tsub_idxs[rem_block]:
                if counts[idx] == 1:
                    U.append(idx)
            
            if not U:
                # Can delete directly
                current_selected.pop(rem_idx)
                improved = True
                break
            
            # Find a block B2 that covers all of U
            # U is a list of tsub indices. We need a block that contains all of them.
            # This is equivalent to: intersection of (blocks covering each u in U) is non-empty.
            # Or: A[:, U] has a row that is all 1s.
            
            # Let's check all blocks.
            # For speed, we can precompute for each tsub, the list of blocks covering it.
            # But we don't have that. Let's just scan.
            
            # Actually, we can do: candidate_blocks = [j for j in range(num_kblocks) if all(A[j, u] for u in U)]
            # This is slow. Let's use numpy.
            # A[:, U] is a matrix. We want rows where all are 1.
            # np.all(A[:, U], axis=1)
            
            # But U might be large. Let's limit U size for speed? No, U is at most 35.
            # Let's just do it.
            
            if len(U) > 10: # Too many constraints, unlikely to find a block
                continue
            
            # Find blocks covering all of U
            # We can use the fact that if a block covers all of U, it must cover the first element of U.
            # So we can filter by blocks covering U[0].
            
            # But we don't have tsub_to_blocks. Let's build it once?
            # No, let's just scan all blocks. It's 50k blocks, and we check 10 elements each. 500k ops. Fine.
            
            found_swap = False
            for j in range(num_kblocks):
                if j == rem_block:
                    continue
                # Check if j covers all of U
                ok = True
                for u in U:
                    if A[j, u] == 0:
                        ok = False
                        break
                if ok:
                    # Swap rem_block with j
                    current_selected[rem_idx] = j
                    # Check if we can now delete j or rem_block?
                    # Actually, we just swapped. Now let's see if we can delete the new block or any other.
                    # But first, let's just accept the swap and let polish handle deletions.
                    # However, the swap itself doesn't reduce count.
                    # We need the swap to enable a deletion.
                    # So after swap, we should run polish.
                    
                    # Let's just do the swap and then polish.
                    candidate = polish(current_selected[:])
                    if len(candidate) < best_len:
                        best_selected = candidate
                        best_len = len(candidate)
                        current_selected = candidate[:]
                        improved = True
                        break
                    else:
                        # Revert swap
                        current_selected[rem_idx] = rem_block
                        # Try next block? Or break?
                        # Let's break to save time and try another random rem_block.
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
        
        return best_selected

    # Main
    # 1. Initial Greedy
    initial = greedy_construct()
    initial = polish(initial)
    
    # 2. Local Search
    best = local_search(initial)
    
    # 3. Restarts
    # Use randomized greedy: pick from top K gains
    def greedy_construct_topk(K=5):
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        
        while time.monotonic() < deadline - 10.0:
            if covered.all():
                break
            
            uncovered_mask = (~covered).astype(np.int8)
            gains = A.dot(uncovered_mask)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find blocks with gain >= max_gain - 1 (or top K)
            threshold = max_gain - 1
            candidates = np.where(gains >= threshold)[0]
            
            if len(candidates) == 0:
                candidates = np.where(gains == max_gain)[0]
            
            best_block = int(candidates[np.random.randint(len(candidates))])
            selected.append(best_block)
            covered[block_tsub_idxs[best_block]] = True
        
        return selected

    num_restarts = 10
    for i in range(num_restarts):
        if time.monotonic() > deadline - 20.0:
            break
        candidate = greedy_construct_topk(5)
        candidate = polish(candidate)
        candidate = local_search(candidate, max_iters=100)
        
        if len(candidate) < len(best):
            best = candidate
    
    # Final polish
    best = polish(best)
    
    result_blocks = [list(all_kblocks[i]) for i in best]
    
    return result_blocks
