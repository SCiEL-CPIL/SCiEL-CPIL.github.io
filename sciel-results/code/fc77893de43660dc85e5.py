import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Data structures
    # block_tsubs[b] = list of t-subset indices covered by block b
    block_tsubs = [[] for _ in range(n_blocks)]
    # tsub_blocks[ts] = list of block indices covering t-subset ts
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Bitmasks for fast operations in greedy
    tsub_mask = [0] * n_blocks       # tsub_mask[b] has bits for tsubs covered by block b
    tsub_blocks_mask = [0] * n_tsubsets # tsub_blocks_mask[ts] has bits for blocks covering ts

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
            tsub_blocks[idx].append(b_idx)
        tsub_mask[b_idx] = mask

    full_tsub_mask = (1 << n_tsubsets) - 1

    def greedy_cover(seed):
        """
        Standard greedy using heap for gain calculation.
        Randomizes tie-breaking.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0
        
        # Initial gains: number of tsubs covered by each block
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = tsub_mask[b_idx].bit_count()
            
        # Heap entries: (-gain, random_tie_breaker, b_idx)
        # We use a random tie breaker to ensure different runs
        heap = []
        for b_idx in range(n_blocks):
            heap.append((-gains[b_idx], rng.random(), b_idx))
        heapq.heapify(heap)
        
        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break

            best_block = -1
            best_gain = 0

            # Pop stale entries
            while heap:
                neg_g, _, b_idx = heapq.heappop(heap)
                g = -neg_g
                # Check if this entry is still valid
                # 1. Is the block already selected?
                if selected_mask & (1 << b_idx):
                    continue
                # 2. Is the gain still correct?
                # We need to verify the gain against current uncovered_mask
                # This is expensive. 
                # Alternative: Lazy deletion is tricky if gains change.
                # The inspiration code updated gains explicitly and pushed new entries.
                # Let's stick to the inspiration's explicit update method which was faster.
                
                # Re-implementing the explicit update method from inspiration 1
                # It maintained a `gains` array and pushed updated gains to heap.
                # Let's restart the greedy implementation to match that pattern.
                break 
            # If we broke out of the inner while, we need to restart the greedy logic 
            # or I should just copy the working logic from inspiration 1.
            pass

    # Let's re-write greedy_cover properly based on Inspiration 1's logic which was measured to be fast.
    def greedy_cover_v2(seed):
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0
        
        # Initial gains
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = tsub_mask[b_idx].bit_count()
            
        heap = [(-gains[b_idx], b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)
        
        # To handle ties, we can just rely on the heap order. 
        # If we want randomization, we can perturb initial gains slightly or 
        # use a different selection method.
        # For now, deterministic greedy is a good baseline.
        # We will run multiple seeds by permuting the block order in the heap initialization?
        # Or just run the same greedy, it's deterministic.
        # To get different solutions, we can randomize the tie-breaking.
        
        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break

            best_block = -1
            best_gain = 0
            best_tie = 0 # Not used in this simple version, but we can add it
            
            # Pop stale entries
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b_idx] and not (selected_mask & (1 << b_idx)):
                    best_block = b_idx
                    best_gain = g
                    break
                # Stale or already selected, discard

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1 # Mark as selected
            
            # Update gains for blocks covering newly covered tsubs
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]
            
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                # For every block covering this ts_idx, decrease its gain
                cm = tsub_blocks_mask[ts_idx]
                while cm:
                    lsb2 = cm & -cm
                    b_idx2 = lsb2.bit_length() - 1
                    if not (selected_mask & (1 << b_idx2)):
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], b_idx2))
                    cm ^= lsb2
                m ^= lsb

        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        # cover_count[ts] = number of selected blocks covering ts
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in list(block_set):
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    # This shouldn't happen if Phase 1 is exhaustive, 
                    # but if it does, remove it.
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidates using set intersection
                # Start with the most constrained unique tsub
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(tsub_blocks[ts_idx])

                if not candidates:
                    continue

                candidates.discard(b)
                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                # Limit candidates to avoid too much work per swap attempt
                # But since we need to find *any* valid swap, we should check all?
                # Checking all might be slow if candidates are many.
                # However, valid candidates must cover ALL unique tsubs.
                # The intersection usually reduces this set significantly.
                
                swapped = False
                for r in cand_list:
                    if r in block_set:
                        continue

                    # Perform swap
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break # Restart the loop to check for new redundancies

        return list(block_set)

    best_blocks = None
    best_size = float('inf')
    
    # Run greedy + local search with a few seeds
    # The greedy is deterministic, so seed only affects local search randomization.
    # To get different greedy results, we would need to randomize the greedy itself.
    # Let's just run local search on the single greedy solution with different seeds to find different local optima?
    # No, local search on the same start might converge to the same place.
    # We should try to randomize the greedy.
    
    # Let's modify greedy to accept a seed for tie-breaking.
    def greedy_cover_random(seed):
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0
        
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = tsub_mask[b_idx].bit_count()
            
        # Heap: (-gain, tie_breaker, b_idx)
        heap = []
        for b_idx in range(n_blocks):
            heap.append((-gains[b_idx], rng.random(), b_idx))
        heapq.heapify(heap)
        
        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break

            best_block = -1
            best_gain = 0

            while heap:
                neg_g, _, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b_idx] and not (selected_mask & (1 << b_idx)):
                    best_block = b_idx
                    best_gain = g
                    break

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1
            
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]
            
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                cm = tsub_blocks_mask[ts_idx]
                while cm:
                    lsb2 = cm & -cm
                    b_idx2 = lsb2.bit_length() - 1
                    if not (selected_mask & (1 << b_idx2)):
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], rng.random(), b_idx2))
                    cm ^= lsb2
                m ^= lsb

        return selected

    # Run multiple iterations
    for seed in range(10):
        if time.time() > deadline - 20.0:
            break
            
        greedy_res = greedy_cover_random(seed)
        ls_res = local_search(greedy_res, seed)
        
        if len(ls_res) < best_size:
            best_size = len(ls_res)
            best_blocks = ls_res
            
        # If we find a very good solution, we might stop early?
        # The known best is 151. If we get close, we should keep trying.
        
    if best_blocks is None:
        # Fallback to simple greedy if something went wrong
        return [list(all_blocks[i]) for i in greedy_cover_v2(0)]
        
    return [list(all_blocks[i]) for i in best_blocks]
