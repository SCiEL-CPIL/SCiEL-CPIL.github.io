import itertools
import time
import random
import heapq
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 140.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build block_tsubs: for each block, list of t-subset indices it covers
    block_tsubs = [[] for _ in range(n_blocks)]
    for bi, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            block_tsubs[bi].append(tsub_idx[ts])
    
    # Build tsub_blocks: for each t-subset, list of blocks covering it
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
    
    # Precompute as numpy arrays
    block_tsubs_arr = [np.array(ts_list, dtype=np.int32) for ts_list in block_tsubs]
    tsub_blocks_arr = [np.array(bi_list, dtype=np.int32) for bi_list in tsub_blocks]
    tsub_blocks_set = [set(bi_list) for bi_list in tsub_blocks]
    
    best_blocks = None
    best_indices = None
    best_len = float('inf')
    
    def primal_dual_greedy(seed=0):
        """
        Primal-dual approximation for set cover.
        Simultaneously increases dual variables for uncovered t-subsets.
        A block is selected when its cost equals the sum of dual variables
        of uncovered t-subsets it covers (within the block's coverage).
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        dual = np.zeros(n_tsubsets, dtype=np.float64)
        blocks_used = []
        selected = np.zeros(n_blocks, dtype=bool)
        
        # Heap entries: (effective_cost, block_index)
        # effective_cost = cost - sum(dual[j] for j in uncovered tsubsets covered by block)
        # We use a lazy heap approach
        
        # Initial: all blocks have cost 1 (unit cost)
        # We'll track for each block the sum of duals of its uncovered t-subsets
        
        # For efficiency, maintain for each block the count of uncovered t-subsets it covers
        # and use that as a proxy for dual sum (since all duals increase at same rate)
        # Actually in primal-dual, duals increase at rate 1 for all uncovered elements,
        # so dual[j] = time when j becomes covered (or current time if still uncovered)
        
        # Simpler approach: track for each block how many of its t-subsets are still uncovered
        # The "primal-dual" criterion: select block when cost >= sum of duals of its uncovered elements
        # Since all duals increase uniformly, this is equivalent to selecting when
        # cost >= (number of uncovered elements in block) * current_rate... 
        # Actually the standard primal-dual for unweighted set cover:
        # - Increase all duals for uncovered elements at rate 1
        # - When a block's cost (1) equals the sum of duals of its uncovered elements, select it
        # - Since duals all increase at same rate, the sum for a block with u uncovered elements
        #   is u * t (where t is the "time" since start), but duals are not all equal...
        
        # Let me implement a cleaner version:
        # Track dual[j] for each t-subset j. Initially 0.
        # At each step, find the minimum rate at which some block becomes "tight".
        # A block b is tight when sum(dual[j] for j in block_tsubs[b] if uncovered[j]) >= cost[b] = 1
        # The rate of increase for dual[j] is 1 if uncovered[j], 0 otherwise.
        # The rate of increase for the sum for block b is (number of uncovered t-subsets in b).
        # So the time until block b becomes tight is (1 - current_sum) / rate_b if rate_b > 0.
        
        # This is complex. Let me use a simpler but effective variant:
        # Weighted greedy where weights are based on LP dual estimates.
        
        # Actually, let me implement the standard primal-dual properly but efficiently:
        # We maintain a priority queue of (time_until_tight, block_index).
        # When we select a block, we update the queue for all blocks sharing its t-subsets.
        
        # For this problem size, let me use a simpler effective approach:
        # Greedy with LP-inspired weighting.
        
        # Compute initial weights: for each t-subset, weight = 1/frequency
        # This approximates the LP dual
        freq = np.array([len(tsub_blocks[i]) for i in range(n_tsubsets)], dtype=np.float64)
        dual_weights = 1.0 / freq  # LP dual approximation
        
        # Weighted greedy: at each step, select block maximizing 
        # sum(dual_weights[j] for j in uncovered t-subsets covered by block)
        # This is equivalent to minimizing cost/weight ratio
        
        # Precompute weighted coverage for each block
        # gain[b] = sum of dual_weights for uncovered t-subsets in block b
        gain = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            gain[bi] = np.sum(dual_weights[block_tsubs_arr[bi]])
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            max_gain = gain.max()
            if max_gain <= 1e-12:
                break
            
            max_blocks = np.where(gain >= max_gain - 1e-12)[0]
            chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
            
            blocks_used.append(chosen)
            selected[chosen] = True
            
            # Update gains
            chosen_tsubs = block_tsubs_arr[chosen]
            for ts_idx in chosen_tsubs:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrease gain for all blocks covering this t-subset
                    blocks_covering = tsub_blocks_arr[ts_idx]
                    gain[blocks_covering] -= dual_weights[ts_idx]
            
            gain[chosen] = 0.0
        
        return blocks_used
    
    def standard_greedy(seed=0):
        """Standard greedy set cover with random tie-breaking."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        gain = np.array([len(ts_list) for ts_list in block_tsubs], dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            max_gain = gain.max()
            if max_gain == 0:
                break
            
            max_blocks = np.where(gain == max_gain)[0]
            chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
            
            blocks_used.append(chosen)
            
            chosen_tsubs = block_tsubs_arr[chosen]
            for ts_idx in chosen_tsubs:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    blocks_covering = tsub_blocks_arr[ts_idx]
                    mask = gain[blocks_covering] > 0
                    if mask.any():
                        gain[blocks_covering[mask]] -= 1
            
            gain[chosen] = -1
        
        return blocks_used
    
    def remove_redundant_blocks(block_indices):
        """Remove redundant blocks from the solution."""
        if not block_indices:
            return block_indices
        
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in block_indices:
            cover_count[block_tsubs_arr[bi]] += 1
                
        kept = [True] * len(block_indices)
        changed = True
        while changed:
            changed = False
            for i, bi in enumerate(block_indices):
                if not kept[i]:
                    continue
                redundant = True
                for ts_idx in block_tsubs_arr[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    cover_count[block_tsubs_arr[bi]] -= 1
                    changed = True
                    
        return [block_indices[i] for i in range(len(block_indices)) if kept[i]]
    
    def local_search(block_indices, seed=0, max_swap_rounds=30):
        """Local search with 1-for-1 swaps."""
        if not block_indices:
            return block_indices
        
        current_blocks = list(block_indices)
        rng = random.Random(seed)
        
        for round_num in range(max_swap_rounds):
            if time.time() > deadline - 15:
                break
            
            current_blocks = remove_redundant_blocks(current_blocks)
            
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            for bi in current_blocks:
                cover_count[block_tsubs_arr[bi]] += 1
            
            current_set = set(current_blocks)
            
            improved = False
            indices_order = list(range(len(current_blocks)))
            rng.shuffle(indices_order)
            
            for i in indices_order:
                if time.time() > deadline - 15:
                    break
                
                bi = current_blocks[i]
                bi_tsubs = block_tsubs_arr[bi]
                unique_tsubs = bi_tsubs[cover_count[bi_tsubs] == 1]
                
                if len(unique_tsubs) == 0:
                    continue
                
                # Sort unique t-subsets by rarity
                unique_tsubs_sorted = unique_tsubs[np.argsort([len(tsub_blocks[ts]) for ts in unique_tsubs])]
                
                candidates = tsub_blocks_set[unique_tsubs_sorted[0]].copy()
                for ts_idx in unique_tsubs_sorted[1:]:
                    candidates &= tsub_blocks_set[ts_idx]
                    if not candidates:
                        break
                
                candidates -= current_set
                
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for new_bi in cand_list[:8]:
                    current_blocks[i] = new_bi
                    current_set.discard(bi)
                    current_set.add(new_bi)
                    improved = True
                    break
                
                if improved:
                    break
            
            if not improved:
                break
        
        current_blocks = remove_redundant_blocks(current_blocks)
        return current_blocks
    
    def two_exchange_search(block_indices, seed=0, max_rounds=10):
        """Try 2-for-1 exchanges: remove two blocks, add one that covers all their unique t-subsets."""
        if not block_indices:
            return block_indices
        
        current_blocks = list(block_indices)
        rng = random.Random(seed)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 15:
                break
            
            current_blocks = remove_redundant_blocks(current_blocks)
            
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            for bi in current_blocks:
                cover_count[block_tsubs_arr[bi]] += 1
            
            current_set = set(current_blocks)
            improved = False
            
            # Try random pairs
            n = len(current_blocks)
            pairs_tried = 0
            max_pairs = min(200, n * (n - 1) // 2)
            
            while pairs_tried < max_pairs:
                if time.time() > deadline - 15:
                    break
                i, j = rng.sample(range(n), 2)
                pairs_tried += 1
                
                bi, bj = current_blocks[i], current_blocks[j]
                
                # Find t-subsets uniquely covered by bi or bj
                bi_tsubs = block_tsubs_arr[bi]
                bj_tsubs = block_tsubs_arr[bj]
                
                # t-subsets that would become uncovered if we remove both bi and bj
                # These are t-subsets with cover_count == 1 that are in bi or bj
                # Actually, t-subsets with cover_count == 1 in bi, or cover_count == 2 where one is bi and one is bj
                
                unique_to_pair = set()
                for ts_idx in bi_tsubs:
                    if cover_count[ts_idx] == 1:
                        unique_to_pair.add(ts_idx)
                    elif cover_count[ts_idx] == 2:
                        # Check if bj also covers it
                        if ts_idx in set(bj_tsubs):
                            unique_to_pair.add(ts_idx)
                
                for ts_idx in bj_tsubs:
                    if cover_count[ts_idx] == 1:
                        unique_to_pair.add(ts_idx)
                
                if not unique_to_pair:
                    continue
                
                # Find a block covering all t-subsets in unique_to_pair
                # Start with the rarest
                unique_list = list(unique_to_pair)
                unique_list.sort(key=lambda x: len(tsub_blocks[x]))
                
                candidates = tsub_blocks_set[unique_list[0]].copy()
                for ts_idx in unique_list[1:]:
                    candidates &= tsub_blocks_set[ts_idx]
                    if not candidates:
                        break
                
                candidates -= current_set
                
                if candidates:
                    # Perform the 2-for-1 exchange
                    new_bi = rng.choice(list(candidates))
                    current_blocks[i] = new_bi
                    current_blocks.pop(j)
                    current_set.discard(bi)
                    current_set.discard(bj)
                    current_set.add(new_bi)
                    improved = True
                    break
            
            if not improved:
                break
        
        current_blocks = remove_redundant_blocks(current_blocks)
        return current_blocks
    
    # Main search loop: alternate between primal-dual and standard greedy
    num_trials = 60
    for seed in range(num_trials):
        if time.time() > deadline - 30:
            break
        
        # Alternate between methods
        if seed % 2 == 0:
            blocks_idx = primal_dual_greedy(seed)
        else:
            blocks_idx = standard_greedy(seed)
        
        if not blocks_idx:
            continue
        
        # Apply local search
        blocks_idx = local_search(blocks_idx, seed, max_swap_rounds=25)
        
        # Try 2-exchange
        if time.time() < deadline - 20:
            blocks_idx = two_exchange_search(blocks_idx, seed + 500, max_rounds=5)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_indices = list(blocks_idx)
            best_blocks = [all_blocks[i] for i in best_indices]
            
            if best_len <= 145 and time.time() < deadline - 25:
                polished = local_search(best_indices, seed + 2000, max_swap_rounds=40)
                if len(polished) < best_len:
                    best_len = len(polished)
                    best_indices = list(polished)
                    best_blocks = [all_blocks[i] for i in best_indices]
                
                if time.time() < deadline - 20:
                    polished2 = two_exchange_search(best_indices, seed + 3000, max_rounds=8)
                    if len(polished2) < best_len:
                        best_len = len(polished2)
                        best_indices = list(polished2)
                        best_blocks = [all_blocks[i] for i in best_indices]
            
        if best_len <= 142:
            break
    
    # Try MILP with warm start if we have time
    if best_indices and time.time() < deadline - 25:
        # Build sparse matrix
        row_indices = []
        col_indices = []
        for bi, tsubs in enumerate(block_tsubs):
            for ts_idx in tsubs:
                row_indices.append(ts_idx)
                col_indices.append(bi)
        
        data = np.ones(len(row_indices), dtype=np.float64)
        A = csr_matrix((data, (row_indices, col_indices)), shape=(n_tsubsets, n_blocks))
        
        c = np.ones(n_blocks, dtype=np.float64)
        lb = np.ones(n_tsubsets, dtype=np.float64)
        ub = np.full(n_tsubsets, np.inf, dtype=np.float64)
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(n_blocks, dtype=np.int8)
        bounds = Bounds(0, 1)
        
        x0 = np.zeros(n_blocks)
        for i in best_indices:
            x0[i] = 1.0
        
        try:
            remaining_time = int(deadline - time.time() - 10)
            if remaining_time > 5:
                result = milp(
                    c=c,
                    constraints=constraints,
                    integrality=integrality,
                    bounds=bounds,
                    x0=x0,
                    options={'time_limit': remaining_time, 'mip_gap': 0.005}
                )
                
                if result.status in (0, 1):
                    x = result.x
                    selected_idx = [i for i in range(n_blocks) if x[i] > 0.5]
                    if selected_idx and len(selected_idx) < best_len:
                        best_len = len(selected_idx)
                        best_indices = selected_idx
                        best_blocks = [all_blocks[i] for i in best_indices]
        except Exception:
            pass
    
    if best_blocks is None:
        blocks_idx = standard_greedy(0)
        blocks_idx = local_search(blocks_idx, 0, max_swap_rounds=10)
        best_blocks = [all_blocks[i] for i in blocks_idx]
        
    return best_blocks
