<<<<<<< SEARCH
def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all blocks and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block to its t-subset indices
    block_tsubs = [None] * n_blocks
    # Map t-subset index to list of block indices covering it
    tsub_blocks = [[] for _ in range(n_t)]
    
    for i, block in enumerate(all_blocks):
        idxs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_tsubs[i] = idxs
        for idx in idxs:
            tsub_blocks[idx].append(i)

    best_blocks = None
    best_count = float('inf')

    # We will run the heap-based greedy multiple times with different randomizations?
    # The heap greedy is deterministic if we break ties consistently.
    # To introduce diversity, we can randomize the initial order or tie-breaking.
    # However, the "global" nature of the heap greedy often makes it robust.
    # Let's try a few restarts with slight perturbations or just one strong run.
    # Given the speed, we can afford multiple restarts.
    
    n_restarts = 10

    for restart in range(n_restarts):
        if not time_ok():
            break
            
        random.seed(12345 + restart * 7919)
        
        # Lazy Heap Greedy
        gains = [len(block_tsubs[i]) for i in range(n_blocks)]
        covered = [False] * n_t
        covered_count = 0
        selected_blocks = []
        
        # Heap stores (-gain, block_id)
        # We use a lazy deletion approach: when popping, check if gain is still valid
        heap = [(-gains[i], i) for i in range(n_blocks)]
        heapq.heapify(heap)
        
        while covered_count < n_t and time_ok():
            # Pop the best block
            neg_gain, b_idx = heapq.heappop(heap)
            current_gain = -neg_gain
            
            # Check if this gain is still valid (lazy deletion)
            if current_gain != gains[b_idx]:
                continue
            
            if current_gain == 0:
                # No more gains to be had, but we might not be covered?
                # If greedy gets stuck, we might need to fill randomly, but 
                # with C(19,7,4) and 50k blocks, it should cover everything.
                break
                
            # Select this block
            selected_blocks.append(b_idx)
            
            # Update coverage
            for idx in block_tsubs[b_idx]:
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
                    # Update gains of all blocks covering this t-subset
                    for other_b_idx in tsub_blocks[idx]:
                        if gains[other_b_idx] > 0:
                            gains[other_b_idx] -= 1
                            # Push updated gain to heap
                            heapq.heappush(heap, (-gains[other_b_idx], other_b_idx))

        if covered_count == n_t:
            # Convert indices to blocks
            current_blocks = [all_blocks[i] for i in selected_blocks]
            
            # Deletion phase: remove redundant blocks
            if time_ok():
                tsub_block_count = [0] * n_t
                for b_idx in selected_blocks:
                    for idx in block_tsubs[b_idx]:
                        tsub_block_count[idx] += 1
                        
                changed = True
                while changed and time_ok():
                    changed = False
                    for i in range(len(selected_blocks)):
                        if not time_ok():
                            break
                        removable = True
                        for idx in block_tsubs[selected_blocks[i]]:
                            if tsub_block_count[idx] <= 1:
                                removable = False
                                break
                        if removable:
                            for idx in block_tsubs[selected_blocks[i]]:
                                tsub_block_count[idx] -= 1
                            selected_blocks.pop(i)
                            changed = True
                            break

            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [all_blocks[i] for i in selected_blocks]

            if best_count <= 140:
                break

    if best_blocks is None:
        # Fallback: should not happen
        return []

    # MILP polish
    if _HAS_MILP and time_ok():
        best_blocks = _milp_polish(v, k, t, best_blocks, t_sub_to_idx, all_t_subsets, block_tsubs, deadline, all_blocks, n_blocks, tsub_blocks)

    return best_blocks
=======
def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all blocks and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block to its t-subset indices
    block_tsubs = [None] * n_blocks
    # Map t-subset index to list of block indices covering it
    tsub_blocks = [[] for _ in range(n_t)]
    
    for i, block in enumerate(all_blocks):
        idxs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_tsubs[i] = idxs
        for idx in idxs:
            tsub_blocks[idx].append(i)

    best_blocks = None
    best_count = float('inf')

    # Multiple restarts with smooth surrogate gain for diversity
    n_restarts = 15

    for restart in range(n_restarts):
        if not time_ok():
            break
            
        random.seed(12345 + restart * 7919)
        
        # Temperature for smooth surrogate (log-sum-exp soft-max)
        # Lower temperature = more deterministic, higher = more diverse
        temperature = 0.5 + 0.1 * restart
        
        # Lazy Heap Greedy with smooth surrogate
        gains = [float(len(block_tsubs[i])) for i in range(n_blocks)]
        covered = [False] * n_t
        covered_count = 0
        selected_blocks = []
        
        # Heap stores (-smooth_gain, block_id)
        # We use a lazy deletion approach: when popping, check if gain is still valid
        # For smooth surrogate, we add small random noise scaled by temperature
        def compute_smooth_gain(g):
            """Smooth surrogate: g + small noise * temperature"""
            noise = random.gauss(0, 0.01) * temperature
            return g + noise
        
        heap = []
        for i in range(n_blocks):
            sg = compute_smooth_gain(gains[i])
            heap.append((-sg, i))
        heapq.heapify(heap)
        
        while covered_count < n_t and time_ok():
            # Pop the best block
            neg_sg, b_idx = heapq.heappop(heap)
            
            # Recompute smooth gain to check validity (lazy deletion)
            current_sg = compute_smooth_gain(gains[b_idx])
            if abs(current_sg + neg_sg) > 0.001:
                continue
            
            if gains[b_idx] <= 0:
                break
                
            # Select this block
            selected_blocks.append(b_idx)
            
            # Update coverage
            for idx in block_tsubs[b_idx]:
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
                    # Update gains of all blocks covering this t-subset
                    for other_b_idx in tsub_blocks[idx]:
                        if gains[other_b_idx] > 0:
                            gains[other_b_idx] -= 1
                            # Push updated smooth gain to heap
                            new_sg = compute_smooth_gain(gains[other_b_idx])
                            heapq.heappush(heap, (-new_sg, other_b_idx))

        if covered_count == n_t:
            # Convert indices to blocks
            current_blocks = [all_blocks[i] for i in selected_blocks]
            
            # Deletion phase: remove redundant blocks (optimized)
            if time_ok():
                tsub_block_count = [0] * n_t
                for b_idx in selected_blocks:
                    for idx in block_tsubs[b_idx]:
                        tsub_block_count[idx] += 1
                        
                # Sort by number of unique t-subsets (those with count==1) first
                # Blocks with more unique coverage are harder to remove
                def unique_count(b_idx):
                    cnt = 0
                    for idx in block_tsubs[b_idx]:
                        if tsub_block_count[idx] == 1:
                            cnt += 1
                    return cnt
                
                # Try removing blocks with 0 unique t-subsets first
                changed = True
                while changed and time_ok():
                    changed = False
                    # Find candidates in reverse order (less likely to be needed first)
                    for i in range(len(selected_blocks) - 1, -1, -1):
                        if not time_ok():
                            break
                        b_idx = selected_blocks[i]
                        removable = True
                        for idx in block_tsubs[b_idx]:
                            if tsub_block_count[idx] <= 1:
                                removable = False
                                break
                        if removable:
                            for idx in block_tsubs[b_idx]:
                                tsub_block_count[idx] -= 1
                            selected_blocks.pop(i)
                            changed = True
                            break

            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [all_blocks[i] for i in selected_blocks]

            if best_count <= 140:
                break

    if best_blocks is None:
        # Fallback: should not happen
        return []

    # Swap-based local search polish (high-precision pass)
    if time_ok():
        best_blocks = _swap_polish(v, k, t, best_blocks, t_sub_to_idx, all_t_subsets, block_tsubs, deadline, all_blocks, n_blocks, tsub_blocks)

    # MILP polish
    if _HAS_MILP and time_ok():
        best_blocks = _milp_polish(v, k, t, best_blocks, t_sub_to_idx, all_t_subsets, block_tsubs, deadline, all_blocks, n_blocks, tsub_blocks)

    return best_blocks
>>>>>>> REPLACE

Now add the swap polish function before `_milp_polish`:

