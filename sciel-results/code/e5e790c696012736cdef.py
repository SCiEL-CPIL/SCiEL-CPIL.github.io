import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precompute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    # Map triple to index for fast counting
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # We will work with a fixed number of blocks N.
    # We need to find the minimum N such that we can cover all triples.
    # Lower bound for C(24,7,3) is 69. Upper bound 76.
    # Let's start with a reasonable N and try to minimize it.
    
    # To be safe and fast, we can try a range of N values.
    # Or, we can start with a greedy construction to get an initial N, 
    # then use SA to try to reduce the effective coverage by merging or just 
    # running SA with a fixed N and checking if all triples are covered.
    # If not covered, we increase N or accept a penalty.
    
    # Strategy:
    # 1. Generate a random initial set of blocks.
    # 2. Define cost = number of uncovered triples + lambda * (N - target_N) ? 
    #    Actually, simpler: 
    #    We fix N. We run SA to minimize the number of uncovered triples.
    #    If we reach 0 uncovered triples, we have a valid covering with N blocks.
    #    We can then try smaller N.
    
    # To speed up, let's estimate a good starting N.
    # C(24,3) = 2024 triples.
    # Each block covers C(7,3) = 35 triples.
    # Lower bound N >= ceil(2024 / 35) = 58.
    # But overlaps are significant. 69 is the known LB.
    # Let's try N from 70 down to 65? Or start with 76 and try to reduce?
    # Reducing N is hard. It's easier to find a valid covering for a given N.
    
    # Let's try to find the smallest N in [65, 76] that admits a covering.
    # We will run SA for each N, but that's too slow if we do it sequentially for many N.
    # Better: Start with a greedy solution to get an initial N0.
    # Then run SA with N0, trying to remove blocks? No, SA works on fixed size.
    
    # Alternative: 
    # Run SA with a variable number of blocks? Complex.
    
    # Let's stick to fixed N SA.
    # We will iterate N from a lower bound up to an upper bound.
    # But we want the MINIMUM N. So we should check if N=65 works, then 66, etc.
    # However, if N is too small, SA will fail to cover all triples.
    # We can use the number of uncovered triples as the objective.
    # If we find 0 uncovered, we return.
    
    # To save time, we can use a greedy heuristic to get a decent starting point for each N,
    # or just random. Random might be too slow.
    
    # Let's implement a fast greedy to get an initial N.
    def greedy_construct():
        uncovered = set(all_triples)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple
            start_triple = random.choice(list(uncovered))
            block = list(start_triple)
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                # Calculate gain for each candidate
                gains = []
                for x in candidates:
                    gain = 0
                    for pair in itertools.combinations(block, t-1):
                        tri = tuple(sorted(pair + (x,)))
                        if tri in uncovered:
                            gain += 1
                    gains.append((gain, x))
                
                if not gains:
                    break
                    
                max_gain = max(g for g, _ in gains)
                best_candidates = [x for g, x in gains if g == max_gain]
                
                if max_gain > 0:
                    block.append(random.choice(best_candidates))
                else:
                    block.append(random.choice(candidates))
            
            block.sort()
            blocks.append(block)
            
            for tri in itertools.combinations(block, t):
                if tri in uncovered:
                    uncovered.remove(tri)
                    
        return blocks

    # Get an initial upper bound
    init_blocks = greedy_construct()
    current_best_n = len(init_blocks)
    best_blocks_found = init_blocks
    
    # Now try to improve by running SA for N < current_best_n
    # We will try N values from current_best_n - 1 down to 65 (or lower bound)
    # To avoid spending too much time on impossible Ns, we can limit the time per N.
    
    # SA Parameters
    def run_sa(n_blocks, time_limit):
        # Initialize blocks randomly
        blocks = [list(random.sample(range(v), k)) for _ in range(n_blocks)]
        
        # Calculate initial coverage
        counts = [0] * num_triples
        for b in blocks:
            for tri in itertools.combinations(b, t):
                counts[triple_to_idx[tri]] += 1
        
        uncovered = sum(1 for c in counts if c == 0)
        
        # SA Loop
        T = 10.0
        cooling = 0.999
        best_uncovered = uncovered
        best_blocks = blocks[:]
        
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            if uncovered == 0:
                break
                
            # Choose a block to replace
            i = random.randint(0, n_blocks - 1)
            old_block = blocks[i]
            
            # Generate a new block
            # Heuristic: try to cover some uncovered triples
            # Pick a random uncovered triple (if any) to include in new block
            if uncovered > 0:
                # Find indices of uncovered triples
                # This is expensive if done every step. 
                # Optimization: maintain a list of uncovered indices? 
                # Or just pick random triples until we find an uncovered one?
                # Since we have counts, we can check.
                # Let's pick a random triple index, if it's uncovered, use it.
                # If not, just pick a random block.
                # To be more directed:
                # Pick a random point from the old block? No.
                
                # Let's try to construct a new block that covers at least one uncovered triple.
                # Pick a random uncovered triple.
                # We need a fast way to get a random uncovered triple.
                # We can maintain a list of uncovered triple indices.
                
                # For simplicity in this first pass, let's just pick a random block.
                # But that's slow.
                
                # Let's maintain a list of uncovered triple indices.
                pass
            
            # Better approach for SA move:
            # 1. Pick a block index i.
            # 2. Generate a new block.
            #    - To make it effective, we can bias the new block towards covering uncovered triples.
            #    - Pick a random uncovered triple (if list is maintained).
            #    - Extend it to a k-block.
            
            # Let's maintain `uncovered_indices` list.
            # This requires updating it whenever counts change.
            
            # Re-implementing with maintained uncovered list for speed.
            
            # Actually, let's just use a simpler SA that doesn't maintain the list explicitly in the inner loop 
            # but uses the counts array. 
            # To pick a "good" new block, we can sample a few random blocks and pick the one 
            # that covers the most currently uncovered triples? 
            # Or just random.
            
            # Let's try:
            # 1. Pick random i.
            # 2. Generate random new_block.
            # 3. Calculate delta_uncovered.
            
            new_block = list(random.sample(range(v), k))
            
            # Calculate change in uncovered count
            # Old block contributes to counts.
            # New block contributes to counts.
            
            # This is O(k^t) per move. For k=7, t=3, 35 triples.
            # 35 operations per move. Very fast.
            
            delta = 0
            # Remove old block's contribution
            for tri in itertools.combinations(old_block, t):
                idx = triple_to_idx[tri]
                if counts[idx] == 1:
                    delta += 1 # Becomes uncovered
                counts[idx] -= 1
            
            # Add new block's contribution
            for tri in itertools.combinations(new_block, t):
                idx = triple_to_idx[tri]
                if counts[idx] == 0:
                    delta -= 1 # Becomes covered
                counts[idx] += 1
            
            # Check if new block is distinct from others? 
            # The problem says "blocks must be distinct".
            # If new_block == old_block, delta is 0.
            # If new_block == other_block, we have a duplicate.
            # Duplicates are invalid.
            # We should reject moves that create duplicates.
            # Checking distinctness is O(N). 
            # N is ~70. O(70) per move is acceptable.
            
            is_dup = False
            for j in range(n_blocks):
                if j == i:
                    continue
                if blocks[j] == new_block:
                    is_dup = True
                    break
            
            if is_dup:
                # Revert counts
                for tri in itertools.combinations(new_block, t):
                    idx = triple_to_idx[tri]
                    counts[idx] -= 1
                for tri in itertools.combinations(old_block, t):
                    idx = triple_to_idx[tri]
                    counts[idx] += 1
                continue
            
            # Acceptance criterion
            if delta < 0 or random.random() < math.exp(-delta / T):
                blocks[i] = new_block
                uncovered += delta
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_blocks = blocks[:]
            else:
                # Revert counts
                for tri in itertools.combinations(new_block, t):
                    idx = triple_to_idx[tri]
                    counts[idx] -= 1
                for tri in itertools.combinations(old_block, t):
                    idx = triple_to_idx[tri]
                    counts[idx] += 1
            
            T *= cooling
            
        return best_blocks, best_uncovered

    # Try to find a solution with fewer blocks
    # We know current_best_n is valid.
    # Let's try N = current_best_n - 1, current_best_n - 2, ...
    # Limit time per N to avoid getting stuck.
    
    # How much time to spend on each N?
    # Total time ~145s.
    # If we try 5 values of N, we have ~29s each.
    
    # Let's prioritize lower Ns.
    # Start from current_best_n - 1 down to max(65, current_best_n - 5)
    
    # Actually, if current_best_n is 76, we try 75, 74, 73, 72, 71.
    # If we find a valid one, we update best.
    
    # To be more efficient, we can run SA for a fixed N and see if it converges to 0 uncovered.
    
    # Let's adjust the SA to be more directed.
    # In the move generation, instead of purely random new_block, 
    # we can try to include a point from an uncovered triple.
    
    # Modified SA with directed moves:
    def run_sa_directed(n_blocks, time_limit):
        blocks = [list(random.sample(range(v), k)) for _ in range(n_blocks)]
        
        counts = [0] * num_triples
        for b in blocks:
            for tri in itertools.combinations(b, t):
                counts[triple_to_idx[tri]] += 1
        
        uncovered_indices = [i for i, c in enumerate(counts) if c == 0]
        uncovered = len(uncovered_indices)
        
        T = 10.0
        cooling = 0.999
        best_uncovered = uncovered
        best_blocks = blocks[:]
        
        t_start = time.time()
        
        while time.time() - t_start < time_limit and uncovered > 0:
            i = random.randint(0, n_blocks - 1)
            old_block = blocks[i]
            
            # Generate new block
            # Strategy: Pick a random uncovered triple if available
            if uncovered_indices:
                target_triple_idx = random.choice(uncovered_indices)
                target_triple = all_triples[target_triple_idx]
                # Start with target triple
                new_block_set = set(target_triple)
                # Fill rest randomly
                remaining = [x for x in range(v) if x not in new_block_set]
                random.shuffle(remaining)
                for x in remaining:
                    if len(new_block_set) == k:
                        break
                    new_block_set.add(x)
                new_block = list(new_block_set)
            else:
                new_block = list(random.sample(range(v), k))
            
            # Check duplicate
            is_dup = False
            for j in range(n_blocks):
                if j == i:
                    continue
                if blocks[j] == new_block:
                    is_dup = True
                    break
            
            if is_dup:
                continue
            
            # Calculate delta
            delta = 0
            for tri in itertools.combinations(old_block, t):
                idx = triple_to_idx[tri]
                if counts[idx] == 1:
                    delta += 1
                counts[idx] -= 1
            
            for tri in itertools.combinations(new_block, t):
                idx = triple_to_idx[tri]
                if counts[idx] == 0:
                    delta -= 1
                counts[idx] += 1
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                blocks[i] = new_block
                uncovered += delta
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_blocks = blocks[:]
            else:
                # Revert
                for tri in itertools.combinations(new_block, t):
                    idx = triple_to_idx[tri]
                    counts[idx] -= 1
                for tri in itertools.combinations(old_block, t):
                    idx = triple_to_idx[tri]
                    counts[idx] += 1
            
            T *= cooling
            
        return best_blocks, best_uncovered

    # Main loop: Try to reduce N
    # We will try N values in descending order from current_best_n - 1.
    # We will spend a fraction of the time on each.
    
    # Let's try up to 3 values of N to save time for the best one.
    # Or, we can just run SA on the current N to improve it? 
    # No, SA on fixed N doesn't reduce N.
    # We MUST try smaller N to get a better score.
    
    # Let's try N = current_best_n - 1.
    # If it works, we have a better solution.
    # If not, we keep the current best.
    
    # To maximize chance, we can run multiple restarts of SA for a given N within the time limit.
    
    # Let's define a function to try a specific N
    def try_n(n, time_limit):
        # Run SA multiple times
        best = None
        best_unc = float('inf')
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            blocks, unc = run_sa_directed(n, 5.0) # 5s per run
            if unc < best_unc:
                best_unc = unc
                best = blocks
            if unc == 0:
                return best, 0
        
        return best, best_unc

    # Try N = current_best_n - 1
    # Allocate time based on remaining time.
    
    # Let's try a few Ns.
    # Start with current_best_n - 1.
    
    # If time is tight, we might only be able to try one N.
    
    # Let's try N = current_best_n - 1 with most of the time.
    
    # Wait, if current_best_n is 76, trying 75 might be hard.
    # Maybe we should just run SA on the current set of blocks but allow "deletion" moves?
    # That's complex.
    
    # Let's stick to: 
    # 1. Greedy to get N0.
    # 2. Try N0-1 with SA.
    # 3. If success, try N0-2.
    
    # To be safe, let's just try N0-1.
    
    # But what if N0 is already optimal? Then we waste time.
    # We can check if N0 is close to the lower bound.
    # LB is 69.
    
    # Let's try N values from current_best_n - 1 down to max(69, current_best_n - 3).
    
    # Allocate time:
    # Total remaining time ~ 100s (after greedy).
    # Let's split it among the candidate Ns.
    
    candidate_ns = list(range(current_best_n - 1, max(69, current_best_n - 4), -1))
    
    if not candidate_ns:
        return best_blocks_found
    
    # Distribute time
    time_per_n = 25.0
    
    for n in candidate_ns:
        if time.time() > deadline - 5:
            break
            
        # Run SA for this N
        # We can run multiple short runs or one long run.
        # One long run is better for SA.
        
        # But if we fail, we should try the next N.
        
        # Let's run SA for 20s for each N.
        
        # Actually, if we find a valid covering for N, we should update best and continue to N-1.
        
        blocks_n, unc_n = run_sa_directed(n, 20.0)
        
        if unc_n == 0:
            # Found a valid covering with n blocks
            best_blocks_found = blocks_n
            current_best_n = n
            # Continue to try smaller N?
            # Yes, loop will continue to next smaller N if we adjust logic.
            # But my loop is fixed range. 
            # Let's just break if we find a valid one and try to optimize it further?
            # No, we want the SMALLEST N.
            # So if we find valid for N, we should try N-1.
            
            # The loop structure `for n in candidate_ns` handles this if candidate_ns is ordered correctly.
            # candidate_ns is [N0-1, N0-2, ...]
            # So if N0-1 works, we update best. Then we try N0-2.
            
        # If unc_n > 0, we didn't find a covering for this N.
        # We keep the previous best.

    return best_blocks_found

