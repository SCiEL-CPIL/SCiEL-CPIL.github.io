import itertools
import random
import time

# Precomputation for v=24, k=7, t=3
V = 24
K = 7
T = 3
TOTAL_TRIPLES = 2024

# Map each triple (a, b, c) to a unique index 0..2023
triplet_indices = {}
triplets_list = []
idx = 0
for a in range(V):
    for b in range(a + 1, V):
        for c in range(b + 1, V):
            triplet_indices[(a, b, c)] = idx
            triplets_list.append((a, b, c))
            idx += 1

# Precompute all 7-blocks and the list of triplet indices they cover
all_blocks = []
block_triplets = []  # List of lists of triplet indices

# Generate all C(24,7) blocks
for block_tuple in itertools.combinations(range(V), K):
    all_blocks.append(block_tuple)
    # Compute triplets in this block
    covered = []
    for t_tuple in itertools.combinations(block_tuple, T):
        covered.append(triplet_indices[tuple(sorted(t_tuple))])
    block_triplets.append(covered)

NUM_BLOCKS = len(all_blocks)

def construct(v, k, t):
    # The task specifies v, k, t might be passed, but the context is fixed to 24,7,3.
    # We assume the evaluator passes 24, 7, 3. If not, we fall back to a simple greedy.
    if v != 24 or k != 7 or t != 3:
        # Fallback for other parameters (not expected in this specific task instance)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Greedy Initialization
    # We want to start with a covering that is valid, or close to it.
    # Greedy Set Cover
    uncovered_count = [0] * TOTAL_TRIPLES  # 0 means uncovered
    for i in range(TOTAL_TRIPLES):
        uncovered_count[i] = 1
    
    current_blocks = []
    current_coverage = [0] * TOTAL_TRIPLES  # count of blocks covering each triplet
    
    # Greedy selection
    # To speed up, we can just pick blocks that cover the most uncovered triples.
    # Since we have precomputed block_triplets, we can iterate.
    # However, checking all 346k blocks each step is expensive.
    # Let's use a simpler greedy: pick random blocks? No, greedy set cover is better.
    # Optimized Greedy:
    # We maintain a list of uncovered triplets.
    # For each step, find the block that covers the most uncovered triplets.
    
    # Actually, a random restart local search from a random initial set might be more robust
    # to find a valid covering if we start with a slightly larger set.
    
    # Let's try to find a valid covering with N blocks.
    # Best known is 76. Let's aim for 78, 77, 76.
    
    best_valid_blocks = None
    
    # Helper to run local search for a given number of blocks N
    def local_search(N, max_iterations, start_blocks=None):
        if start_blocks is None:
            # Random initialization
            current_blocks = random.sample(range(NUM_BLOCKS), N)
        else:
            current_blocks = list(start_blocks)
        
        # Initialize coverage counts
        counts = [0] * TOTAL_TRIPLES
        for b_idx in current_blocks:
            for t_idx in block_triplets[b_idx]:
                counts[t_idx] += 1
        
        # Calculate initial uncovered
        uncovered = sum(1 for c in counts if c == 0)
        
        if uncovered == 0:
            return current_blocks, 0
        
        best_uncovered = uncovered
        best_blocks = list(current_blocks)
        
        # Simulated Annealing parameters
        # T0 = 1.0, T_end = 0.01, cooling rate
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        i = 0
        while i < max_iterations and time.time() < deadline:
            # Pick a random block to replace
            pos = random.randint(0, N - 1)
            old_block_idx = current_blocks[pos]
            
            # Pick a new random block
            new_block_idx = random.randint(0, NUM_BLOCKS - 1)
            
            if new_block_idx == old_block_idx:
                i += 1
                continue
            
            # Calculate delta in uncovered count
            # Remove old_block_idx
            # For each triplet in old_block:
            #   if counts[t] == 1, then it becomes 0 -> uncovered increases by 1
            #   counts[t] -= 1
            # Add new_block_idx
            # For each triplet in new_block:
            #   if counts[t] == 0, then it becomes 1 -> uncovered decreases by 1
            #   counts[t] += 1
            
            # We need to be careful with order if a triplet is in both.
            # Let's just compute the net change.
            
            delta = 0
            old_trips = block_triplets[old_block_idx]
            new_trips = block_triplets[new_block_idx]
            
            # To avoid double counting issues in delta calculation,
            # it's safer to apply the changes to a temporary structure or just 
            # re-evaluate the affected triplets.
            
            # Affected triplets are union of old_trips and new_trips.
            # Since sets are small (35 each), we can just iterate.
            
            # Step 1: Remove old
            # We'll store the state of counts for affected triplets before change
            # Actually, let's just do the update and rollback if rejected.
            
            # Update counts
            for t_idx in old_trips:
                counts[t_idx] -= 1
                if counts[t_idx] == 0:
                    delta += 1
            
            for t_idx in new_trips:
                counts[t_idx] += 1
                if counts[t_idx] == 1:
                    delta -= 1
            
            new_uncovered = uncovered + delta
            
            if new_uncovered < uncovered or (T > T_min and random.random() < pow(2.71828, -new_uncovered / T) and new_uncovered > uncovered):
                # Accept
                current_blocks[pos] = new_block_idx
                uncovered = new_uncovered
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_blocks = list(current_blocks)
                    if uncovered == 0:
                        return best_blocks, 0
            else:
                # Rollback
                for t_idx in old_trips:
                    counts[t_idx] += 1
                for t_idx in new_trips:
                    counts[t_idx] -= 1
            
            T *= cooling_rate
            i += 1
            
        return best_blocks, best_uncovered

    # Strategy:
    # 1. Try to find a valid covering with N=76, 77, 78...
    # We can start with N=80 (easy to cover) and try to reduce?
    # Or start with N=76 and try to make it valid.
    
    # Let's try increasing N from 76 upwards until we find a valid one.
    # If we find one with 76, we return it.
    # If we find one with 77, we keep it and try 76 with the remaining time.
    
    # To speed up finding a valid covering, we can use a greedy initialization for local search.
    # Greedy construction of N blocks:
    def greedy_init(N):
        counts = [0] * TOTAL_TRIPLES
        blocks = []
        # We want to pick blocks that cover many uncovered triplets.
        # This is hard to do efficiently without precomputation of "uncovered count per block".
        # Let's just use random blocks for local search start, but maybe bias towards random.
        # Actually, a good heuristic: Start with random N blocks.
        return random.sample(range(NUM_BLOCKS), N)

    # Let's try N from 76 to 80
    # We allocate time for each N.
    
    # First, let's try to get *any* valid covering quickly to have a baseline.
    # Greedy Set Cover (slow but reliable for valid)
    # Let's run a fast greedy first to get a valid set, then optimize.
    
    # Fast Greedy:
    # 1. Start with all triplets uncovered.
    # 2. Pick block covering most uncovered.
    # 3. Repeat.
    
    # To make greedy fast, we can precompute for each block the list of triplets.
    # We maintain a set of uncovered triplets.
    # For each block, count how many of its triplets are in the uncovered set.
    # This is O(NUM_BLOCKS * 35) per step. 346k * 35 ~ 12M ops.
    # If we need ~80 steps, that's 1B ops. Too slow in Python.
    
    # Alternative: Randomized Greedy.
    # Pick 100 random blocks, keep the one with max gain.
    
    def randomized_greedy(N_target):
        counts = [0] * TOTAL_TRIPLES
        blocks = []
        for _ in range(N_target):
            best_block = -1
            best_gain = -1
            # Sample 200 random blocks
            for _ in range(200):
                b_idx = random.randint(0, NUM_BLOCKS - 1)
                gain = 0
                for t_idx in block_triplets[b_idx]:
                    if counts[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block != -1:
                blocks.append(best_block)
                for t_idx in block_triplets[best_block]:
                    counts[t_idx] += 1
        
        # Check if valid
        uncovered = sum(1 for c in counts if c == 0)
        return blocks, uncovered

    # Let's try to find a valid covering with N=80 using randomized greedy.
    # Then use local search to try to reduce N or just keep it if we can't go lower.
    
    # Actually, the local search is the main workhorse.
    # Let's run local search for N=76, 77, 78.
    
    # We will give more time to smaller N if we haven't found a valid one yet.
    
    found_valid = None
    
    # Try N=76
    # Run local search for a while
    # If time runs out, we return the best valid found.
    
    # To ensure we return *something* valid, we should have a fallback.
    # The seed program always returns a valid covering.
    # We must ensure our program does too.
    
    # Let's run a simple greedy first to get a guaranteed valid covering (might be ~85 blocks).
    # Then we try to improve it with local search to get down to 76-78.
    
    # 1. Guaranteed Valid Covering (Greedy)
    # We'll use the simple greedy from the seed, but it's slow.
    # Let's use the randomized greedy to get a valid covering quickly.
    # If randomized greedy with N=90 fails to cover all, increase N.
    
    valid_blocks = None
    for N_try in [80, 85, 90, 95]:
        # Run randomized greedy
        # To make it deterministic-ish or just fast, we run a few iterations?
        # No, just one run. If it's not valid, try larger N.
        # Actually, let's just run the local search directly on a random set of size N_try.
        # If it converges to 0, we have a valid covering.
        
        # Let's try to find a valid covering with N=80 first.
        # If we find one, we try to reduce to 79, 78...
        
        pass

    # Revised Plan:
    # 1. Run local search for N=76 for 5 seconds.
    # 2. If valid, return it.
    # 3. Run local search for N=77 for 5 seconds.
    # 4. If valid, return it (unless we found 76).
    # ...
    # 5. If no valid found, run a robust greedy to get *some* valid covering.
    
    # Let's implement a robust greedy that is fast enough.
    # Simple Greedy with randomization to avoid worst-case slowness.
    
    def get_valid_covering():
        # Use randomized greedy to build a covering.
        # Start with N=100, reduce if possible?
        # Or just build until all covered.
        
        counts = [0] * TOTAL_TRIPLES
        blocks = []
        # Keep track of uncovered count
        uncovered = TOTAL_TRIPLES
        
        # We pick blocks one by one.
        # To speed up, we pick from a random subset of blocks.
        
        while uncovered > 0 and time.time() < deadline - 10.0:
            best_block = -1
            best_gain = -1
            # Sample 500 blocks
            for _ in range(500):
                b_idx = random.randint(0, NUM_BLOCKS - 1)
                gain = 0
                for t_idx in block_triplets[b_idx]:
                    if counts[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1 or best_gain == 0:
                # No gain found in sample, pick random
                best_block = random.randint(0, NUM_BLOCKS - 1)
            
            blocks.append(best_block)
            for t_idx in block_triplets[best_block]:
                if counts[t_idx] == 0:
                    uncovered -= 1
                counts[t_idx] += 1
        
        if uncovered == 0:
            return blocks
        else:
            # Fallback: Use the seed's greedy (slow but correct)
            # This should rarely happen if we have enough time.
            uncovered_set = set(itertools.combinations(range(v), t))
            blocks_seed = []
            while uncovered_set:
                block = list(min(uncovered_set))
                while len(block) < k:
                    def gain(x):
                        return sum(1 for rest in itertools.combinations(block, t - 1)
                                   if tuple(sorted(rest + (x,))) in uncovered_set)
                    block.append(max((x for x in range(v) if x not in block), key=gain))
                block.sort()
                blocks_seed.append(block)
                uncovered_set.difference_update(itertools.combinations(block, t))
            return blocks_seed

    # Strategy:
    # 1. Try to find a valid covering with N=76 using local search.
    # 2. If successful, return it.
    # 3. Try N=77.
    # 4. Try N=78.
    # 5. If all fail, get a valid covering with greedy (likely >80) and return it.
    
    # We will allocate time slices.
    # Time available: 110s.
    # Slice 1: 10s for N=76
    # Slice 2: 10s for N=77
    # Slice 3: 10s for N=78
    # Slice 4: 10s for N=79
    # Slice 5: 10s for N=80
    # Remaining: Greedy fallback.
    
    best_valid = None
    best_size = float('inf')
    
    for N in [76, 77, 78, 79, 80]:
        if time.time() > deadline - 20.0:
            break
            
        # Run local search for this N
        # Use a random start
        blocks, unc = local_search(N, max_iterations=1000000)
        
        if unc == 0:
            # We found a valid covering with N blocks
            if N < best_size:
                best_size = N
                best_valid = blocks
                
            # If we found 76, we can stop searching for larger N
            # but we might try to find another 76? No, 76 is the target.
            # If we found 76, we should probably just return it immediately?
            # Or keep searching for a better one? 76 is the known best.
            # So if we find 76, we return it.
            if N == 76:
                # Convert to lists
                return [list(all_blocks[b]) for b in best_valid]
                
    # If we didn't find a small valid covering, try to get *any* valid covering
    if best_valid is not None:
        return [list(all_blocks[b]) for b in best_valid]
        
    # Fallback to greedy
    valid_blocks = get_valid_covering()
    
    # Try to improve the greedy result with local search?
    # The greedy result might be ~85 blocks.
    # We can try to swap blocks to reduce the count?
    # Local search minimizes uncovered, not block count.
    # To reduce block count, we'd need to remove blocks and see if it's still valid.
    # This is a different problem (set cover minimization).
    
    # For now, just return the greedy result.
    return valid_blocks

