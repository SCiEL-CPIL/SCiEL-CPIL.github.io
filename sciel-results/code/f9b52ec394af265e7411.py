import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute all t-subsets (triplets)
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Helper to get triplet ID for a sorted tuple
    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    # Precompute for each point, the list of triplets it participates in
    # point_triplets[p] = list of triplet IDs containing p
    point_triplets = [[] for _ in range(v)]
    for p in range(v):
        for i, triplet in enumerate(all_triplets):
            if p in triplet:
                point_triplets[p].append(i)
    
    # Precompute for each pair of points, the list of triplets containing both
    # pair_triplets[(a,b)] = list of triplet IDs containing a and b
    # Since v=24, pairs are 24*23/2 = 276. Small.
    pair_triplets = {}
    for a in range(v):
        for b in range(a+1, v):
            pair_triplets[(a,b)] = []
    for i, triplet in enumerate(all_triplets):
        for a in range(3):
            for b in range(a+1, 3):
                p1, p2 = triplet[a], triplet[b]
                if p1 > p2: p1, p2 = p2, p1
                pair_triplets[(p1,p2)].append(i)
    
    def greedy_construct(v, k, t, seed=0):
        random.seed(seed)
        # uncovered is a set of IDs
        uncovered = set(range(n_triplets))
        blocks = []
        
        while uncovered:
            # Pick an uncovered triplet to start the block
            # Random helps diversify
            start_triplet_id = random.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            
            while len(block) < k:
                best_point = -1
                max_gain = -1
                # Candidate points
                candidates = [p for p in range(v) if p not in block]
                
                # Calculate gain for each candidate
                # Gain = number of uncovered triplets that will be covered if p is added
                # A triplet {a,b,c} is covered if {a,b,c} is subset of block
                # When we add p, we cover triplets of form {x, y, p} where {x,y} subset of block
                # and {x, y, p} is in uncovered.
                
                # Precompute pairs in block for efficiency
                pairs_in_block = list(itertools.combinations(block, t-1))
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        # Form triplet
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p
                
                if max_gain == 0:
                    # No point improves coverage of existing uncovered triplets?
                    # Just pick a random one to complete the block
                    best_point = random.choice(candidates)
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
        
        return blocks

    # 1. Initial Construction
    # Try a few seeds to get a good starting point
    best_blocks = None
    best_count = float('inf')
    
    # Use more seeds if time permits, but start with 10
    num_seeds = 10
    for seed in range(num_seeds):
        if time.time() > deadline - 10: # Leave 10s for optimization
            break
        blocks = greedy_construct(v, k, t, seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback if greedy failed (shouldn't happen)
        best_blocks = greedy_construct(v, k, t, 0)

    # 2. Local Search Optimization
    # Maintain coverage count array
    # count[tid] = number of blocks in current solution covering triplet tid
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                count[tid] += delta

    update_counts(best_blocks, 1)
    
    current_blocks = list(best_blocks)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                # Remove
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        # Iterate pairs
        # To speed up, we can randomize order or just iterate
        # Randomize order of pairs to avoid getting stuck in local minima
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for i, j in pairs:
            if time.time() > deadline:
                return blocks, False
                
            b1 = blocks[i]
            b2 = blocks[j]
            
            # Determine U_need: triplets that will become uncovered if b1, b2 removed
            # A triplet T is in U_need if count[T] == (1 if T in b1 else 0) + (1 if T in b2 else 0)
            # AND count[T] > 0 (it must be covered currently)
            
            u_need = []
            
            # Precompute set of triplet IDs for b1 and b2
            ids_b1 = set()
            for triplet in itertools.combinations(b1, t):
                ids_b1.add(get_id(triplet))
                
            ids_b2 = set()
            for triplet in itertools.combinations(b2, t):
                ids_b2.add(get_id(triplet))
            
            # Union of IDs covered by b1 and b2
            check_ids = ids_b1 | ids_b2
            
            for tid in check_ids:
                c = count[tid]
                covered_by_b1 = 1 if tid in ids_b1 else 0
                covered_by_b2 = 1 if tid in ids_b2 else 0
                if c == covered_by_b1 + covered_by_b2:
                    # This triplet is ONLY covered by b1 and/or b2
                    # So it must be covered by the new block
                    u_need.append(all_triplets[tid])
            
            if not u_need:
                # Both are redundant together?
                # If u_need is empty, it means S \ {b1, b2} covers everything.
                # So we can remove BOTH.
                # Update counts
                update_counts([b1, b2], -1)
                # Remove from list
                # Be careful with indices
                return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
            
            # Check if a single block can cover u_need
            # Union of points in u_need
            p_need = set()
            for triplet in u_need:
                p_need.update(triplet)
            
            if len(p_need) <= k:
                # We can form a block containing p_need
                # Construct new block
                new_block = list(p_need)
                # Fill with random/other points to reach size k
                while len(new_block) < k:
                    # Pick a point not in new_block
                    # Using random to vary
                    p = random.randint(0, v-1)
                    if p not in new_block:
                        new_block.append(p)
                new_block.sort()
                
                # Replace b1, b2 with new_block
                # Update counts
                update_counts([b1, b2], -1)
                update_counts([new_block], 1)
                
                # Update list
                new_blocks = []
                for idx, b in enumerate(blocks):
                    if idx != i and idx != j:
                        new_blocks.append(b)
                new_blocks.append(new_block)
                
                return new_blocks, True

        return blocks, False

    # Main Optimization Loop
    progress_made = True
    while progress_made and time.time() < deadline:
        progress_made = False
        # Try removing redundants
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True
            
        # Try swapping 2 for 1
        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True
                # After a swap, we might create new redundants, so loop continues
        
    return current_blocks

# Test/Entry
if __name__ == "__main__":
    # Test with small parameters to ensure no crash
    # blocks = construct(8, 4, 2)
    # print(len(blocks))
    
    # Run for the actual task parameters if needed by evaluator
    # The evaluator calls construct(24, 7, 3)
    pass
