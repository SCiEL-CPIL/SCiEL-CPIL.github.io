import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    # Keep a safety margin of 5 seconds before the 120s limit
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    
    # Precompute all t-subsets for a given block is done on the fly
    # but we can cache or just compute. C(7,3)=35 is small.
    
    def get_subsets(block):
        return set(itertools.combinations(block, t))
        
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(itertools.combinations(all_points, t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to anchor the new block
            # Using a random choice ensures we explore different parts of the space
            anchor = random.choice(list(uncovered))
            
            # Points available to add to the anchor
            # Anchor has t points. We need k - t more.
            other_points = [p for p in all_points if p not in anchor]
            
            best_block = None
            best_gain = -1
            
            # Number of random candidates to sample
            # Higher is better but slower. 200 is a good balance.
            num_samples = 200
            
            candidates = set()
            
            # Generate candidates
            for _ in range(num_samples):
                # Randomly pick k-t points from other_points
                add_pts = random.sample(other_points, k - t)
                # Combine and sort to ensure unique representation
                block = tuple(sorted(anchor + add_pts))
                candidates.add(block)
                
            # Evaluate candidates
            for block in candidates:
                block_subs = get_subsets(block)
                gain = len(uncovered.intersection(block_subs))
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                # Fallback: should not happen if uncovered is not empty
                add_pts = random.sample(other_points, k - t)
                best_block = tuple(sorted(anchor + add_pts))
                
            blocks.append(list(best_block))
            
            # Update uncovered set
            best_subs = get_subsets(best_block)
            uncovered.difference_update(best_subs)
            
        return blocks

    def reduce_blocks(blocks):
        if not blocks:
            return []
            
        # Convert to frozensets for hashing and iteration
        block_sets = [frozenset(b) for b in blocks]
        n = len(block_sets)
        
        # Map each triple to the list of indices of blocks covering it
        triple_to_blocks = {}
        for i, b in enumerate(block_sets):
            for tr in itertools.combinations(b, t):
                if tr not in triple_to_blocks:
                    triple_to_blocks[tr] = []
                triple_to_blocks[tr].append(i)
                
        removed = [False] * n
        
        # Iteratively remove non-essential blocks
        # A block is non-essential if every triple it covers is also covered by at least one other active block.
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                
            # Check if block i is redundant
            redundant = True
            for tr in itertools.combinations(block_sets[i], t):
                # Check if any other active block covers tr
                found_other = False
                for j in triple_to_blocks[tr]:
                    if j != i and not removed[j]:
                        found_other = True
                        break
                if not found_other:
                    redundant = False
                    break
                    
            if redundant:
                removed[i] = True
                changed = True
                
        new_blocks = [list(block_sets[i]) for i in range(n) if not removed[i]]
        return new_blocks

    best_blocks = None
    best_count = float('inf')
    
    # Iterated Greedy
    # We run multiple trials with different seeds.
    # We stop when we hit the deadline or find a very good solution.
    trial = 0
    while time.time() < deadline:
        if trial == 0:
             # First trial with fixed seed for reproducibility in case of crash
             blocks = greedy_construct(42)
        else:
             blocks = greedy_construct(trial)
             
        reduced_blocks = reduce_blocks(blocks)
        
        if len(reduced_blocks) < best_count:
            best_count = len(reduced_blocks)
            best_blocks = reduced_blocks
            
        # If we found a covering with 76 blocks (best known), we can try to improve
        # but we should also be ready to return.
        # The lower bound is 69.
        if best_count <= 76:
            # Run a few more trials to see if we can go lower, but don't spend too much time
            # If we hit 70, it's a major breakthrough.
            if best_count <= 70:
                break
                
        trial += 1
        # Limit trials if we are very fast, to avoid infinite loop if time check is slow?
        # No, time check is robust.
        
    if best_blocks is None:
        # Fallback
        blocks = greedy_construct(0)
        best_blocks = reduce_blocks(blocks)

    return best_blocks
