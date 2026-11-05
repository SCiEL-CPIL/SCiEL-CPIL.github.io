import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    id_to_subset = {i: s for s, i in subset_to_id.items()}
    
    # subsets_by_point[x] = list of subset IDs containing x
    subsets_by_point = [[] for _ in range(v)]
    for i, s in enumerate(all_subsets):
        for x in s:
            subsets_by_point[x].append(i)
            
    best_blocks = None
    best_count = float('inf')
    
    def greedy_once(seed_val):
        random.seed(seed_val)
        uncovered = set(range(len(all_subsets)))
        blocks = []
        
        while uncovered:
            # Find the "hardest" uncovered subset
            # Heuristic: The subset whose elements have the least "coverage potential"
            # Potential of a point x: number of uncovered subsets containing x
            # We want to pick a subset S where the sum of potentials of its elements is LOW
            # This ensures we cover the "isolated" subsets first.
            
            min_potential = float('inf')
            chosen_sid = None
            # Sample a few subsets to find a hard one, or check all?
            # Checking all is O(N * t). N=3876, t=4. ~15k ops. Fast.
            for sid in uncovered:
                s = id_to_subset[sid]
                # Calculate potential: sum of len(uncovered subsets containing x) for x in s
                # We can precompute point_potentials?
                # point_potentials[x] = len([sid for sid in uncovered if x in id_to_subset[sid]])
                # Updating this dynamically is expensive.
                # Approximation: Just use the static count? No, that's constant.
                # Let's just pick a random subset. The "hardest" heuristic is risky without dynamic potentials.
                
                pass
            
            # Fallback to random selection, but with a twist:
            # Pick a random subset, but if there are many, pick one with lower "degree"?
            # Let's stick to Random Selection but improve the Block Construction.
            
            start_subset_id = random.choice(list(uncovered))
            start_subset = id_to_subset[start_subset_id]
            
            # Force the block to contain all elements of start_subset?
            # No, that uses up 4 slots. We have 7.
            # Let's start with a random element from start_subset.
            first_point = random.choice(start_subset)
            block = [first_point]
            block_set = {first_point}
            
            while len(block) < k:
                best_gain = -1
                candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for sid in subsets_by_point[x]:
                        if sid in uncovered:
                            s = id_to_subset[sid]
                            if all(e in block_set for e in s if e != x):
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [x]
                    elif gain == best_gain:
                        candidates.append(x)
                
                if not candidates:
                    break
                
                # Randomized Top-K Selection
                # Sort candidates by gain? They are all best_gain.
                # Just pick random from top 5
                if len(candidates) > 5:
                    next_point = random.choice(candidates[:5])
                else:
                    next_point = random.choice(candidates)
                
                block.append(next_point)
                block_set.add(next_point)
                
                to_remove = []
                for sid in subsets_by_point[next_point]:
                    if sid in uncovered:
                        s = id_to_subset[sid]
                        if all(elem in block_set for elem in s):
                            to_remove.append(sid)
                
                uncovered.difference_update(to_remove)
            
            block.sort()
            blocks.append(block)
        
        # Post-processing: Remove redundant blocks
        cover_count = [0] * len(all_subsets)
        block_subsets = []
        for b in blocks:
            b_set = set(b)
            subs_in_b = []
            for i, s in enumerate(all_subsets):
                if all(elem in b_set for elem in s):
                    cover_count[i] += 1
                    subs_in_b.append(i)
            block_subsets.append(subs_in_b)
            
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                redundant = True
                for sid in block_subsets[i]:
                    if cover_count[sid] <= 1:
                        redundant = False
                        break
                if redundant:
                    for sid in block_subsets[i]:
                        cover_count[sid] -= 1
                    blocks.pop(i)
                    block_subsets.pop(i)
                    changed = True
                    break
        
        return blocks

    start_time = time.time()
    num_runs = 10
    for i in range(num_runs):
        blocks = greedy_once(i)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    elapsed = time.time() - start_time
    avg_time = elapsed / num_runs
    
    remaining_time = 110 - elapsed
    if remaining_time > 0:
        additional_runs = int(remaining_time / avg_time)
        for i in range(num_runs, num_runs + additional_runs):
            blocks = greedy_once(i)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                
    return best_blocks
