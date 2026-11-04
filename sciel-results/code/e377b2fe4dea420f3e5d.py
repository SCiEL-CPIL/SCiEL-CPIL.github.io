import itertools
import random
import time

def construct(v: int, k: int, t: int) -> List[List[int]]:
    start_time = time.time()
    # Allow 140 seconds for search, leaving 10s buffer for return/serialization
    deadline = start_time + 140.0
    
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper to check time
    def time_up():
        return time.time() > deadline

    # Randomized Greedy with Restarts
    for restart in range(10000): # High limit, controlled by time
        if time_up():
            break
        
        uncovered = set(all_t_subsets)
        blocks = []
        valid = True
        
        while uncovered:
            if time_up():
                valid = False
                break
            
            # Randomly select an uncovered t-subset
            # Converting to list is O(N). N is small here.
            target = random.choice(list(uncovered))
            uncovered.discard(target)
            
            block = list(target)
            
            # Fill block to size k
            while len(block) < k:
                if time_up():
                    valid = False
                    break
                
                best_gain = -1
                best_candidates = []
                
                # Find point that maximizes newly covered t-subsets
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Check t-1 combinations from block
                    for comb in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(comb + (x,)))
                        if new_sub in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if best_candidates:
                    # Randomly choose among best to diversify
                    next_point = random.choice(best_candidates)
                    block.append(next_point)
                else:
                    # Should not happen
                    valid = False
                    break
            
            if not valid:
                break
            
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        if valid and not uncovered:
            current_count = len(blocks)
            if current_count < best_count:
                best_count = current_count
                best_blocks = [b[:] for b in blocks]
                
                # Early stop if we hit a known good bound? 
                # C(24,7,3) best known is 76.
                # If we find 76, we can stop.
                if best_count <= 76:
                    break

    if best_blocks is None:
        # Fallback: Deterministic Greedy (Seed logic)
        # This is slower per restart but deterministic.
        # We run it once, hoping it finishes before deadline.
        if time.time() < deadline:
            uncovered = set(all_t_subsets)
            blocks = []
            valid = True
            while uncovered:
                if time.time() > deadline:
                    valid = False
                    break
                target = min(uncovered)
                uncovered.discard(target)
                block = list(target)
                while len(block) < k:
                    if time.time() > deadline:
                        valid = False
                        break
                    best_gain = -1
                    best_point = None
                    for x in range(v):
                        if x in block:
                            continue
                        gain = 0
                        for comb in itertools.combinations(block, t - 1):
                            new_sub = tuple(sorted(comb + (x,)))
                            if new_sub in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    if best_point is not None:
                        block.append(best_point)
                    else:
                        # Fallback to any
                        for x in range(v):
                            if x not in block:
                                block.append(x)
                                break
                if not valid:
                    break
                block.sort()
                blocks.append(block)
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)
            
            if valid and not uncovered:
                best_blocks = blocks
            else:
                # Emergency: If even deterministic failed to complete in time,
                # we have a problem. But for v=24, it should be fast.
                # If we truly have no valid blocks, return empty? No, must be valid.
                # We assume at least one strategy finishes.
                pass

    return best_blocks
