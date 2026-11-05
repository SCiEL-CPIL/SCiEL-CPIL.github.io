import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s for overhead/reduction
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy trials
    while time.time() < deadline:
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Greedy construction with lookahead
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset to start the block
            start_set = tuple(random.choice(list(uncovered)))
            block = list(start_set)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                remaining_slots = k - len(block)
                candidates = [x for x in range(v) if x not in block]
                
                for p in candidates:
                    temp_block = block + [p]
                    # Calculate immediate gain
                    immediate_gain = 0
                    for s in itertools.combinations(temp_block, t):
                        if tuple(sorted(s)) in uncovered:
                            immediate_gain += 1
                    
                    if remaining_slots == 1:
                        total_gain = immediate_gain
                    else:
                        # Lookahead 1 step
                        best_q_gain = 0
                        for q in [x for x in range(v) if x not in temp_block]:
                            temp_block2 = temp_block + [q]
                            new_gain_q = 0
                            for s in itertools.combinations(temp_block2, t):
                                if tuple(sorted(s)) in uncovered and q in s:
                                    new_gain_q += 1
                            if new_gain_q > best_q_gain:
                                best_q_gain = new_gain_q
                        total_gain = immediate_gain + best_q_gain
                    
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_point = p
                
                if best_point != -1:
                    block.append(best_point)
                else:
                    # Fallback if no gain? Should not happen
                    block.append(candidates[0])
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for s in itertools.combinations(block, t):
                uncovered.discard(tuple(sorted(s)))
        
        if blocks and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks] # Deep copy
    
    if best_blocks is None:
        # Fallback to a simple greedy if something went wrong
        uncovered = set(all_t_subsets)
        best_blocks = []
        while uncovered:
            start_set = tuple(min(uncovered))
            block = list(start_set)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for p in [x for x in range(v) if x not in block]:
                    temp_block = block + [p]
                    gain = sum(1 for s in itertools.combinations(temp_block, t) if tuple(sorted(s)) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                block.append(best_point)
            block.sort()
            best_blocks.append(block)
            for s in itertools.combinations(block, t):
                uncovered.discard(tuple(sorted(s)))
    
    # Post-processing: Remove redundant blocks
    best_blocks = reduce_blocks(best_blocks, v, t)
    
    return best_blocks

def reduce_blocks(blocks, v, t):
    if not blocks:
        return blocks
    
    # Count how many blocks cover each t-subset
    covered_count = {}
    for b in blocks:
        for s in itertools.combinations(b, t):
            ts = tuple(sorted(s))
            covered_count[ts] = covered_count.get(ts, 0) + 1
    
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for b in blocks:
            is_redundant = True
            for s in itertools.combinations(b, t):
                ts = tuple(sorted(s))
                if covered_count[ts] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove b and update counts
                for s in itertools.combinations(b, t):
                    ts = tuple(sorted(s))
                    covered_count[ts] -= 1
                changed = True
            else:
                new_blocks.append(b)
        blocks = new_blocks
        
    return blocks
