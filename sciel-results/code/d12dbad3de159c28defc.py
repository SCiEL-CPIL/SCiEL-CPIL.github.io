import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    all_points = list(range(v))
    # Pre-compute all t-subsets to initialize cover_count and uncovered
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    uncovered = set(all_t_subsets)
    blocks = []
    cover_count = {s: 0 for s in all_t_subsets}
    
    def get_subsets(block):
        return itertools.combinations(block, t)
        
    def add_block(block):
        blocks.append(block)
        for s in get_subsets(block):
            if s in uncovered:
                uncovered.remove(s)
            cover_count[s] += 1
            
    def remove_block(block):
        # Using list.remove is O(N), but N is small
        blocks.remove(block)
        for s in get_subsets(block):
            cover_count[s] -= 1
            if cover_count[s] == 0:
                uncovered.add(s)
                
    # Greedy Construction
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset to start the block
        target_subset = random.choice(list(uncovered))
        current_block = list(target_subset)
        
        # Add k-t more points
        while len(current_block) < k:
            best_point = -1
            best_gain = -1
            candidates = [p for p in all_points if p not in current_block]
            
            for p in candidates:
                gain = 0
                # Check subsets formed by p and t-1 points from current_block
                for rest in itertools.combinations(current_block, t - 1):
                    s = tuple(sorted(rest + (p,)))
                    if s in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
                elif gain == best_gain and best_point != -1:
                    if random.random() < 0.5:
                        best_point = p
            
            if best_point == -1:
                best_point = random.choice(candidates)
                
            current_block.append(best_point)
            
        current_block.sort()
        add_block(current_block)
        
    # Local Search Phase
    while time.time() < deadline:
        redundant_found = False
        
        # 1. Check for redundant blocks
        for i in range(len(blocks)):
            block = blocks[i]
            is_redundant = True
            for s in get_subsets(block):
                if cover_count[s] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                remove_block(block)
                redundant_found = True
                break
        
        if redundant_found:
            continue
            
        # 2. Try a swap
        if len(blocks) == 0:
            break
            
        i = random.randint(0, len(blocks) - 1)
        block = blocks[i]
        
        x = random.choice(block)
        candidates = [p for p in all_points if p not in block]
        if not candidates:
            continue
        y = random.choice(candidates)
        
        new_block = sorted((set(block) - {x}) | {y})
        
        # Check validity: unique subsets of block must be covered by new_block
        unique_subsets = set()
        for s in get_subsets(block):
            if cover_count[s] == 1:
                unique_subsets.add(s)
                
        if unique_subsets:
            new_block_subsets = set(get_subsets(new_block))
            if not unique_subsets.issubset(new_block_subsets):
                continue
        
        # Perform swap
        old_subsets = set(get_subsets(block))
        new_subsets = set(get_subsets(new_block))
        
        for s in old_subsets:
            cover_count[s] -= 1
            if cover_count[s] == 0:
                uncovered.add(s)
                
        for s in new_subsets:
            cover_count[s] += 1
            if s in uncovered:
                uncovered.remove(s)
                
        blocks[i] = new_block

    return blocks
