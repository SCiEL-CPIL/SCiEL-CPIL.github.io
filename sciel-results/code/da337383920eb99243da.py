import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    points = range(v)
    all_5s = list(itertools.combinations(points, t))
    num_5s = len(all_5s)
    map_5_to_idx = {s: i for i, s in enumerate(all_5s)}
    
    all_8s = list(itertools.combinations(points, k))
    num_8s = len(all_8s)
    map_8_to_id = {b: i for i, b in enumerate(all_8s)}
    
    block_5s = [[] for _ in range(num_8s)]
    s_to_blocks = [[] for _ in range(num_5s)]
    
    for b_id, b in enumerate(all_8s):
        subs = list(itertools.combinations(b, t))
        idxs = [map_5_to_idx[s] for s in subs]
        block_5s[b_id] = idxs
        for idx in idxs:
            s_to_blocks[idx].append(b_id)
            
    def run_greedy(seed):
        random.seed(seed)
        uncovered = set(range(num_5s))
        cover_count = [0] * num_5s
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            idx = random.choice(list(uncovered))
            candidates = s_to_blocks[idx]
            
            best_block = -1
            best_gain = -1
            
            # Pick top 5 candidates randomly?
            # Just max gain for simplicity first.
            for b_id in candidates:
                gain = 0
                for s_idx in block_5s[b_id]:
                    if s_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_id
            
            if best_block == -1:
                break
                
            blocks.append(best_block)
            for s_idx in block_5s[best_block]:
                cover_count[s_idx] += 1
                if cover_count[s_idx] == 1:
                    uncovered.discard(s_idx)
                    
        return blocks, cover_count

    def remove_redundant(blocks, cover_count):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b_id = blocks[i]
                is_redundant = True
                for s_idx in block_5s[b_id]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    blocks.pop(i)
                    for s_idx in block_5s[b_id]:
                        cover_count[s_idx] -= 1
                    changed = True
        return blocks, cover_count

    best_blocks = None
    best_cover_count = None
    best_len = float('inf')
    
    # Run multiple greedy tries
    for i in range(5):
        if time.time() > deadline - 20: # Leave time for local search
            break
        b, c = run_greedy(i)
        b, c = remove_redundant(b, c)
        if len(b) < best_len:
            best_len = len(b)
            best_blocks = b
            best_cover_count = c
            
    if best_blocks is None:
        # Fallback: just return empty or first greedy?
        # Should not happen.
        return []

    # Local Search on best_blocks
    blocks = best_blocks
    cover_count = best_cover_count
    
    while time.time() < deadline:
        # Snapshot
        blocks_snap = blocks[:]
        cover_count_snap = cover_count[:]
        
        # Move: Add Random Block
        b_new_id = random.choice(range(num_8s))
        blocks.append(b_new_id)
        for s_idx in block_5s[b_new_id]:
            cover_count[s_idx] += 1
            
        # Remove Redundant
        blocks, cover_count = remove_redundant(blocks, cover_count)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
        else:
            # Revert
            blocks = blocks_snap
            cover_count = cover_count_snap
            
        # Move: Swap
        if time.time() > deadline:
            break
        if blocks:
            i = random.choice(range(len(blocks)))
            b_old_id = blocks[i]
            
            # Temporarily remove
            blocks.pop(i)
            for s_idx in block_5s[b_old_id]:
                cover_count[s_idx] -= 1
            
            # Find uncovered
            uncovered = [s_idx for s_idx in block_5s[b_old_id] if cover_count[s_idx] == 0]
            
            if uncovered:
                s_rand = random.choice(uncovered)
                candidates = s_to_blocks[s_rand]
                found_swap = False
                for b_new_id in candidates:
                    if b_new_id == b_old_id:
                        continue
                    # Check if b_new_id covers all uncovered
                    ok = True
                    for s_idx in uncovered:
                        if cover_count[s_idx] == 0:
                            # Check if b_new_id covers s_idx
                            if s_idx not in block_5s[b_new_id]:
                                ok = False
                                break
                    if ok:
                        # Swap
                        blocks.append(b_new_id)
                        for s_idx in block_5s[b_new_id]:
                            cover_count[s_idx] += 1
                        found_swap = True
                        break
                
                if found_swap:
                    blocks, cover_count = remove_redundant(blocks, cover_count)
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                    else:
                        blocks = blocks_snap
                        cover_count = cover_count_snap
                else:
                    # Revert removal
                    blocks.insert(i, b_old_id)
                    for s_idx in block_5s[b_old_id]:
                        cover_count[s_idx] += 1
            else:
                # b_old_id was redundant? Restore
                blocks.insert(i, b_old_id)
                for s_idx in block_5s[b_old_id]:
                    cover_count[s_idx] += 1

    return [list(all_8s[b_id]) for b_id in blocks]
