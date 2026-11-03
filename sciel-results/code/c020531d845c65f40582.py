import itertools
import random
import time

def construct(v, k, t):
    deadline = time.time() + 115
    rng = random.Random(42)
    
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    best_blocks = None
    best_count = float('inf')

    def get_block_tsubs(block):
        res = set()
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                res.add(idx)
        return res

    def greedy_construct(deadline):
        uncovered = set(range(n_tsubs))
        blocks = []
        
        while uncovered and time.time() < deadline - 5:
            seed_idx = next(iter(uncovered))
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        candidates = [x]
                    elif gain == best_gain and gain > 0:
                        candidates.append(x)
                
                if best_point == -1:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining)
                    else:
                        break
                
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) < k:
                continue

            block.sort()
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)
        
        return blocks

    def reduce_blocks(blocks, deadline):
        if not blocks:
            return blocks
        
        cover_count = [0] * n_tsubs
        block_tsubs = [get_block_tsubs(b) for b in blocks]
        for tsubs in block_tsubs:
            for idx in tsubs:
                cover_count[idx] += 1
        
        # 1. Remove redundant
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(blocks)):
                is_redundant = True
                for idx in block_tsubs[i]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in block_tsubs[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    changed = True
                    break
        
        # 2. Swap reduction
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                is_redundant = True
                for idx in block_tsubs[i]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in block_tsubs[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    break
                
                unique_tsubs = [idx for idx in block_tsubs[i] if cover_count[idx] == 1]
                if not unique_tsubs:
                    continue
                
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if time.time() > deadline - 5:
                        break
                    block_j = blocks[j]
                    block_j_set = set(block_j)
                    
                    for u_idx in unique_tsubs:
                        u_ts = all_tsubs[u_idx]
                        u_set = set(u_ts)
                        common = block_j_set & u_set
                        if len(common) == t - 1:
                            missing = [p for p in u_ts if p not in block_j_set]
                            if len(missing) != 1:
                                continue
                            missing_p = missing[0]
                            candidates_for_removal = [p for p in block_j if p not in common]
                            
                            for remove_p in candidates_for_removal:
                                new_block_j = [p if p != remove_p else missing_p for p in block_j]
                                new_block_j_set = set(new_block_j)
                                if len(new_block_j_set) != k:
                                    continue
                                new_tsubs_j = get_block_tsubs(new_block_j)
                                if u_idx not in new_tsubs_j:
                                    continue
                                
                                valid = True
                                for old_idx in block_tsubs[j]:
                                    if old_idx not in new_tsubs_j:
                                        if cover_count[old_idx] <= 1:
                                            valid = False
                                            break
                                if not valid:
                                    continue
                                
                                # Apply
                                for idx in block_tsubs[i]:
                                    cover_count[idx] -= 1
                                for idx in block_tsubs[j]:
                                    cover_count[idx] -= 1
                                for idx in new_tsubs_j:
                                    cover_count[idx] += 1
                                
                                blocks.pop(i)
                                block_tsubs.pop(i)
                                blocks[j] = new_block_j
                                block_tsubs[j] = new_tsubs_j
                                
                                improved = True
                                break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        return blocks

    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        blocks = greedy_construct(deadline)
        if not blocks:
            continue
        
        blocks = reduce_blocks(blocks, deadline)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 320:
                break

    if best_blocks is None:
        blocks = greedy_construct(deadline)
        best_blocks = blocks

    return best_blocks
