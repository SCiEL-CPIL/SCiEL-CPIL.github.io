import itertools
import random
import time

def construct(v, k, t):
    deadline = time.time() + 138
    
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    block_tsub_cache = {}
    def get_block_tsubs(block):
        if block not in block_tsub_cache:
            block_tsub_cache[block] = tuple(tsub_index[ts] for ts in itertools.combinations(block, t))
        return block_tsub_cache[block]
    
    def greedy_build(rng):
        uncovered = [1] * n_tsub
        blocks = []
        
        while True:
            min_c = min(uncovered)
            if min_c >= 1:
                break
            candidates = [i for i, c in enumerate(uncovered) if c == min_c]
            ts_idx = rng.choice(candidates)
            ts = all_tsubsets[ts_idx]
            
            block = list(ts)
            block_set = set(block)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        idx = tsub_index.get(cand)
                        if idx is not None and uncovered[idx] < 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
            
            block = tuple(sorted(block))
            blocks.append(block)
            for idx in get_block_tsubs(block):
                uncovered[idx] -= 1
        
        return blocks
    
    def build_coverage(blocks):
        covered = [0] * n_tsub
        for b in blocks:
            for idx in get_block_tsubs(b):
                covered[idx] += 1
        return covered
    
    def try_remove(blocks, rng):
        covered = build_coverage(blocks)
        new_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            indices = list(range(len(new_blocks)))
            rng.shuffle(indices)
            for i in indices:
                if i >= len(new_blocks):
                    continue
                b = new_blocks[i]
                tsub_idxs = get_block_tsubs(b)
                if all(covered[idx] > 1 for idx in tsub_idxs):
                    for idx in tsub_idxs:
                        covered[idx] -= 1
                    new_blocks.pop(i)
                    changed = True
        return new_blocks
    
    def local_search(blocks, rng, dl):
        blocks = list(blocks)
        blocks = try_remove(blocks, rng)
        
        n_iter = 0
        while time.time() < dl and n_iter < 2000:
            n_iter += 1
            improved = False
            
            # Try removal
            new_blocks = try_remove(blocks, rng)
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                improved = True
                continue
            
            # Try swap
            if len(blocks) > 0:
                covered = build_coverage(blocks)
                i = rng.randint(0, len(blocks) - 1)
                b_old = blocks[i]
                old_tsubs = get_block_tsubs(b_old)
                
                for idx in old_tsubs:
                    covered[idx] -= 1
                
                uncovered_now = [idx for idx in range(n_tsub) if covered[idx] == 0]
                
                if not uncovered_now:
                    blocks.pop(i)
                    improved = True
                else:
                    best_block = None
                    best_score = -1
                    unc_set = set(uncovered_now)
                    for _ in range(50):
                        cand = tuple(sorted(rng.sample(range(v), k)))
                        cand_tsubs = get_block_tsubs(cand)
                        cand_set = set(cand_tsubs)
                        if unc_set.issubset(cand_set):
                            score = len(cand_tsubs)
                            if score > best_score:
                                best_score = score
                                best_block = cand
                    
                    if best_block is not None:
                        blocks[i] = best_block
                        improved = True
                    else:
                        for idx in old_tsubs:
                            covered[idx] += 1
        
        return blocks
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy builds
    rng = random.Random(42)
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        r = random.Random(42 + trial * 104729)
        blocks = greedy_build(r)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    # Phase 2: Local search on the best
    if best_blocks is not None:
        dl = deadline - 5
        best_blocks = local_search(best_blocks, random.Random(999), dl)
        best_len = len(best_blocks)
    
    # Phase 3: More trials with local search
    trial = 10
    while time.time() < deadline - 10:
        trial += 1
        r = random.Random(trial * 7919 + 1)
        blocks = greedy_build(r)
        dl = min(deadline - 5, time.time() + 20)
        blocks = local_search(blocks, r, dl)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    return [list(b) for b in best_blocks]
