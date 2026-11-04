import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    all_points = list(range(v))
    t_subs = list(itertools.combinations(all_points, t))
    n_t = len(t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # Precompute t-subset indices for all possible blocks
    all_blocks = list(itertools.combinations(all_points, k))
    block_t_indices = {}
    for b in all_blocks:
        block_t_indices[b] = np.array([t_sub_to_idx[sub] for sub in itertools.combinations(b, t)], dtype=np.int32)

    def fast_greedy(seed_val):
        random.seed(seed_val)
        covered = np.zeros(n_t, dtype=np.int32)
        uncovered_count = n_t
        blocks = []
        
        def gain(x, block):
            cnt = 0
            for rest in itertools.combinations(block, t - 1):
                sub = tuple(sorted(rest + (x,)))
                idx = t_sub_to_idx[sub]
                if covered[idx] == 0:
                    cnt += 1
            return cnt

        while uncovered_count > 0:
            while True:
                idx = random.randint(0, n_t - 1)
                if covered[idx] == 0:
                    start_sub = t_subs[idx]
                    break
            
            block = list(start_sub)
            block_set = set(block)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                candidates = [x for x in all_points if x not in block_set]
                random.shuffle(candidates)
                
                for x in candidates:
                    g = gain(x, block)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                
                if best_x is None or best_gain <= 0:
                    for x in candidates:
                        block.append(x)
                        block_set.add(x)
                        if len(block) == k:
                            break
                    break
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            idxs = block_t_indices[tuple(block)]
            new_covers = covered[idxs] == 0
            if np.any(new_covers):
                uncovered_count -= np.sum(new_covers)
            covered[idxs] += 1
        
        return blocks

    def local_search(blocks, max_iters=100):
        # 1. Remove redundant
        changed = True
        while changed:
            changed = False
            covered = np.zeros(n_t, dtype=np.int32)
            for b in blocks:
                covered[block_t_indices[tuple(b)]] += 1
            
            new_blocks = []
            for b in blocks:
                idxs = block_t_indices[tuple(b)]
                if np.all(covered[idxs] > 1):
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        
        # 2. Swaps
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Remove redundant again
            covered = np.zeros(n_t, dtype=np.int32)
            for b in blocks:
                covered[block_t_indices[tuple(b)]] += 1
            
            new_blocks = []
            for b in blocks:
                idxs = block_t_indices[tuple(b)]
                if np.all(covered[idxs] > 1):
                    improved = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            
            if not blocks:
                break

            # Try one random swap
            bi = random.randint(0, len(blocks) - 1)
            B = blocks[bi]
            pi = random.randint(0, len(B) - 1)
            p = B[pi]
            
            candidates = [x for x in all_points if x not in B]
            if not candidates:
                continue
            q = random.choice(candidates)
            
            B_new = B.copy()
            B_new[pi] = q
            B_new.sort()
            
            idxs_B = block_t_indices[tuple(B)]
            idxs_B_new = block_t_indices[tuple(B_new)]
            
            lost = np.setdiff1d(idxs_B, idxs_B_new)
            gained = np.setdiff1d(idxs_B_new, idxs_B)
            
            # Check validity: all lost subsets must be covered by others
            valid = True
            if len(lost) > 0:
                if np.any(covered[lost] <= 1):
                    valid = False
            
            if valid:
                # Update covered
                covered[lost] -= 1
                covered[gained] += 1
                
                # Check if B_new is redundant
                if np.all(covered[idxs_B_new] > 1):
                    # Remove B_new
                    covered[idxs_B_new] -= 1
                    blocks.pop(bi)
                    improved = True
                else:
                    # Revert
                    covered[lost] += 1
                    covered[gained] -= 1

        return blocks

    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        blocks = fast_greedy(i)
        
        if time.time() < deadline - 5:
            blocks = local_search(blocks, max_iters=50)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len <= 335:
            break

    return best_blocks
