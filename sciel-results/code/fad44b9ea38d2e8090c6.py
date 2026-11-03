import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_points = set(range(v))
    
    def get_t_subsets(block):
        return set(itertools.combinations(block, t))

    def greedy_cover(uncovered_list, time_limit=None):
        blocks = []
        uncovered_set = set(uncovered_list)
        while uncovered_set:
            if time_limit and time.time() > time_limit:
                break
            # Pick random uncovered
            seed_t = random.choice(tuple(uncovered_set))
            block = list(seed_t)
            
            while len(block) < k:
                best_point = -1
                max_gain = -1
                for x in all_points - set(block):
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        new_t = tuple(sorted(rest + (x,)))
                        if new_t in uncovered_set:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = x
                if best_point == -1:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(tuple(block))
            for t_sub in get_t_subsets(block):
                uncovered_set.discard(t_sub)
        
        return blocks, uncovered_set

    def optimize(blocks):
        # 1. Remove redundancies
        # 2. Try merges
        # Return optimized blocks
        
        # Helper to remove redundant blocks
        def remove_redundant(blks):
            if not blks: return blks
            block_t_subsets = [get_t_subsets(B) for B in blks]
            coverage_count = {}
            for t_subs in block_t_subsets:
                for s in t_subs:
                    coverage_count[s] = coverage_count.get(s, 0) + 1
            
            new_blocks = []
            new_block_t_subsets = []
            for i, B in enumerate(blks):
                is_redundant = True
                for s in block_t_subsets[i]:
                    if coverage_count[s] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    for s in block_t_subsets[i]:
                        coverage_count[s] -= 1
                else:
                    new_blocks.append(B)
                    new_block_t_subsets.append(block_t_subsets[i])
            return new_blocks, new_block_t_subsets, coverage_count

        blocks, block_t_subsets, coverage_count = remove_redundant(blocks)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if i >= len(blocks): break
                for j in range(i + 1, len(blocks)):
                    if j >= len(blocks): break
                    
                    U = set()
                    for s in block_t_subsets[i]:
                        c = coverage_count[s]
                        c_after = c - 1
                        if s in block_t_subsets[j]:
                            c_after -= 1
                        if c_after == 0:
                            U.add(s)
                    for s in block_t_subsets[j]:
                        if s not in block_t_subsets[i]:
                            c = coverage_count[s]
                            c_after = c - 1
                            if c_after == 0:
                                U.add(s)
                    
                    if not U:
                        # Remove both
                        del blocks[j]
                        del block_t_subsets[j]
                        del blocks[i]
                        del block_t_subsets[i]
                        # Recompute coverage
                        coverage_count = {}
                        for t_subs in block_t_subsets:
                            for s in t_subs:
                                coverage_count[s] = coverage_count.get(s, 0) + 1
                        changed = True
                        break
                    else:
                        P_U = set()
                        for s in U:
                            P_U.update(s)
                        
                        if len(P_U) <= k:
                            remaining_points = list(all_points - P_U)
                            random.shuffle(remaining_points)
                            new_block = list(P_U) + remaining_points[:k - len(P_U)]
                            new_block.sort()
                            new_block_t = get_t_subsets(new_block)
                            
                            # Replace
                            del blocks[j]
                            del block_t_subsets[j]
                            del blocks[i]
                            del block_t_subsets[i]
                            
                            blocks.append(tuple(new_block))
                            block_t_subsets.append(new_block_t)
                            
                            coverage_count = {}
                            for t_subs in block_t_subsets:
                                for s in t_subs:
                                    coverage_count[s] = coverage_count.get(s, 0) + 1
                            
                            changed = True
                            break
                if changed:
                    break
            
            if changed:
                # Check for new redundancies
                blocks, block_t_subsets, coverage_count = remove_redundant(blocks)
        
        return blocks

    # Main
    uncovered_list = list(itertools.combinations(range(v), t))
    blocks, uncovered_set = greedy_cover(uncovered_list, time_limit=deadline - 10)
    
    # Finish if not done
    while uncovered_set:
        if time.time() > deadline - 2:
             # Emergency
             seed_t = random.choice(tuple(uncovered_set))
             block = list(seed_t)
             while len(block) < k:
                 avail = all_points - set(block)
                 block.append(random.choice(list(avail)))
             block.sort()
             blocks.append(tuple(block))
             for t_sub in get_t_subsets(block):
                 uncovered_set.discard(t_sub)
             break
        seed_t = random.choice(tuple(uncovered_set))
        block = list(seed_t)
        while len(block) < k:
            best_point = -1
            max_gain = -1
            for x in all_points - set(block):
                gain = 0
                for rest in itertools.combinations(block, t-1):
                    new_t = tuple(sorted(rest + (x,)))
                    if new_t in uncovered_set:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_point = x
            if best_point == -1: break
            block.append(best_point)
        block.sort()
        blocks.append(tuple(block))
        for t_sub in get_t_subsets(block):
            uncovered_set.discard(t_sub)

    # Optimize
    if time.time() < deadline - 5:
        blocks = optimize(blocks)

    return [list(b) for b in blocks]
