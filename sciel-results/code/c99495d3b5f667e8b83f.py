"""Covering design C(19,8,5) - greedy + reduction."""
import itertools
import time
import random


def construct(v, k, t):
    deadline = time.time() + 105  # safety margin under 120s
    
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    cov = [0] * n_t
    uncovered_set = set(range(n_t))
    blocks = []
    
    # Greedy construction
    while uncovered_set and time.time() < deadline - 25:
        # Pick a random uncovered t-subset as seed
        seed_sub = random.choice(tuple(uncovered_set))
        seed_set = set(seed_sub)
        
        block = list(seed_sub)
        block_set = set(block)
        
        # Greedily add points to maximize coverage of uncovered t-subs
        while len(block) < k and time.time() < deadline - 25:
            best_x = None
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    s = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx.get(s)
                    if idx is not None and idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                available = [x for x in range(v) if x not in block_set]
                if not available:
                    break
                best_x = random.choice(available)
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update coverage
        for s in itertools.combinations(block, t):
            idx = t_sub_to_idx[s]
            cov[idx] += 1
            if cov[idx] == 1:
                uncovered_set.discard(idx)
    
    # Reduction phase: remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            block = blocks[i]
            redundant = True
            for s in itertools.combinations(block, t):
                idx = t_sub_to_idx[s]
                if cov[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for s in itertools.combinations(block, t):
                    idx = t_sub_to_idx[s]
                    cov[idx] -= 1
                del blocks[i]
                changed = True
    
    return [list(b) for b in blocks]
