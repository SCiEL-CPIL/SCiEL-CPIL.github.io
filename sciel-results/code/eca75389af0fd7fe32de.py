import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130
    
    all_tsubsets = list(itertools.combinations(range(v), t))
    tset_frozen = set(frozenset(ts) for ts in all_tsubsets)
    
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        seed = rng.randint(0, 10**9)
        local_rng = random.Random(seed)
        
        urg_bias = 0.3 + local_rng.random() * 2.0
        rar_bias = local_rng.random() * 1.0
        
        blocks = _greedy(v, k, t, tset_frozen, urg_bias, rar_bias, local_rng)
        if blocks is None:
            continue
        
        n = len(blocks)
        if n < best_count:
            best_count = n
            best_blocks = blocks
    
    if best_blocks is None:
        best_blocks = _greedy(v, k, t, tset_frozen, 1.0, 0.0, random.Random(0))
    
    return best_blocks


def _greedy(v, k, t, tset_frozen, urg_bias, rar_bias, rng):
    uncovered = set(tset_frozen)
    blocks = []
    point_usage = [0] * v
    
    while uncovered:
        # Pick most urgent uncovered t-subset
        # Urgency: prefer subsets with high-usage points (harder to cover)
        if len(uncovered) > 200:
            sample = rng.sample(list(uncovered), 200)
        else:
            sample = list(uncovered)
        
        best_urg = -1
        init_ts = None
        for ts in sample:
            u = sum(point_usage[p] for p in ts) + rng.random() * 0.1
            if u > best_urg:
                best_urg = u
                init_ts = ts
        
        if init_ts is None:
            init_ts = min(uncovered)
        
        # Start block with the t points from init_ts
        block = set(init_ts)
        for p in init_ts:
            point_usage[p] += 1
        
        # Greedily add remaining k-t points
        while len(block) < k:
            best_score = -1
            best_p = None
            
            for p in range(v):
                if p in block:
                    continue
                
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    cand = frozenset(rest) | {p}
                    if cand in uncovered:
                        gain += 1
                
                if gain <= 0:
                    continue
                
                rarity = (v - point_usage[p]) / v
                score = gain * urg_bias + rarity * rar_bias * 20 + rng.random() * 0.01
                
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p is None:
                unused = [p for p in range(v) if p not in block]
                if not unused:
                    break
                best_p = rng.choice(unused)
            
            block.add(best_p)
            point_usage[best_p] += 1
        
        blocks.append(sorted(block))
        
        for ts in itertools.combinations(block, t):
            uncovered.discard(frozenset(ts))
    
    return blocks
