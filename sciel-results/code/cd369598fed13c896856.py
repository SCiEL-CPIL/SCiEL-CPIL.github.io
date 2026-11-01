import itertools
import random
import time

def construct(v, k, t):
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {frozenset(tr): i for i, tr in enumerate(triples)}
    all_indices = set(range(len(triples)))
    
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(all_indices)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple
            start_idx = random.choice(list(uncovered))
            start_triple = triples[start_idx]
            block = list(start_triple)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block]
                random.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    for u, w in itertools.combinations(block, t-1):
                        tr = frozenset((x, u, w))
                        idx = triple_to_idx[tr]
                        if idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            for tr in itertools.combinations(block, t):
                idx = triple_to_idx[frozenset(tr)]
                if idx in uncovered:
                    uncovered.remove(idx)
        
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        cover_count = [0] * len(triples)
        block_triples = []
        
        for b in blocks:
            b_trs = []
            for tr in itertools.combinations(b, t):
                idx = triple_to_idx[frozenset(tr)]
                b_trs.append(idx)
                cover_count[idx] += 1
            block_triples.append(b_trs)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                redundant = True
                for idx in block_triples[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for idx in block_triples[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_triples.pop(i)
                    changed = True
                    break
        
        return blocks

    start_time = time.time()
    best_blocks = None
    best_len = float('inf')
    
    while time.time() - start_time < 110:
        seed = random.randint(0, 10**9)
        blocks = greedy_construct(seed)
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    return best_blocks
