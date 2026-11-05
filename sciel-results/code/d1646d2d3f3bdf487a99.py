import itertools
import time
import random

def construct(v, k, t):
    deadline = time.time() + 110
    
    all_points = list(range(v))
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    def block_to_triples(block):
        return [triple_to_idx[tr] for tr in itertools.combinations(block, t)]
    
    # Greedy initialization
    uncovered = bytearray([1] * n_triples)
    uncovered_count = n_triples
    blocks = []
    block_triples = []
    
    while uncovered_count > 0 and time.time() < deadline:
        idx = -1
        for i in range(n_triples):
            if uncovered[i]:
                idx = i
                break
        if idx == -1:
            break
        
        seed_triple = all_triples[idx]
        block = list(seed_triple)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in all_points:
                if x in block:
                    continue
                gain = 0
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        tr = tuple(sorted((block[i], block[j], x)))
                        ti = triple_to_idx.get(tr)
                        if ti is not None and uncovered[ti]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                break
            block.append(best_point)
        
        block.sort()
        bt = block_to_triples(block)
        for ti in bt:
            if uncovered[ti]:
                uncovered[ti] = 0
                uncovered_count -= 1
        blocks.append(block)
        block_triples.append(bt)
    
    # Reduction
    cover_count = [0] * n_triples
    for bt in block_triples:
        for ti in bt:
            cover_count[ti] += 1
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for bi in range(len(blocks)):
            if time.time() > deadline:
                break
            redundant = True
            for ti in block_triples[bi]:
                if cover_count[ti] <= 1:
                    redundant = False
                    break
            if redundant:
                for ti in block_triples[bi]:
                    cover_count[ti] -= 1
                blocks.pop(bi)
                block_triples.pop(bi)
                changed = True
                break
    
    # Local search
    best_blocks = list(blocks)
    best_block_triples = list(block_triples)
    best_count = len(blocks)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        
        cur_blocks = list(best_blocks)
        cur_block_triples = list(best_block_triples)
        cur_cover = [0] * n_triples
        for bt in cur_block_triples:
            for ti in bt:
                cur_cover[ti] += 1
        
        single_triples = [ti for ti in range(n_triples) if cur_cover[ti] == 1]
        if not single_triples:
            break
        
        best_candidate = None
        best_score = 0
        
        for _ in range(2000):
            if time.time() > deadline:
                break
            st = random.choice(single_triples)
            triple = all_triples[st]
            remaining = [x for x in all_points if x not in triple]
            extra = random.sample(remaining, k - t)
            candidate = tuple(sorted(triple + extra))
            
            score = 0
            for tr in itertools.combinations(candidate, t):
                ti = triple_to_idx[tr]
                if cur_cover[ti] == 1:
                    score += 1
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        if best_candidate is None or best_score < 5:
            break
        
        cur_blocks.append(list(best_candidate))
        cur_bt = block_to_triples(best_candidate)
        cur_block_triples.append(cur_bt)
        for ti in cur_bt:
            cur_cover[ti] += 1
        
        reduced = True
        while reduced:
            reduced = False
            for bi in range(len(cur_blocks)):
                if time.time() > deadline:
                    break
                redundant = True
                for ti in cur_block_triples[bi]:
                    if cur_cover[ti] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ti in cur_block_triples[bi]:
                        cur_cover[ti] -= 1
                    cur_blocks.pop(bi)
                    cur_block_triples.pop(bi)
                    reduced = True
                    break
        
        if len(cur_blocks) < best_count:
            best_count = len(cur_blocks)
            best_blocks = list(cur_blocks)
            best_block_triples = list(cur_block_triples)
    
    return [list(b) for b in best_blocks]
