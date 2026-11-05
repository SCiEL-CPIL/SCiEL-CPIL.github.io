import itertools
import random
import time

def construct(v, k, t):
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def greedy_randomized(seed):
        rng = random.Random(seed)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset to start
            start = tuple(rng.choice(list(uncovered)))
            block = list(start)
            while len(block) < k:
                best_x = None
                best_gain = -1
                best_tie = -1
                candidates = [x for x in range(v) if x not in block]
                rng.shuffle(candidates)
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        if tuple(sorted(rest + (x,))) in uncovered:
                            gain += 1
                    if gain > best_gain or (gain == best_gain and rng.random() < 0.5):
                        best_gain = gain
                        best_x = x
                if best_x is not None:
                    block.append(best_x)
            block.sort()
            blocks.append(tuple(block))
            for comb in itertools.combinations(block, t):
                uncovered.discard(comb)
        return [list(b) for b in blocks]
    
    def local_search(blocks, seed):
        rng = random.Random(seed)
        blocks = [list(b) for b in blocks]
        uncovered = set(itertools.combinations(range(v), t))
        for block in blocks:
            for comb in itertools.combinations(block, t):
                uncovered.discard(comb)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove a block if it's redundant
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break
                block = blocks[i]
                # Check if all t-subsets of this block are covered by other blocks
                redundant = True
                for comb in itertools.combinations(block, t):
                    covered = False
                    for j, other in enumerate(blocks):
                        if j == i:
                            continue
                        if all(c in other for c in comb):
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    improved = True
                    break
            
            # Try swapping points in blocks
            if not improved and time.time() < deadline:
                for i in range(len(blocks)):
                    if time.time() > deadline:
                        break
                    for j in range(k):
                        if time.time() > deadline:
                            break
                        old_pt = blocks[i][j]
                        best_new_pt = None
                        best_score = -1
                        candidates = [x for x in range(v) if x not in blocks[i]]
                        rng.shuffle(candidates)
                        for new_pt in candidates:
                            # Compute how many new t-subsets this swap covers
                            new_block = blocks[i][:]
                            new_block[j] = new_pt
                            new_block.sort()
                            new_covered = 0
                            for comb in itertools.combinations(new_block, t):
                                if comb in uncovered:
                                    new_covered += 1
                            # Penalize overlap with other blocks
                            overlap = 0
                            for comb in itertools.combinations(new_block, t):
                                for other in blocks:
                                    if other is blocks[i]:
                                        continue
                                    if all(c in other for c in comb):
                                        overlap += 1
                                        break
                            score = new_covered - overlap * 0.1
                            if score > best_score:
                                best_score = score
                                best_new_pt = new_pt
                        if best_new_pt is not None and best_score > 0:
                            blocks[i][j] = best_new_pt
                            blocks[i].sort()
                            # Update uncovered
                            for comb in itertools.combinations(blocks[i], t):
                                uncovered.discard(comb)
                            improved = True
                            break
                    if improved:
                        break
        return blocks
    
    best_blocks = None
    best_len = float('inf')
    seed = 0
    while time.time() < deadline:
        blocks = greedy_randomized(seed)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
        
        # Apply local search
        if time.time() < deadline - 10:
            improved = local_search(blocks, seed + 1000)
            if len(improved) < best_len:
                best_blocks = improved
                best_len = len(improved)
        
        seed += 1
    
    return best_blocks if best_blocks is not None else greedy_randomized(0)
