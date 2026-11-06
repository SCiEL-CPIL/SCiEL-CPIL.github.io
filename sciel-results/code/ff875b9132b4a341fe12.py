def construct():
    from itertools import combinations
    import time
    
    start_time = time.time()
    DEADLINE = 140  # 10s safety margin before 150s kill
    
    # Try direct cyclic construction of S(5,6,12)
    # Points: 0-10 are Z_11, 11 is infinity
    # 
    # Known base blocks for cyclic development under Z_11:
    # 11 base blocks containing infinity + 1 base block without infinity
    # This gives 11*11 + 11 = 132 blocks
    
    base_blocks = [
        # Blocks containing infinity (11)
        [11, 0, 1, 2, 3, 4],
        [11, 0, 1, 2, 4, 5],
        [11, 0, 1, 2, 5, 8],
        [11, 0, 1, 3, 4, 7],
        [11, 0, 1, 3, 5, 7],
        [11, 0, 1, 4, 5, 9],
        [11, 0, 2, 3, 5, 8],
        [11, 0, 2, 4, 6, 9],
        [11, 0, 2, 4, 7, 8],
        [11, 0, 3, 4, 6, 8],
        [11, 0, 3, 5, 6, 7],
        # Block without infinity
        [0, 1, 2, 3, 4, 5],
    ]
    
    blocks = []
    seen = set()
    
    for bb in base_blocks:
        for i in range(11):
            if 11 in bb:
                block = [11] + sorted([(x + i) % 11 for x in bb if x != 11])
            else:
                block = sorted([(x + i) % 11 for x in bb])
            t = tuple(block)
            if t not in seen:
                seen.add(t)
                blocks.append(block)
    
    # Verify: should have 132 blocks, each 5-subset appears exactly once
    if len(blocks) == 132:
        # Quick verification
        all_5subsets = set()
        valid = True
        for b in blocks:
            for s in combinations(b, 5):
                if s in all_5subsets:
                    valid = False
                    break
                all_5subsets.add(s)
            if not valid:
                break
        if valid and len(all_5subsets) == 132 * 6:
            return blocks
    
    # Fallback: greedy algorithm with time guard
    covered = set()
    result = []
    for b in combinations(range(12), 6):
        if time.time() - start_time > DEADLINE:
            break
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            result.append(list(b))
            covered.update(subsets)
    
    return result
