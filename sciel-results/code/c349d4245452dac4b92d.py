def _try_2_for_1(blocks, cnt, coverage, rev_coverage, n_k, rng, deadline):
    n = len(blocks)
    if n < 2: return False, blocks, cnt
    
    # Try all pairs? N is ~150. 150*150/2 = 11250 pairs.
    # Each pair check involves:
    # 1. Decrement counts (35+35 ops)
    # 2. Find orphans (70 ops)
    # 3. If orphans, intersect lists (small)
    # 4. Revert counts (70 ops)
    # Total ~200 ops per pair. 11250 * 200 = 2.2M ops.
    # In Python, 2.2M ops is ~0.1-0.2s.
    # So we can do a full pass every 0.5s.
    
    for i in range(n):
        if _elapsed() > deadline: return False, blocks, cnt
        b1 = blocks[i]
        for j in range(i+1, n):
            b2 = blocks[j]
            
            # Decrement
            for idx in coverage[b1]: cnt[idx] -= 1
            for idx in coverage[b2]: cnt[idx] -= 1
            
            # Find orphans
            orphans = []
            # Check only affected
            affected = set(coverage[b1]) | set(coverage[b2])
            for idx in affected:
                if cnt[idx] == 0:
                    orphans.append(idx)
            
            if not orphans:
                # Remove both
                blocks.pop(j)
                blocks.pop(i)
                return True, blocks, cnt
            else:
                # Find candidate
                u0 = orphans[0]
                candidates = set(rev_coverage[u0])
                for u in orphans[1:]:
                    if not candidates: break
                    candidates.intersection_update(rev_coverage[u])
                
                if candidates:
                    b_new = next(iter(candidates))
                    if b_new != b1 and b_new != b2:
                        # Add b_new
                        for idx in coverage[b_new]: cnt[idx] += 1
                        blocks.pop(j)
                        blocks.pop(i)
                        blocks.append(b_new)
                        return True, blocks, cnt
                
            # Revert
            for idx in coverage[b1]: cnt[idx] += 1
            for idx in coverage[b2]: cnt[idx] += 1
            
    return False, blocks, cnt
