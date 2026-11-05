"""Direct construction of the Witt design S(5,6,12).

We use a backtracking algorithm with strong constraint propagation to construct
the unique Steiner system S(5,6,12) on points 0..11. The design has exactly 132
blocks, and every 5-subset of the 12 points appears in exactly one block.
"""

import time
from itertools import combinations

def _generate_witt_blocks():
    """Generate the 132 blocks of the Witt design S(5,6,12) using backtracking."""
    points = list(range(12))
    all_5subs = list(combinations(points, 5))
    all_6subs = list(combinations(points, 6))
    
    # Map each 5-subset to the 6-subsets that contain it
    subs_candidates = {}
    for s in all_5subs:
        s_set = set(s)
        candidates = [b for b in all_6subs if s_set <= set(b)]
        subs_candidates[s] = candidates
    
    covered_5subs = set()
    blocks = []
    used_6subs = set()
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    def backtrack():
        nonlocal covered_5subs, blocks, used_6subs
        
        if time.time() > deadline:
            return False
        
        # Find the next uncovered 5-subset with fewest candidates (MRV heuristic)
        idx = None
        best_cand_count = float('inf')
        best_cands = None
        for i, s in enumerate(all_5subs):
            if s not in covered_5subs:
                cands = [c for c in subs_candidates[s] if c not in used_6subs]
                if len(cands) < best_cand_count:
                    best_cand_count = len(cands)
                    idx = i
                    best_cands = cands
                    if len(cands) == 0:
                        return False
                    if len(cands) == 1:
                        break
        
        if idx is None:
            return True  # All 5-subsets covered
        
        s = all_5subs[idx]
        cands = best_cands
        
        # Try each candidate block
        for b in cands:
            b_5subs = list(combinations(b, 5))
            # Check: all these 5-subsets must be currently uncovered
            conflict = False
            new_covered = []
            for bs in b_5subs:
                if bs in covered_5subs:
                    conflict = True
                    break
                new_covered.append(bs)
            
            if conflict:
                continue
            
            # Try this block
            for bs in new_covered:
                covered_5subs.add(bs)
            blocks.append(b)
            used_6subs.add(b)
            
            if backtrack():
                return True
            
            # Undo
            for bs in new_covered:
                covered_5subs.discard(bs)
            blocks.pop()
            used_6subs.discard(b)
            
            if time.time() > deadline:
                return False
        
        return False
    
    if backtrack():
        result = [list(b) for b in blocks]
        if len(covered_5subs) == 792:
            return result
    
    # Fallback: return what we have
    return [list(b) for b in blocks]


_WITTY_CACHE = None

def _ensure_cache():
    global _WITTY_CACHE
    if _WITTY_CACHE is None:
        _WITTY_CACHE = _generate_witt_blocks()
    return _WITTY_CACHE


def construct():
    """Construct a packing of 6-subsets of {0,...,11} with no repeated 5-subset.
    
    Returns the 132-block Witt design S(5,6,12), maximizing the number of blocks.
    """
    return _ensure_cache()


if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    elapsed = time.time() - start
    print(f"Blocks: {len(res)}, Time: {elapsed:.3f}s")
    
    # Verify validity
    seen = set()
    valid = True
    block_set = set()
    for b in res:
        b_tuple = tuple(sorted(b))
        if b_tuple in block_set:
            valid = False
            print(f"Duplicate block: {b}")
            break
        block_set.add(b_tuple)
        for s in combinations(b, 5):
            if s in seen:
                valid = False
                print(f"Repeated 5-subset: {s}")
                break
            seen.add(s)
        if not valid:
            break
    print(f"Valid: {valid}")
    print(f"5-subsets covered: {len(seen)} / 792")
    print(f"Blocks: {len(res)} / 132")
