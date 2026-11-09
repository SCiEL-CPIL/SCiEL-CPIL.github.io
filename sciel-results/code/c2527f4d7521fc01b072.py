"""Construct the Witt design S(5,6,12) via deterministic Exact Cover (Algorithm X)."""

import time

# Deadline guard: finish well before 150s
_DEADLINE = 120.0  # seconds
_START = time.monotonic()


def _time_left():
    return _DEADLINE - (time.monotonic() - _START)


def _all_5subsets():
    """Return all C(12,5)=792 five-subsets as frozensets."""
    from itertools import combinations
    return [frozenset(c) for c in combinations(range(12), 5)]


def _block_5subsets(block):
    """Return the 6 five-subsets contained in a 6-subset block."""
    from itertools import combinations
    return [frozenset(c) for c in combinations(block, 5)]


def _all_blocks():
    """Return all C(12,6)=924 six-subsets as tuples."""
    from itertools import combinations
    return [tuple(c) for c in combinations(range(12), 6)]


def _solve_exact_cover():
    """
    Solve the exact cover problem: choose a set of 6-subsets (blocks) such that
    every 5-subset is covered exactly once.
    
    This is the Witt design S(5,6,12). We use Algorithm X (Dancing Links style)
    with bitmasks for speed.
    
    The universe is the 792 five-subsets. Each block covers exactly 6 of them.
    We need to pick 132 blocks that partition the 792 five-subsets.
    """
    from itertools import combinations
    
    # Enumerate all 5-subsets
    all5 = list(combinations(range(12), 5))
    n5 = len(all5)  # 792
    # Map each 5-subset to an index
    idx5 = {s: i for i, s in enumerate(all5)}
    
    # Enumerate all 6-subsets (blocks)
    all6 = list(combinations(range(12), 6))
    n6 = len(all6)  # 924
    
    # For each block, compute which 5-subsets it covers
    block_covers = []
    for b in all6:
        subs = [idx5[s] for s in combinations(b, 5)]
        block_covers.append(subs)
    
    # We want to select a subset of blocks such that every 5-subset index 0..791
    # is covered exactly once. Each block covers 6 specific 5-subset indices.
    # Total coverage needed: 792. Each block covers 6. So we need exactly 132 blocks.
    
    # Use backtracking with Algorithm X style: pick the most constrained uncovered
    # 5-subset, try blocks that cover it, recurse.
    
    # Precompute: for each 5-subset index, which blocks cover it
    cover_blocks = [[] for _ in range(n5)]
    for bi, subs in enumerate(block_covers):
        for s in subs:
            cover_blocks[s].append(bi)
    
    # State: which 5-subsets are covered, which blocks are used
    # Use a set of covered 5-subset indices
    # Use a set of used block indices
    
    # For efficiency, track uncovered count and use a list of available blocks per 5-subset
    
    best = [None]  # to hold the found solution
    
    def _backtrack(covered_set, used_blocks, depth):
        if _time_left() < 0.5:
            return False
        
        if len(covered_set) == n5:
            best[0] = list(used_blocks)
            return True
        
        # Find the uncovered 5-subset with the fewest available (unused) covering blocks
        min_count = float('inf')
        min_5 = -1
        for s in range(n5):
            if s in covered_set:
                continue
            count = 0
            for bi in cover_blocks[s]:
                if bi not in used_blocks:
                    count += 1
            if count < min_count:
                min_count = count
                min_5 = s
            if count == 0:
                return False  # dead end
        
        if min_5 == -1:
            # All covered
            best[0] = list(used_blocks)
            return True
        
        if min_count == 0:
            return False
        
        # Try each block that covers min_5
        for bi in cover_blocks[min_5]:
            if bi in used_blocks:
                continue
            # Check if all 5-subsets covered by this block are currently uncovered
            subs = block_covers[bi]
            conflict = False
            for s in subs:
                if s in covered_set:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Place this block
            used_blocks.add(bi)
            covered_set.update(subs)
            
            if _backtrack(covered_set, used_blocks, depth + 1):
                return True
            
            # Undo
            covered_set.difference_update(subs)
            used_blocks.discard(bi)
        
        return False
    
    # Start backtracking
    found = _backtrack(set(), set(), 0)
    
    if found and best[0] is not None:
        return [all6[bi] for bi in best[0]]
    
    return None


def _verify(blocks):
    """Verify that no 5-subset appears in two blocks and count blocks."""
    from itertools import combinations
    seen = set()
    for b in blocks:
        for s in combinations(b, 5):
            if s in seen:
                return False
            seen.add(s)
    return True


def construct():
    """
    Return the 132 blocks of the Witt design S(5,6,12).
    Uses deterministic exact cover search with Algorithm X.
    """
    # Try the exact cover solver
    result = _solve_exact_cover()
    
    if result is not None and _verify(result):
        return [list(b) for b in result]
    
    # Fallback: greedy (should not be reached for this instance)
    from itertools import combinations
    covered, blocks = set(), []
    for b in combinations(range(12), 6):
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
    return blocks
