"""Covering design C(24,7,3): fast greedy with bitset acceleration, local search, and SA polish."""
import itertools
import time
import random
from collections import defaultdict


def _build_greedy(v, k, t, uncovered, rng):
    """Build a greedy covering from the given uncovered set using efficient gain computation."""
    blocks = []
    uncovered = set(uncovered)
    
    while uncovered:
        # Pick a random uncovered triple to start
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        block_set = set(block)
        
        while len(block) < k:
            # Compute gain for each candidate point using bitset-like approach
            best_x = -1
            best_gain = -1
            
            # Pre-compute the (t-1)-combinations of current block
            rest_combos = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in rest_combos:
                    triple = frozenset(rest + (x,))
                    if triple in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                # Fallback: pick random point not in block
                candidates = [x for x in range(v) if x not in block_set]
                best_x = rng.choice(candidates)
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove all covered triples
        for triple in itertools.combinations(block, t):
            uncovered.discard(frozenset(triple))
    
    return blocks


def _prune_redundant_blocks(blocks, v, k, t, deadline):
    """Remove redundant blocks (those whose triples are all covered by other blocks)."""
    if not blocks:
        return blocks
    
    n = len(blocks)
    # Build coverage count
    triple_count = defaultdict(int)
    block_triples = []
    for b in blocks:
        triples = [frozenset(c) for c in itertools.combinations(b, t)]
        block_triples.append(triples)
        for tr in triples:
            triple_count[tr] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            redundant = True
            for tr in block_triples[i]:
                if triple_count[tr] <= 1:
                    redundant = False
                    break
            if redundant:
                for tr in block_triples[i]:
                    triple_count[tr] -= 1
                    if triple_count[tr] == 0:
                        del triple_count[tr]
                blocks.pop(i)
                block_triples.pop(i)
                n -= 1
                improved = True
    
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce the number of blocks via local search (redundancy + pair merge)."""
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    if n == 0:
        return blocks
    
    # Build coverage count
    triple_count = defaultdict(int)
    block_triples = []
    for b in blocks:
        triples = [frozenset(c) for c in itertools.combinations(b, t)]
        block_triples.append(triples)
        for tr in triples:
            triple_count[tr] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing each block
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            redundant = True
            for tr in block_triples[i]:
                if triple_count[tr] <= 1:
                    redundant = False
                    break
            if redundant:
                for tr in block_triples[i]:
                    triple_count[tr] -= 1
                    if triple_count[tr] == 0:
                        del triple_count[tr]
                blocks.pop(i)
                block_triples.pop(i)
                n -= 1
                improved = True
        
        # Try replacing two blocks with one
        if n >= 2 and time.time() < deadline:
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                for j in range(n - 1, i, -1):
                    if time.time() > deadline:
                        break
                    union_triples = set(block_triples[i]) | set(block_triples[j])
                    points = set()
                    for tr in union_triples:
                        points.update(tr)
                    if len(points) <= k:
                        new_block = sorted(points)
                        if len(new_block) < k:
                            for p in range(v):
                                if p not in new_block:
                                    new_block.append(p)
                                if len(new_block) == k:
                                    break
                        new_block = sorted(new_block[:k])
                        new_block_set = set(new_block)
                        covers_all = all(
                            set(tr).issubset(new_block_set) 
                            for tr in union_triples
                        )
                        if covers_all:
                            for tr in block_triples[i]:
                                triple_count[tr] -= 1
                                if triple_count[tr] == 0:
                                    del triple_count[tr]
                            for tr in block_triples[j]:
                                triple_count[tr] -= 1
                                if triple_count[tr] == 0:
                                    del triple_count[tr]
                            for tr in itertools.combinations(new_block, t):
                                triple_count[frozenset(tr)] += 1
                            blocks[i] = new_block
                            block_triples[i] = [frozenset(c) for c in itertools.combinations(new_block, t)]
                            blocks.pop(j)
                            block_triples.pop(j)
                            n -= 1
                            improved = True
                            break
                if improved:
                    break
    
    return blocks


def _sa_polish(blocks, v, k, t, deadline, rng):
    """Simulated annealing polish: try swapping points in blocks to enable removals."""
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    if n == 0:
        return blocks
    
    # Build coverage count
    triple_count = defaultdict(int)
    block_triples = []
    for b in blocks:
        triples = [frozenset(c) for c in itertools.combinations(b, t)]
        block_triples.append(triples)
        for tr in triples:
            triple_count[tr] += 1
    
    best_n = n
    best_blocks = [list(b) for b in blocks]
    
    temp = 1.0
    cooling = 0.995
    
    while time.time() < deadline and temp > 0.01:
        # Try random modifications
        if n == 0:
            break
        
        # Pick a random block and try to modify it
        i = rng.randrange(n)
        old_block = blocks[i][:]
        old_triples = block_triples[i][:]
        
        # Try swapping one point in block i with a point from outside
        candidates = [x for x in range(v) if x not in blocks[i]]
        if not candidates:
            temp *= cooling
            continue
        
        pos = rng.randrange(k)
        new_point = rng.choice(candidates)
        
        # Apply swap
        blocks[i][pos] = new_point
        blocks[i].sort()
        
        # Recompute triples for block i
        new_triples = [frozenset(c) for c in itertools.combinations(blocks[i], t)]
        
        # Check if this swap enables any block removal
        # First, update coverage counts
        for tr in old_triples:
            triple_count[tr] -= 1
            if triple_count[tr] == 0:
                del triple_count[tr]
        for tr in new_triples:
            triple_count[tr] += 1
        
        # Try to remove redundant blocks
        removed_any = False
        for j in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            redundant = True
            for tr in block_triples[j]:
                if triple_count[tr] <= 1:
                    redundant = False
                    break
            if redundant:
                for tr in block_triples[j]:
                    triple_count[tr] -= 1
                    if triple_count[tr] == 0:
                        del triple_count[tr]
                blocks.pop(j)
                block_triples.pop(j)
                n -= 1
                removed_any = True
        
        if removed_any and n < best_n:
            best_n = n
            best_blocks = [list(b) for b in blocks]
        else:
            # Revert
            for tr in new_triples:
                triple_count[tr] -= 1
                if triple_count[tr] == 0:
                    del triple_count[tr]
            for tr in old_triples:
                triple_count[tr] += 1
            blocks[i] = old_block
            blocks[i].sort()
            block_triples[i] = old_triples
        
        temp *= cooling
    
    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    best_blocks = None
    best_count = float('inf')
    
    # First run: deterministic greedy
    rng = random.Random(42)
    uncovered = set(all_tsets)
    blocks = _build_greedy(v, k, t, uncovered, rng)
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = [list(b) for b in blocks]
    
    # Local search on best solution
    if time.time() < deadline:
        ls_blocks = _local_search(best_blocks, v, k, t, deadline)
        if len(ls_blocks) < best_count:
            best_count = len(ls_blocks)
            best_blocks = ls_blocks
    
    # Randomized restarts
    attempt = 0
    while time.time() < deadline - 10:  # leave 10s buffer for SA
        attempt += 1
        rng = random.Random(attempt * 1000 + 7)
        uncovered = set(all_tsets)
        blocks = _build_greedy(v, k, t, uncovered, rng)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            # Try local search on this new best
            if time.time() < deadline - 10:
                ls_blocks = _local_search(best_blocks, v, k, t, deadline - 10)
                if len(ls_blocks) < best_count:
                    best_count = len(ls_blocks)
                    best_blocks = ls_blocks
    
    # SA polish on best solution
    if time.time() < deadline - 2:
        rng = random.Random(12345)
        sa_blocks = _sa_polish(best_blocks, v, k, t, deadline - 2, rng)
        if len(sa_blocks) < best_count:
            best_count = len(sa_blocks)
            best_blocks = sa_blocks
    
    return best_blocks
