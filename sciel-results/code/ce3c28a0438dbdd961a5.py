"""Covering design C(24,7,3): fast greedy with randomized restarts and enhanced local search."""
import itertools
import time
import random
from collections import defaultdict


def _greedy_build(v, k, t, uncovered_set, rng):
    """Build a greedy covering from the given uncovered set using randomized start."""
    blocks = []
    uncovered = set(uncovered_set)
    
    while uncovered:
        # Pick a random uncovered triple to start
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        block_set = set(block)
        
        while len(block) < k:
            # Compute gain for each candidate point
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered triples would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    triple = frozenset(rest + (x,))
                    if triple in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove all covered triples
        for triple in itertools.combinations(block, t):
            uncovered.discard(frozenset(triple))
    
    return [list(b) for b in blocks]


def _local_search(blocks, v, k, t, deadline):
    """Enhanced local search: remove redundant, merge 2 blocks, replace blocks."""
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    # Build coverage count and block-triple mapping
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
        
        # Phase 1: Remove redundant blocks
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
        
        # Phase 2: Try merging two blocks into one
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
        
        # Phase 3: Try replacing a block with a better one
        if time.time() < deadline:
            for i in range(n):
                if time.time() > deadline:
                    break
                # Find triples that are only covered by block i
                unique_triples = set()
                for tr in block_triples[i]:
                    if triple_count[tr] == 1:
                        unique_triples.add(tr)
                
                if not unique_triples:
                    continue
                
                # Try to find a new block that covers all unique_triples and more
                # Start with the points from unique_triples
                required_points = set()
                for tr in unique_triples:
                    required_points.update(tr)
                
                if len(required_points) > k:
                    continue
                
                # Build a candidate block starting from required_points
                candidate = sorted(required_points)
                candidate_set = set(candidate)
                
                # Add points to maximize coverage of uncovered/low-count triples
                while len(candidate) < k:
                    best_x = -1
                    best_gain = -1
                    for x in range(v):
                        if x in candidate_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(candidate, t - 1):
                            triple = frozenset(rest + (x,))
                            if triple_count.get(triple, 0) <= 1:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x == -1:
                        break
                    candidate.append(best_x)
                    candidate_set.add(best_x)
                
                candidate = sorted(candidate[:k])
                candidate_set = set(candidate)
                
                # Verify candidate covers all unique_triples
                if not all(set(tr).issubset(candidate_set) for tr in unique_triples):
                    continue
                
                # Check if replacement is beneficial: covers more low-count triples
                new_triples = [frozenset(c) for c in itertools.combinations(candidate, t)]
                
                # Count how many triples would have lower coverage after replacement
                benefit = 0
                for tr in new_triples:
                    if triple_count.get(tr, 0) <= 1:
                        benefit += 1
                
                # Only replace if we don't lose coverage
                old_triples_set = set(block_triples[i])
                for tr in old_triples_set:
                    if triple_count[tr] == 1 and tr not in set(new_triples):
                        benefit -= 10  # penalty for losing unique coverage
                
                if benefit > 0:
                    # Perform replacement
                    for tr in block_triples[i]:
                        triple_count[tr] -= 1
                        if triple_count[tr] == 0:
                            del triple_count[tr]
                    for tr in new_triples:
                        triple_count[tr] = triple_count.get(tr, 0) + 1
                    blocks[i] = candidate
                    block_triples[i] = new_triples
                    improved = True
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    best_blocks = None
    best_count = float('inf')
    
    # First run: deterministic-ish greedy
    rng = random.Random(42)
    uncovered = set(all_tsets)
    blocks = _greedy_build(v, k, t, uncovered, rng)
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = [list(b) for b in blocks]
    
    # Local search on first solution
    if time.time() < deadline - 10:
        ls_blocks = _local_search(best_blocks, v, k, t, deadline - 10)
        if len(ls_blocks) < best_count:
            best_count = len(ls_blocks)
            best_blocks = ls_blocks
    
    # Randomized restarts
    attempt = 0
    while time.time() < deadline - 15:  # leave 15s buffer
        attempt += 1
        rng = random.Random(attempt * 7919 + 13)
        uncovered = set(all_tsets)
        blocks = _greedy_build(v, k, t, uncovered, rng)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            # Local search on this new best
            if time.time() < deadline - 15:
                ls_blocks = _local_search(best_blocks, v, k, t, deadline - 15)
                if len(ls_blocks) < best_count:
                    best_count = len(ls_blocks)
                    best_blocks = ls_blocks
        elif len(blocks) <= best_count + 3:
            # Even if not immediately better, try local search
            if time.time() < deadline - 15:
                ls_blocks = _local_search(blocks, v, k, t, deadline - 15)
                if len(ls_blocks) < best_count:
                    best_count = len(ls_blocks)
                    best_blocks = ls_blocks
    
    return best_blocks
