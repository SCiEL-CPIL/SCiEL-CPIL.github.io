import itertools
import time
import random
from collections import defaultdict
import numpy as np

def _precompute_triples(v, t):
    triples = []
    triple_to_idx = {}
    for combo in itertools.combinations(range(v), t):
        idx = len(triples)
        triples.append(combo)
        triple_to_idx[combo] = idx
    return triples, triple_to_idx

def _precompute_pair_triples(v, t, triple_to_idx):
    pair_triples = {}
    for a in range(v):
        for b in range(a + 1, v):
            lst = []
            for c in range(v):
                if c == a or c == b:
                    continue
                tri = tuple(sorted((a, b, c)))
                if tri in triple_to_idx:
                    lst.append((c, triple_to_idx[tri]))
            pair_triples[(a, b)] = lst
    return pair_triples

def _greedy_fast(v, k, t, seed, triples, triple_to_idx, all_points, pair_triples):
    rng = random.Random(seed)
    n_triples = len(triples)
    
    uncovered = [True] * n_triples
    uncovered_count = n_triples
    blocks = []
    
    while uncovered_count > 0:
        while True:
            idx = rng.randint(0, n_triples - 1)
            if uncovered[idx]:
                break
        
        triple = triples[idx]
        block = list(triple)
        block_set = set(block)
        
        uncovered[idx] = False
        uncovered_count -= 1
        
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                gain = 0
                n_block = len(block)
                for i in range(n_block):
                    for j in range(i + 1, n_block):
                        a, b = block[i], block[j]
                        key = (a, b) if a < b else (b, a)
                        for third, tri_idx in pair_triples[key]:
                            if third == x and uncovered[tri_idx]:
                                gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            if candidates:
                next_point = rng.choice(candidates)
            else:
                remaining = [x for x in all_points if x not in block_set]
                next_point = rng.choice(remaining)
            
            block.append(next_point)
            block_set.add(next_point)
        
        block.sort()
        blocks.append(block)
        
        for tri in itertools.combinations(block, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            if uncovered[tri_idx]:
                uncovered[tri_idx] = False
                uncovered_count -= 1
        
        if uncovered_count == 0:
            break
    
    return blocks

def _build_coverage(blocks, triples, triple_to_idx):
    n_triples = len(triples)
    cov_count = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for tri in itertools.combinations(b, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            cov_count[tri_idx] += 1
    return cov_count

def _local_search_fast(v, k, t, blocks, triples, triple_to_idx, deadline):
    blocks = [list(b) for b in blocks]
    n_triples = len(triples)
    
    cov_count = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for tri in itertools.combinations(b, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            cov_count[tri_idx] += 1
    
    improved = True
    rng = random.Random(42)
    
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            b = blocks[i]
            block_triples = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
            
            valid = True
            for tri_idx in block_triples:
                if cov_count[tri_idx] <= 1:
                    valid = False
                    break
            
            if valid:
                for tri_idx in block_triples:
                    cov_count[tri_idx] -= 1
                blocks.pop(i)
                improved = True
                break
    
    return blocks

def _local_search_swap(v, k, t, blocks, triples, triple_to_idx, deadline):
    blocks = [list(b) for b in blocks]
    n_triples = len(triples)
    rng = random.Random(999)
    
    cov_count = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for tri in itertools.combinations(b, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            cov_count[tri_idx] += 1
    
    improved = True
    max_iters = 0
    
    while improved and time.time() < deadline:
        improved = False
        max_iters += 1
        if max_iters > 300:
            break
        
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() >= deadline:
                break
            b = blocks[i]
            block_triples = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
            if all(cov_count[ti] > 1 for ti in block_triples):
                for ti in block_triples:
                    cov_count[ti] -= 1
                blocks.pop(i)
                improved = True
        
        if not improved:
            for _ in range(30):
                if time.time() >= deadline:
                    break
                if len(blocks) == 0:
                    break
                    
                idx = rng.randint(0, len(blocks) - 1)
                old_x = rng.choice(blocks[idx])
                new_x = rng.randint(0, v - 1)
                
                if new_x == old_x or new_x in blocks[idx]:
                    continue
                
                old_block = blocks[idx]
                new_block = sorted([new_x] + [p for p in old_block if p != old_x])
                
                old_tris = set(itertools.combinations(old_block, t))
                new_tris = set(itertools.combinations(new_block, t))
                
                valid = True
                for tri in old_tris:
                    ti = triple_to_idx[tuple(sorted(tri))]
                    if cov_count[ti] <= 1 and tri not in new_tris:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                for tri in old_tris:
                    ti = triple_to_idx[tuple(sorted(tri))]
                    cov_count[ti] -= 1
                for tri in new_tris:
                    ti = triple_to_idx[tuple(sorted(tri))]
                    cov_count[ti] += 1
                blocks[idx] = new_block
                improved = True
                break
    
    for i in range(len(blocks) - 1, -1, -1):
        if time.time() >= deadline:
            break
        b = blocks[i]
        block_triples = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
        if all(cov_count[ti] > 1 for ti in block_triples):
            for ti in block_triples:
                cov_count[ti] -= 1
            blocks.pop(i)
    
    return blocks

def _column_generation_pricing(v, k, t, weights, triples, triple_to_idx, all_points, pair_triples, rng, max_blocks=20):
    """
    Pricing subproblem: find up to max_blocks 7-blocks that minimize weighted sum of uncovered triples.
    weights[i] = dual variable for triple i (higher = more important to cover)
    Returns list of blocks found.
    """
    found_blocks = []
    used_points = set()
    
    for _ in range(max_blocks):
        if time.time() > 115:
            break
        
        # Pick a random starting triple weighted by its weight
        total_w = sum(weights)
        if total_w == 0:
            break
        r = rng.random() * total_w
        cumulative = 0
        start_idx = 0
        for i in range(len(weights)):
            cumulative += weights[i]
            if cumulative >= r:
                start_idx = i
                break
        
        triple = triples[start_idx]
        block = list(triple)
        block_set = set(block)
        
        while len(block) < k:
            best_score = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Score: sum of weights of triples that would be covered by adding x
                score = 0.0
                n_block = len(block)
                for i in range(n_block):
                    for j in range(i + 1, n_block):
                        a, b = block[i], block[j]
                        key = (a, b) if a < b else (b, a)
                        for third, tri_idx in pair_triples[key]:
                            if third == x:
                                score += weights[tri_idx]
                
                if score > best_score:
                    best_score = score
                    candidates = [x]
                elif score == best_score:
                    candidates.append(x)
            
            if candidates:
                next_point = rng.choice(candidates)
            else:
                remaining = [x for x in all_points if x not in block_set]
                next_point = rng.choice(remaining)
            
            block.append(next_point)
            block_set.add(next_point)
        
        block.sort()
        found_blocks.append(block)
    
    return found_blocks

def _lp_rounding_and_repair(v, k, t, blocks, triples, triple_to_idx, deadline):
    """
    Column generation style: use LP duals to guide new block generation,
    then round and repair.
    """
    blocks = [list(b) for b in blocks]
    n_triples = len(triples)
    all_points = list(range(v))
    
    # Build coverage
    cov_count = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for tri in itertools.combinations(b, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            cov_count[tri_idx] += 1
    
    rng = random.Random(12345)
    
    # Iterative column generation
    for cg_iter in range(5):
        if time.time() >= deadline:
            break
        
        # Compute dual variables: weight = 1/max(coverage, 1) for uncovered, small for covered
        # This encourages covering rare triples
        weights = np.zeros(n_triples)
        for i in range(n_triples):
            if cov_count[i] == 0:
                weights[i] = 10.0
            elif cov_count[i] == 1:
                weights[i] = 1.0
            else:
                weights[i] = 0.1
        
        # Pricing: generate new blocks
        new_blocks = _column_generation_pricing(
            v, k, t, weights, triples, triple_to_idx, all_points, pair_triples_cache, rng, max_blocks=10
        )
        
        # Add new blocks to the set
        existing_set = set(tuple(b) for b in blocks)
        for nb in new_blocks:
            key = tuple(nb)
            if key not in existing_set:
                blocks.append(nb)
                existing_set.add(key)
                for tri in itertools.combinations(nb, t):
                    tri_idx = triple_to_idx[tuple(sorted(tri))]
                    cov_count[tri_idx] += 1
        
        # Now do a greedy selection: keep blocks in order of "value" (triples covered with low coverage)
        if len(blocks) > 69:
            # Score each block by sum of 1/coverage of its triples
            block_scores = []
            for b in blocks:
                score = 0.0
                for tri in itertools.combinations(b, t):
                    tri_idx = triple_to_idx[tuple(sorted(tri))]
                    if cov_count[tri_idx] > 0:
                        score += 1.0 / cov_count[tri_idx]
                    else:
                        score += 100.0
                block_scores.append((score, b))
            
            # Sort by score descending
            block_scores.sort(key=lambda x: -x[0])
            
            # Greedily select top blocks that maintain coverage
            selected = []
            temp_cov = np.zeros(n_triples, dtype=np.int32)
            for score, b in block_scores:
                if len(selected) >= 69:
                    break
                # Check if this block adds unique coverage
                adds = False
                for tri in itertools.combinations(b, t):
                    tri_idx = triple_to_idx[tuple(sorted(tri))]
                    if temp_cov[tri_idx] == 0:
                        adds = True
                        break
                if adds or len(selected) < 30:
                    selected.append(b)
                    for tri in itertools.combinations(b, t):
                        tri_idx = triple_to_idx[tuple(sorted(tri))]
                        temp_cov[tri_idx] += 1
            
            # Repair: add blocks to cover any remaining gaps
            uncovered_triples = [i for i in range(n_triples) if temp_cov[i] == 0]
            if uncovered_triples:
                for ti in uncovered_triples:
                    if time.time() >= deadline:
                        break
                    triple = triples[ti]
                    # Find a block from original set that covers this triple
                    added = False
                    for b in blocks:
                        if b not in selected:
                            tris_in_b = set(itertools.combinations(b, t))
                            if triple in tris_in_b:
                                selected.append(b)
                                for tri in itertools.combinations(b, t):
                                    tri_idx = triple_to_idx[tuple(sorted(tri))]
                                    temp_cov[tri_idx] += 1
                                added = True
                                break
                    if not added:
                        # Create a new block around this triple
                        new_block = list(triple)
                        new_set = set(new_block)
                        while len(new_block) < k:
                            best_x = -1
                            best_gain = -1
                            for x in all_points:
                                if x in new_set:
                                    continue
                                gain = 0
                                for tri2 in itertools.combinations(new_block, t-1):
                                    ts = tuple(sorted(tri2 + (x,)))
                                    if ts in triple_to_idx and temp_cov[triple_to_idx[ts]] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_x = x
                            if best_x >= 0:
                                new_block.append(best_x)
                                new_set.add(best_x)
                            else:
                                remaining = [x for x in all_points if x not in new_set]
                                if remaining:
                                    new_block.append(rng.choice(remaining))
                                    new_set.add(new_block[-1])
                        new_block.sort()
                        selected.append(new_block)
                        for tri in itertools.combinations(new_block, t):
                            tri_idx = triple_to_idx[tuple(sorted(tri))]
                            temp_cov[tri_idx] += 1
            
            # Verify coverage
            final_cov = np.zeros(n_triples, dtype=np.int32)
            for b in selected:
                for tri in itertools.combinations(b, t):
                    tri_idx = triple_to_idx[tuple(sorted(tri))]
                    final_cov[tri_idx] += 1
            
            if all(final_cov[i] > 0 for i in range(n_triples)):
                blocks = selected
            # Else keep original blocks
    
    return blocks

# Global cache for pair_triples
pair_triples_cache = None

def construct(v, k, t):
    global pair_triples_cache
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Pre-compute
    triples, triple_to_idx = _precompute_triples(v, t)
    all_points = list(range(v))
    pair_triples_cache = _precompute_pair_triples(v, t, triple_to_idx)
    pair_triples = pair_triples_cache
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        if time.time() + 8.0 > deadline:
            break
        
        blocks = _greedy_fast(v, k, t, seed, triples, triple_to_idx, all_points, pair_triples)
        
        # Apply local search
        if time.time() + 10.0 < deadline:
            blocks = _local_search_fast(v, k, t, blocks, triples, triple_to_idx, deadline)
        
        # Try swap-based local search
        if time.time() + 15.0 < deadline:
            blocks = _local_search_swap(v, k, t, blocks, triples, triple_to_idx, deadline)
        
        # Try column generation
        if time.time() + 20.0 < deadline:
            blocks_cg = _column_generation_pricing(
                v, k, t, np.ones(len(triples)), triples, triple_to_idx, all_points, pair_triples, 
                random.Random(seed + 1000), max_blocks=5
            )
            # Merge and re-optimize
            merged = [list(b) for b in blocks]
            existing_set = set(tuple(b) for b in merged)
            for nb in blocks_cg:
                key = tuple(nb)
                if key not in existing_set:
                    merged.append(nb)
                    existing_set.add(key)
            
            # Re-run local search on merged
            if time.time() + 10.0 < deadline:
                merged = _local_search_fast(v, k, t, merged, triples, triple_to_idx, deadline)
            
            if len(merged) < len(blocks):
                blocks = merged
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        if best_len == 69:
            break
        
        seed += 1
    
    return best_blocks
