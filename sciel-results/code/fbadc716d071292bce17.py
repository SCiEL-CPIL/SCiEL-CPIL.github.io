import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # Precompute block_triple_indices and triple_to_blocks
    block_triple_indices = [None] * n_blocks
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        indices = []
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            indices.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = indices
        
    _CACHE[key] = (all_blocks, block_triple_indices, triple_to_blocks, n_triples)
    return _CACHE[key]

def _probabilistic_greedy(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, n_triples, seed, deadline):
    """
    Probabilistic greedy based on Lovasz Local Lemma ideas:
    - Weight uncovered triples by how "hard" they are to cover (fewer blocks cover them).
    - For each step, sample a target triple with probability proportional to its weight.
    - Among blocks covering that triple, pick one that maximizes the expected number of new triples covered,
      with a penalty for high overlap with already-selected blocks (to minimize redundancy).
    """
    rng = random.Random(seed)
    
    # Initial weights: 1 / (number of blocks covering the triple)
    # Triples with fewer covering blocks are "harder"
    weights = [1.0 / max(1, len(triple_to_blocks[i])) for i in range(n_triples)]
    
    uncovered = set(range(n_triples))
    selected_blocks = []
    selected_set = set()
    
    # For overlap penalty, track how many times each triple is covered
    cover_count = [0] * n_triples
    
    start = time.time()
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Sample a target triple with probability proportional to weight
        # Use reservoir-style or direct sampling for efficiency
        # For n_triples ~ 2024, direct sampling is fine
        total_weight = sum(weights[i] for i in uncovered)
        if total_weight <= 0:
            break
            
        r = rng.random() * total_weight
        cumulative = 0
        target = None
        for i in sorted(uncovered):  # sorted for determinism in case of ties? No, rng handles it
            cumulative += weights[i]
            if cumulative >= r:
                target = i
                break
        if target is None:
            target = next(iter(uncovered))
            
        # Get candidate blocks covering this triple
        candidates = triple_to_blocks[target]
        if not candidates:
            break
            
        # Evaluate each candidate block
        # Score = gain (new triples) - lambda * overlap (triples already covered)
        # We want to maximize gain and minimize overlap
        best_block = None
        best_score = -float('inf')
        
        # Sample up to 50 candidates if there are many
        if len(candidates) > 50:
            sampled = rng.sample(candidates, 50)
        else:
            sampled = candidates
            
        for b_idx in sampled:
            if b_idx in selected_set:
                continue
                
            gain = 0
            overlap = 0
            for t_idx in block_triple_indices[b_idx]:
                if t_idx in uncovered:
                    gain += 1
                else:
                    overlap += 1
                    
            # Score: gain is primary, overlap is secondary penalty
            # Use a smooth surrogate: score = gain - 0.1 * overlap
            score = gain - 0.1 * overlap
            
            if score > best_score:
                best_score = score
                best_block = b_idx
                
        if best_block is None:
            # Fallback: pick first available
            for b_idx in candidates:
                if b_idx not in selected_set:
                    best_block = b_idx
                    break
            if best_block is None:
                break
                
        # Select the block
        selected_blocks.append(best_block)
        selected_set.add(best_block)
        
        # Update uncovered and cover_count
        for t_idx in block_triple_indices[best_block]:
            if t_idx in uncovered:
                uncovered.remove(t_idx)
            cover_count[t_idx] += 1
            
        # Update weights: triples that are still uncovered but now have higher cover_count
        # from other blocks are "easier", so lower their weight
        # But we only track uncovered ones
        # Actually, the weight should reflect the remaining difficulty.
        # If a triple is still uncovered, its weight is 1/len(triple_to_blocks[tri])
        # which is static. To make it dynamic, we could adjust based on how many
        # of its covering blocks have already been selected (if selected, they don't help).
        # But for simplicity, keep static weights for now.
        
    return selected_blocks

def _repair_and_polish(v, k, t, initial_blocks, block_triple_indices, triple_to_blocks, n_triples, deadline):
    """
    Repair phase: ensure all triples are covered.
    Polish phase: remove redundant blocks and perform 1-exchanges to reduce count.
    """
    current_blocks = list(initial_blocks)
    triple_counts = [0] * n_triples
    
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
            
    # Repair: cover any uncovered triples
    uncovered = [i for i in range(n_triples) if triple_counts[i] == 0]
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick the most constrained uncovered triple
        target = min(uncovered, key=lambda x: len(triple_to_blocks[x]))
        
        # Find a block covering it
        candidates = [b for b in triple_to_blocks[target] if b not in set(current_blocks)]
        if not candidates:
            # Must add a block even if it overlaps
            candidates = triple_to_blocks[target]
            if not candidates:
                break
            # Pick the one with max gain
            best = max(candidates, key=lambda b: sum(1 for t in block_triple_indices[b] if t in uncovered))
        else:
            best = max(candidates, key=lambda b: sum(1 for t in block_triple_indices[b] if t in uncovered))
            
        current_blocks.append(best)
        for idx in block_triple_indices[best]:
            triple_counts[idx] += 1
            
        # Update uncovered
        uncovered = [i for i in uncovered if triple_counts[i] == 0]
        
    if time.time() > deadline:
        return current_blocks
        
    # Polish: iterative removal of redundant blocks and exchanges
    best_len = len(current_blocks)
    best_blocks = list(current_blocks)
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Phase 1: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_out = current_blocks[i]
                redundant = True
                for idx in block_triple_indices[b_out]:
                    if triple_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    del current_blocks[i]
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] -= 1
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    break
                    
        if not current_blocks:
            break
            
        # Phase 2: Try 1-exchange
        # Pick a block with few unique triples (easy to remove)
        unique_counts = []
        for i, b in enumerate(current_blocks):
            unique = sum(1 for idx in block_triple_indices[b] if triple_counts[idx] == 1)
            unique_counts.append(unique)
            
        # Sort by unique count
        indices = sorted(range(len(current_blocks)), key=lambda i: unique_counts[i])
        
        # Try top 3 candidates
        tried = False
        for pos in indices[:3]:
            if pos >= len(current_blocks):
                continue
                
            b_out = current_blocks[pos]
            critical = [idx for idx in block_triple_indices[b_out] if triple_counts[idx] == 1]
            
            if not critical:
                # Redundant, remove
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = list(current_blocks)
                tried = True
                break
                
            # Find a replacement
            min_tri = min(critical, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_tri])
            for tri in critical:
                if tri != min_tri:
                    candidates.intersection_update(triple_to_blocks[tri])
                    if not candidates:
                        break
                        
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if candidates:
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                best_cand = None
                best_score = -1
                for b_in in cand_list[:10]:
                    # Score: number of critical triples it covers (should be all) + bonus for covering other low-count triples
                    score = 0
                    for idx in block_triple_indices[b_in]:
                        if triple_counts[idx] == 0:
                            score += 100
                        elif triple_counts[idx] == 1:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = b_in
                        
                if best_cand is not None:
                    # Perform exchange
                    del current_blocks[pos]
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] -= 1
                    current_blocks.append(best_cand)
                    for idx in block_triple_indices[best_cand]:
                        triple_counts[idx] += 1
                    tried = True
                    break
                    
        if not tried:
            break
            
    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_blocks, block_triple_indices, triple_to_blocks, n_triples = _precompute(v, k, t)
    
    best_blocks_global = None
    best_len_global = float('inf')
    
    # Run multiple iterations with different seeds
    seed = 0
    while time.time() < deadline - 5:
        remaining = deadline - time.time()
        if remaining < 3:
            break
            
        # Allocate time: 40% for greedy, 60% for polish
        greedy_deadline = time.time() + remaining * 0.4
        polish_deadline = deadline - 2
        
        greedy_blocks = _probabilistic_greedy(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, n_triples, seed, greedy_deadline)
        seed += 1
        
        if not greedy_blocks:
            continue
            
        polished_blocks = _repair_and_polish(v, k, t, greedy_blocks, block_triple_indices, triple_to_blocks, n_triples, polish_deadline)
        
        if len(polished_blocks) < best_len_global:
            best_len_global = len(polished_blocks)
            best_blocks_global = polished_blocks
            
    if best_blocks_global is None:
        # Fallback
        greedy_blocks = _probabilistic_greedy(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, n_triples, 999, deadline - 2)
        best_blocks_global = _repair_and_polish(v, k, t, greedy_blocks, block_triple_indices, triple_to_blocks, n_triples, deadline)
        
    result = [list(all_blocks[idx]) for idx in best_blocks_global]
    return result
