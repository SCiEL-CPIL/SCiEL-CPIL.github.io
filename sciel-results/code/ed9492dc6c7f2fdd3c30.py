"""Covering design C(24,7,3) via greedy + deletion with efficient gain computation."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 115  # safety margin before 120s limit (evaluator kills at 150s)
    
    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    
    # Index each t-set for O(1) lookup
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Track coverage count for each t-set
    cover_count = [0] * num_tsets
    uncovered_count = num_tsets
    blocks = []
    
    def add_block(block):
        nonlocal uncovered_count
        block_set = set(block)
        for c in itertools.combinations(block, t):
            idx = tset_to_idx[c]
            if cover_count[idx] == 0:
                uncovered_count -= 1
            cover_count[idx] += 1
        blocks.append(block)
    
    def remove_block(block):
        nonlocal uncovered_count
        for c in itertools.combinations(block, t):
            idx = tset_to_idx[c]
            cover_count[idx] -= 1
            if cover_count[idx] == 0:
                uncovered_count += 1
        blocks.remove(block)
    
    # Greedy construction
    # For gain computation: when we have a partial block B (size b) and candidate x,
    # gain(x) = number of t-sets S where x in S, S \ {x} subset of B, and S is uncovered
    
    # Precompute all (t-1)-subsets of each possible partial block size
    # For efficiency, we compute gain on the fly but use the cover_count array
    
    while uncovered_count > 0 and time.time() < deadline:
        # Find an uncovered t-set to seed a new block
        seed_tset = None
        for i in range(num_tsets):
            if cover_count[i] == 0:
                seed_tset = all_tsets[i]
                break
        
        if seed_tset is None:
            break
        
        block = list(seed_tset)
        
        while len(block) < k and time.time() < deadline:
            block_set = set(block)
            best_x = -1
            best_gain = -1
            
            # For each candidate x not in block
            for x in range(v):
                if x in block_set:
                    continue
                # gain = number of (t-1)-subsets R of block such that {x}∪R is uncovered
                gain = 0
                for R in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(list(R) + [x]))
                    idx = tset_to_idx[ts]
                    if cover_count[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = random.choice(remaining)
                else:
                    break
            block.append(best_x)
        
        block.sort()
        add_block(block)
    
    # Deletion heuristic: try to remove blocks that are redundant
    # A block is redundant if removing it doesn't uncover any t-set
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            block = blocks[i]
            # Check if this block is the only cover for any t-set
            redundant = True
            for c in itertools.combinations(block, t):
                idx = tset_to_idx[c]
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                remove_block(block)
                improved = True
    
    return blocks
