"""Portfolio search for covering design C(24,7,3) using Max-SAT formulation."""
import itertools
import random
import time
import sys
from collections import defaultdict, Counter
from typing import List, Set, Tuple

# EVOLVE-BLOCK-START

def _precompute_block_tsets(v, k, t):
    """Precompute all t-subsets for every k-subset of v points."""
    block_tsets = {}
    for block in itertools.combinations(range(v), k):
        block_tsets[block] = list(itertools.combinations(block, t))
    return block_tsets


def _walksat_covering(v, k, t, deadline, seed=42):
    """
    Max-SAT / WalkSAT-style local search for covering design.
    Variables: select each block (7-subset) as True/False.
    Clauses: each t-set must be covered by at least one selected block.
    Objective: minimize number of True variables (selected blocks).
    
    We use a two-phase approach:
    Phase 1: Build an initial covering using greedy (fast).
    Phase 2: Use WalkSAT-style moves to reduce block count while maintaining coverage.
    """
    rng = random.Random(seed)
    
    # Precompute all t-sets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    
    # Build a map: for each t-set, which blocks contain it
    tset_to_blocks = defaultdict(list)
    all_blocks = list(itertools.combinations(range(v), k))
    for bi, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            tset_to_blocks[ts].append(bi)
    
    # Phase 1: Greedy initial covering
    uncovered = set(all_tsets)
    selected_blocks = []
    
    # Compute gain for each block
    def compute_block_gain(block):
        """Number of uncovered t-sets in this block."""
        return sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
    
    while uncovered and time.time() < deadline:
        best_block = None
        best_gain = -1
        
        # Sample candidate blocks for speed
        candidates = rng.sample(all_blocks, min(500, len(all_blocks)))
        for block in candidates:
            gain = compute_block_gain(block)
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None or best_gain <= 0:
            # Fallback: try all blocks
            for block in all_blocks:
                gain = compute_block_gain(block)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
        
        if best_block is None:
            break
        
        selected_blocks.append(best_block)
        for ts in itertools.combinations(best_block, t):
            uncovered.discard(ts)
        
        if time.time() > deadline:
            break
    
    if not selected_blocks:
        return []
    
    # Convert to list of lists
    selected = [list(b) for b in selected_blocks]
    
    # Phase 2: WalkSAT-style local search to reduce block count
    # Build coverage index: tset -> list of block indices
    def build_coverage_index(blocks):
        cov = defaultdict(list)
        for i, b in enumerate(blocks):
            bset = set(b)
            for ts in itertools.combinations(b, t):
                cov[ts].append(i)
        return cov
    
    coverage = build_coverage_index(selected)
    
    def is_valid(blocks):
        """Check if every t-set is covered."""
        cov = build_coverage_index(blocks)
        for ts in all_tsets:
            if not cov.get(ts):
                return False
        return True
    
    # Local search: try to remove blocks, then add better ones
    deadline_ls = time.time() + (deadline - time.time()) * 0.5
    
    improved = True
    while improved and time.time() < deadline_ls:
        improved = False
        n = len(selected)
        
        # Try removing each block
        for i in range(n):
            if time.time() >= deadline_ls:
                break
            
            # Check if block i is redundant (all its t-sets covered by other blocks)
            redundant = True
            for ts in itertools.combinations(selected[i], t):
                covering = coverage.get(ts, [])
                if covering == [i] or (len(covering) == 1 and covering[0] == i):
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for ts in itertools.combinations(selected[i], t):
                    if i in coverage.get(ts, []):
                        coverage[ts].remove(i)
                selected.pop(i)
                # Rebuild index after removal
                coverage = build_coverage_index(selected)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove a block and add a different one
        for i in range(n):
            if time.time() >= deadline_ls:
                break
            
            # Find critical t-sets (only covered by block i)
            critical = []
            for ts in itertools.combinations(selected[i], t):
                covering = coverage.get(ts, [])
                if covering == [i]:
                    critical.append(ts)
            
            if not critical:
                # Remove redundant block
                for ts in itertools.combinations(selected[i], t):
                    if i in coverage.get(ts, []):
                        coverage[ts].remove(i)
                selected.pop(i)
                coverage = build_coverage_index(selected)
                improved = True
                break
            
            # Try to find a replacement block that covers all critical t-sets
            # and adds extra coverage
            best_cand = None
            best_extra = -1
            
            # Sample candidates that cover all critical t-sets
            num_tries = 100
            for _ in range(num_tries):
                if time.time() >= deadline_ls:
                    break
                
                cand = sorted(rng.sample(range(v), k))
                cand_set = set(cand)
                
                # Check if candidate covers all critical t-sets
                covers_all = True
                for ts in critical:
                    if not set(ts).issubset(cand_set):
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Count extra t-sets covered (not covered by other blocks)
                extra = 0
                for ts in itertools.combinations(cand, t):
                    covering = coverage.get(ts, [])
                    other_covering = [c for c in covering if c != i]
                    if not other_covering:
                        extra += 1
                
                if extra > best_extra:
                    best_extra = extra
                    best_cand = cand
            
            if best_cand is not None and best_extra > 0:
                # Replace block i with best_cand
                old_block = selected[i]
                for ts in itertools.combinations(old_block, t):
                    if i in coverage.get(ts, []):
                        coverage[ts].remove(i)
                
                selected[i] = best_cand
                for ts in itertools.combinations(best_cand, t):
                    if ts not in coverage:
                        coverage[ts] = []
                    if i not in coverage[ts]:
                        coverage[ts].append(i)
                
                improved = True
                break
    
    return selected


def _greedy_with_pruning(v, k, t, rng, time_limit, mode='basic'):
    """
    Greedy covering with post-hoc pruning.
    mode='basic': standard greedy
    mode='weighted': greedy with pair-frequency weighting
    """
    deadline = time.time() + time_limit
    
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    # Track how many blocks contain each pair (for diversity)
    pair_count = Counter()
    
    def compute_gain(x, partial_block):
        """Number of uncovered t-sets that would be covered by adding x."""
        cnt = 0
        for S in itertools.combinations(partial_block, t - 1):
            if x not in S:
                tset = tuple(sorted(S + (x,)))
                if tset in uncovered:
                    cnt += 1
        return cnt
    
    def compute_pair_penalty(x, partial_block):
        """Penalty based on pair frequency."""
        penalty = 0
        for y in partial_block:
            p = (min(x, y), max(x, y))
            penalty += pair_count[p]
        return penalty
    
    while uncovered and time.time() < deadline:
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            
            best_x = None
            best_score = -1
            
            for x in candidates:
                gain = compute_gain(x, block)
                
                if mode == 'basic':
                    score = float(gain)
                elif mode == 'weighted':
                    penalty = compute_pair_penalty(x, block)
                    score = gain - 0.1 * penalty
                else:
                    score = float(gain)
                
                score += rng.random() * 0.005
                
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update pair counts
        for p in itertools.combinations(block, 2):
            pair_count[p] += 1
        
        # Remove covered t-sets
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
    
    return [list(b) for b in blocks]


def _local_search_prune(blocks, v, k, t, rng, time_limit):
    """
    Local search to prune redundant blocks and improve coverage.
    """
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    
    if not blocks:
        return blocks
    
    all_tsets = list(itertools.combinations(range(v), t))
    
    def build_index():
        idx = defaultdict(list)
        for i, b in enumerate(blocks):
            for ts in itertools.combinations(b, t):
                idx[ts].append(i)
        return idx
    
    idx = build_index()
    
    # Phase 1: Remove redundant blocks iteratively
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                covering = idx.get(ts, [])
                if covering == [i]:
                    redundant = False
                    break
            
            if redundant:
                for ts in itertools.combinations(blocks[i], t):
                    if i in idx.get(ts, []):
                        idx[ts].remove(i)
                blocks.pop(i)
                idx = build_index()
                changed = True
                break
    
    # Phase 2: Try to swap blocks for better coverage
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        
        for i in range(n):
            if time.time() >= deadline:
                break
            
            critical = []
            for ts in itertools.combinations(blocks[i], t):
                covering = idx.get(ts, [])
                if covering == [i]:
                    critical.append(ts)
            
            if not critical:
                for ts in itertools.combinations(blocks[i], t):
                    if i in idx.get(ts, []):
                        idx[ts].remove(i)
                blocks.pop(i)
                idx = build_index()
                improved = True
                break
            
            best_cand = None
            best_extra = -1
            
            for _ in range(50):
                if time.time() >= deadline:
                    break
                
                cand = sorted(rng.sample(range(v), k))
                cand_set = set(cand)
                
                covers_all = all(set(ts).issubset(cand_set) for ts in critical)
                if not covers_all:
                    continue
                
                extra = 0
                for ts in itertools.combinations(cand, t):
                    covering = idx.get(ts, [])
                    other = [c for c in covering if c != i]
                    if not other:
                        extra += 1
                
                if extra > best_extra:
                    best_extra = extra
                    best_cand = cand
            
            if best_cand is not None and best_extra > 0:
                old = blocks[i]
                for ts in itertools.combinations(old, t):
                    if i in idx.get(ts, []):
                        idx[ts].remove(i)
                
                blocks[i] = best_cand
                for ts in itertools.combinations(best_cand, t):
                    if ts not in idx:
                        idx[ts] = []
                    if i not in idx[ts]:
                        idx[ts].append(i)
                
                improved = True
                break
    
    return blocks


def _verify_covering(blocks, v, t):
    """Verify that every t-subset is covered by at least one block."""
    covered = set()
    for b in blocks:
        bset = set(b)
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    
    for ts in itertools.combinations(range(v), t):
        if ts not in covered:
            return False
    return True


def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0  # Safety margin before 150s kill
    
    best_blocks = None
    best_len = float('inf')
    
    # Allocate budget
    greedy_budget = total_budget * 0.40
    walksat_budget = total_budget * 0.35
    ls_budget = total_budget * 0.20
    margin = total_budget * 0.05
    
    # Phase 1: Multiple greedy runs
    configs = [
        ('basic', 42),
        ('weighted', 137),
        ('basic', 274),
        ('weighted', 411),
        ('basic', 548),
    ]
    
    per_greedy = greedy_budget / len(configs)
    
    for mode, seed in configs:
        elapsed = time.time() - start_time
        if elapsed > total_budget - walksat_budget - ls_budget - margin:
            break
        
        rng = random.Random(seed)
        blocks = _greedy_with_pruning(v, k, t, rng, per_greedy, mode=mode)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    # Phase 2: WalkSAT-style local search
    elapsed = time.time() - start_time
    remaining_walksat = walksat_budget - (elapsed - start_time) if elapsed - start_time < walksat_budget else walksat_budget
    
    if remaining_walksat > 10.0 and best_blocks is not None:
        # Run WalkSAT on the best greedy result
        deadline_ws = time.time() + min(remaining_walksat, 50.0)
        walksat_result = _walksat_covering(v, k, t, deadline_ws, seed=999)
        
        if walksat_result and _verify_covering(walksat_result, v, t):
            if len(walksat_result) < best_len:
                best_len = len(walksat_result)
                best_blocks = walksat_result
    
    # Phase 3: Local search refinement
    elapsed = time.time() - start_time
    remaining_ls = total_budget - ls_budget - margin - elapsed
    
    if remaining_ls > 5.0 and best_blocks is not None:
        rng = random.Random(7777)
        refined = _local_search_prune(best_blocks, v, k, t, rng, remaining_ls)
        
        if _verify_covering(refined, v, t):
            if len(refined) < best_len:
                best_blocks = refined
                best_len = len(refined)
    
    # Final safety check
    if time.time() - start_time > total_budget - 3.0:
        return best_blocks
    
    return best_blocks

# EVOLVE-BLOCK-END
