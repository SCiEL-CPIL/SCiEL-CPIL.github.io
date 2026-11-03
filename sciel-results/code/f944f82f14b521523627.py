"""Group Testing / Information-Theoretic Covering C(24,7,3) with DP-style Pruning."""
import itertools
import random
import time
import math
from collections import defaultdict

def construct(v, k, t):
    """
    Construct covering design using Information-Theoretic Group Testing approach:
    1. Entropy-maximizing greedy: select blocks to maximize entropy reduction of uncovered triple distribution
    2. DP-style pruning: group uncovered triples by 'candidate block count' to prune search
    3. Local search with aggressive redundancy removal
    """
    all_triples = list(itertools.combinations(range(v), t))
    all_triples_set = set(all_triples)
    
    # Phase 1: Entropy-maximizing greedy with DP-style pruning
    blocks = _entropy_greedy(v, k, t, all_triples_set)
    
    # Phase 2: Aggressive local search
    blocks = _aggressive_local_search(v, k, t, all_triples_set, blocks)
    
    return blocks

def _entropy_greedy(v, k, t, all_triples_set):
    """
    Greedy that maximizes entropy reduction of uncovered triple distribution.
    Uses DP-style grouping: triples are grouped by how many points they share with
    recently selected blocks, enabling pruning of low-value candidates.
    """
    uncovered = set(all_triples_set)
    blocks = []
    
    # Track point frequencies for entropy computation
    point_in_uncovered = [0] * v
    for triple in uncovered:
        for p in triple:
            point_in_uncovered[p] += 1
    
    # For DP-style pruning: group triples by their "constraint level"
    # A triple's constraint level = number of points that appear in few uncovered triples
    # High constraint = hard to cover = prioritize
    
    while uncovered:
        # Compute entropy of current uncovered distribution
        # H = -sum(p_i * log2(p_i)) where p_i is probability of triple i
        # We approximate by using point frequencies as a proxy
        
        # Select seed triple: maximize entropy contribution
        # Triples with rare points contribute more to entropy uncertainty
        best_seed = None
        best_seed_score = -1
        
        # Sample-based selection for speed (full scan is O(n^2) which is too slow)
        if len(uncovered) > 100:
            candidates = random.sample(list(uncovered), min(50, len(uncovered)))
        else:
            candidates = list(uncovered)
        
        for triple in candidates:
            # Score = sum of -log2(freq) for each point (higher = more informative)
            score = 0.0
            for p in triple:
                freq = point_in_uncovered[p]
                if freq > 0:
                    score += -math.log2(freq / len(uncovered) + 1e-10)
                else:
                    score += 10.0  # Very high priority if point not in uncovered
            if score > best_seed_score:
                best_seed_score = score
                best_seed = triple
        
        if best_seed is None:
            best_seed = next(iter(uncovered))
        
        # Build block around best_seed using entropy-maximizing extension
        block = list(best_seed)
        block_set = set(block)
        
        while len(block) < k:
            best_point = None
            best_entropy_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Compute entropy gain of adding x
                # New triples formed that are currently uncovered
                new_triples = []
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        triple = tuple(sorted([block[i], block[j], x]))
                        if triple in uncovered:
                            new_triples.append(triple)
                
                if not new_triples:
                    continue
                
                # Entropy gain: reduction in uncertainty
                # Approximate by sum of -log2(freq) for newly covered triples
                gain = 0.0
                for triple in new_triples:
                    # Weight by how "surprising" this triple is
                    min_freq = min(point_in_uncovered[p] for p in triple)
                    if min_freq > 0:
                        gain += -math.log2(min_freq / max(1, len(uncovered)) + 1e-10)
                    else:
                        gain += 5.0
                
                if gain > best_entropy_gain:
                    best_entropy_gain = gain
                    best_point = x
            
            if best_point is None or best_entropy_gain <= 0:
                # Fallback: pick point that covers most uncovered triples
                best_point = None
                best_count = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    count = 0
                    for i in range(len(block)):
                        for j in range(i + 1, len(block)):
                            triple = tuple(sorted([block[i], block[j], x]))
                            if triple in uncovered:
                                count += 1
                    if count > best_count:
                        best_count = count
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)
                block_set.add(best_point)
            else:
                block.append(best_point)
                block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered and point frequencies
        covered_triples = set(itertools.combinations(block, t))
        for triple in covered_triples:
            if triple in uncovered:
                uncovered.discard(triple)
                for p in triple:
                    point_in_uncovered[p] -= 1
    
    return blocks

def _aggressive_local_search(v, k, t, all_triples_set, blocks, max_rounds=500):
    """
    Aggressive local search:
    1. Remove all redundant blocks (iterate until stable)
    2. Try to merge/replace blocks using DP-style constraint analysis
    """
    blocks = list(blocks)
    random.seed(42)
    
    def get_covered_triples(block_list):
        covered = set()
        for b in block_list:
            for triple in itertools.combinations(b, t):
                covered.add(triple)
        return covered
    
    # Pass 1: Aggressive redundancy removal
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            trial = blocks[:i] + blocks[i+1:]
            if get_covered_triples(trial) == all_triples_set:
                blocks = trial
                changed = True
                break
    
    # Pass 2: Try to replace blocks with more efficient ones
    for _ in range(max_rounds):
        if len(blocks) <= 1:
            break
        idx = random.randint(0, len(blocks) - 1)
        
        # Find triples only covered by blocks[idx]
        others_covered = set()
        for j, b in enumerate(blocks):
            if j == idx:
                continue
            for triple in itertools.combinations(b, t):
                others_covered.add(triple)
        
        must_cover = set()
        for triple in itertools.combinations(blocks[idx], t):
            if triple not in others_covered:
                must_cover.add(triple)
        
        if not must_cover:
            blocks = blocks[:idx] + blocks[idx+1:]
            continue
        
        # DP-style: check if must_cover can be covered by a single block
        # A block can cover must_cover only if all points in must_cover fit in k points
        # and the structure allows it
        all_points = set()
        for triple in must_cover:
            all_points.update(triple)
        
        if len(all_points) > k:
            continue
        
        # Try to construct a block covering must_cover
        new_block = list(all_points)
        remaining = set(must_cover)
        
        while len(new_block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in new_block:
                    continue
                gain = 0
                for i in range(len(new_block)):
                    for j in range(i + 1, len(new_block)):
                        triple = tuple(sorted([new_block[i], new_block[j], x]))
                        if triple in remaining:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain == 0:
                break
            new_block.append(best_point)
        
        new_block.sort()
        if len(new_block) == k:
            new_covered = set(itertools.combinations(new_block, t))
            if must_cover.issubset(new_covered):
                blocks[idx] = new_block
    
    # Final redundancy pass
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            trial = blocks[:i] + blocks[i+1:]
            if get_covered_triples(trial) == all_triples_set:
                blocks = trial
                changed = True
                break
    
    return blocks
