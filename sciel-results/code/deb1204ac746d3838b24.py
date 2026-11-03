import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a diverse greedy heuristic.
    
    Strategy:
    1. Precompute all t-subsets (triples) and map them to indices.
    2. Maintain a coverage count array to track how many blocks cover each triple.
    3. Iteratively select a new block that maximizes the number of *newly* covered triples.
       To avoid "clumping" (blocks being too similar), we apply a diversity penalty:
       gain = (new_triples_covered) - (alpha * similarity_to_existing_blocks)
       Actually, a simpler and more effective diversity measure for covering designs
       is to prioritize blocks that cover triples with the *lowest* current coverage count.
       This forces the search to spread out.
       
       Specifically, for each candidate block, we calculate a score:
       Score = sum(1 / (coverage_count[triple] + 1) for triple in block if triple is uncovered)
       This heavily weights covering rare/uncovered triples.
       
       To speed up candidate generation, we use a randomized greedy construction for the block itself,
       but we evaluate many candidates and pick the best.
    """
    deadline = time.time() + 140.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    triple_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # State
    covered = np.zeros(n_t_subsets, dtype=bool)
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    selected_blocks = []
    selected_tuples_set = set()
    
    rng = random.Random(42)
    
    # Helper: Generate a random block containing a specific anchor triple
    def generate_block_from_anchor(anchor_triple, rng):
        block = list(anchor_triple)
        block_set = set(block)
        remaining = [x for x in range(v) if x not in block_set]
        
        # Greedily fill the block to maximize coverage of *currently uncovered* triples
        # But we add a bit of randomness to explore different structures
        while len(block) < k:
            if not remaining:
                break
            
            best_gain = -1
            best_points = []
            
            # Precompute (t-1)-subsets of current block for efficiency
            block_subs = list(itertools.combinations(block, t-1))
            
            for p in remaining:
                gain = 0
                for sub in block_subs:
                    new_triple = tuple(sorted(sub + (p,)))
                    idx = triple_to_idx[new_triple]
                    if not covered[idx]:
                        gain += 1
                
                # Add small random noise to encourage diversity
                noise = rng.random() * 0.1
                total = gain + noise
                
                if total > best_gain:
                    best_gain = total
                    best_points = [p]
                elif abs(total - best_gain) < 1e-9:
                    best_points.append(p)
            
            if best_points:
                chosen = rng.choice(best_points)
                block.append(chosen)
                block_set.add(chosen)
                remaining.remove(chosen)
            else:
                # Fallback: random
                chosen = rng.choice(remaining)
                block.append(chosen)
                block_set.add(chosen)
                remaining.remove(chosen)
        
        return tuple(sorted(block))

    # Helper: Calculate the "diversity-weighted gain" of a block
    def calc_diversity_gain(block):
        gain = 0.0
        for sub in itertools.combinations(block, t):
            idx = triple_to_idx[sub]
            if not covered[idx]:
                # Heavily weight uncovered triples
                # The +1 prevents division by zero and softens the penalty for slightly covered items
                gain += 1.0 / (coverage_count[idx] + 1)
        return gain

    # Main Loop
    max_blocks = 200
    step = 0
    
    while time.time() < deadline - 10.0:
        unc = np.where(~covered)[0]
        if len(unc) == 0:
            break
        if step >= max_blocks:
            break
        
        # Identify the "most critical" uncovered triples to anchor our candidates
        # We pick a few random uncovered triples to serve as anchors
        # This ensures we don't just keep picking the same "easiest" triple
        num_anchors = 5
        anchor_indices = rng.sample(range(len(unc)), min(num_anchors, len(unc)))
        
        best_candidate = None
        best_score = -1
        
        # Generate candidates from each anchor
        for ai in anchor_indices:
            unc_idx = unc[ai]
            anchor_triple = all_t_subsets[unc_idx]
            
            # Generate a few variations for this anchor
            for _ in range(4):
                cand = generate_block_from_anchor(anchor_triple, rng)
                
                # Skip if already selected
                if cand in selected_tuples_set:
                    continue
                    
                score = calc_diversity_gain(cand)
                
                if score > best_score:
                    best_score = score
                    best_candidate = cand
        
        if best_candidate is None:
            # Fallback: just pick a random block from the first anchor
            anchor_triple = all_t_subsets[unc[0]]
            best_candidate = generate_block_from_anchor(anchor_triple, rng)
            if best_candidate in selected_tuples_set:
                # If even that is taken, we might be stuck or done
                # Try one more random
                best_candidate = generate_block_from_anchor(anchor_triple, rng)

        # Add the best candidate
        selected_blocks.append(list(best_candidate))
        selected_tuples_set.add(best_candidate)
        
        # Update state
        for sub in itertools.combinations(best_candidate, t):
            idx = triple_to_idx[sub]
            covered[idx] = True
            coverage_count[idx] += 1
            
        step += 1
        
        # Early termination if we seem to be stuck (very low gain)
        if best_score < 1e-6 and step > 10:
             # Check if actually covered
             if covered.all():
                 break

    # Local Search: Remove redundant blocks
    if time.time() < deadline - 2.0:
        # Rebuild coverage counts from scratch for safety
        coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
        for b in selected_blocks:
            for sub in itertools.combinations(b, t):
                coverage_count[triple_to_idx[sub]] += 1
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(len(selected_blocks) - 1, -1, -1):
                b = selected_blocks[i]
                redundant = True
                for sub in itertools.combinations(b, t):
                    idx = triple_to_idx[sub]
                    if coverage_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for sub in itertools.combinations(b, t):
                        coverage_count[triple_to_idx[sub]] -= 1
                    selected_blocks.pop(i)
                    improved = True
                    break # Restart scan

    return selected_blocks
