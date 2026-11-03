import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of triple indices it covers
    # Also build triple -> list of blocks that cover it (for local search)
    block_triples = []
    triple_to_blocks = defaultdict(list)
    
    for i, block in enumerate(all_blocks):
        block_triples.append([])
        for triple in itertools.combinations(block, t):
            ti = triple_to_idx[triple]
            block_triples[i].append(ti)
            triple_to_blocks[ti].append(i)

    # Convert to numpy arrays for faster access where possible
    # block_triples[i] is a list of triple indices
    block_triple_arrays = [np.array(bt, dtype=np.int32) for bt in block_triples]
    
    # Precompute triple frequencies for weighted greedy
    triple_freq = np.zeros(n_triples, dtype=np.float64)
    for bt in block_triples:
        for ti in bt:
            triple_freq[ti] += 1
    triple_weight = 1.0 / np.maximum(triple_freq, 1.0)

    def greedy_cover(use_weighted=False, random_tiebreak=False, rng=None):
        """Run greedy set cover. Returns list of block indices."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        n_uncovered = n_triples
        
        while n_uncovered > 0:
            if time.time() > deadline:
                break
            
            # Compute gains: for each block, how many uncovered triples it covers
            # Use the dual approach: for each uncovered triple, add to all blocks covering it
            gains = np.zeros(n_blocks, dtype=np.float64)
            
            if use_weighted:
                for ti in range(n_triples):
                    if uncovered[ti]:
                        w = triple_weight[ti]
                        for bi in triple_to_blocks[ti]:
                            gains[bi] += w
            else:
                for ti in range(n_triples):
                    if uncovered[ti]:
                        for bi in triple_to_blocks[ti]:
                            gains[bi] += 1.0
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            if random_tiebreak and rng is not None:
                candidates = np.where(gains >= max_gain - 1e-9)[0]
                best_idx = rng.choice(candidates)
            else:
                best_idx = int(np.argmax(gains))
            
            selected.append(best_idx)
            for ti in block_triples[best_idx]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    n_uncovered -= 1
        
        return selected

    def peel_redundant(block_indices):
        """Remove redundant blocks from a solution."""
        if not block_indices:
            return list(block_indices)
        
        n_sel = len(block_indices)
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            for ti in block_triples[bidx]:
                coverage_count[ti] += 1
        
        keep = []
        for bidx in block_indices:
            redundant = True
            for ti in block_triples[bidx]:
                if coverage_count[ti] <= 1:
                    redundant = False
                    break
            if not redundant:
                for ti in block_triples[bidx]:
                    coverage_count[ti] -= 1
                keep.append(bidx)
        
        return keep

    def local_search(block_indices, rng, max_iters=500):
        """Local search: try swaps to reduce block count while maintaining validity."""
        if not block_indices:
            return list(block_indices)
        
        n_sel = len(block_indices)
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            for ti in block_triples[bidx]:
                coverage_count[ti] += 1
        
        current_blocks = list(block_indices)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        no_improve = 0
        max_no_improve = 80
        
        iter_count = 0
        while time.time() < deadline - 5.0 and iter_count < max_iters:
            iter_count += 1
            
            # Try to remove redundant blocks first
            removed = False
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            for i in indices:
                bidx = current_blocks[i]
                redundant = True
                for ti in block_triples[bidx]:
                    if coverage_count[ti] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ti in block_triples[bidx]:
                        coverage_count[ti] -= 1
                    current_blocks.pop(i)
                    removed = True
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                        no_improve = 0
                    break
            
            if removed:
                continue
            
            # Try swaps: replace one block with another
            if len(current_blocks) == 0:
                break
                
            i = rng.randrange(len(current_blocks))
            old_bidx = current_blocks[i]
            old_triples = block_triples[old_bidx]
            
            # Find critical triples (covered only by this block)
            critical_triples = [ti for ti in old_triples if coverage_count[ti] == 1]
            
            if len(critical_triples) == 0:
                no_improve += 1
                if no_improve > max_no_improve and len(current_blocks) > 1:
                    # Perturbation: remove a random block and rebuild
                    j = rng.randrange(len(current_blocks))
                    removed_bidx = current_blocks[j]
                    for ti in block_triples[removed_bidx]:
                        coverage_count[ti] -= 1
                    current_blocks.pop(j)
                    
                    # Rebuild to cover uncovered triples
                    uncovered_mask = (coverage_count == 0)
                    n_unc = np.sum(uncovered_mask)
                    while n_unc > 0:
                        if time.time() > deadline - 5.0:
                            break
                        gains = np.zeros(n_blocks, dtype=np.float64)
                        for ti in range(n_triples):
                            if uncovered_mask[ti]:
                                for bi in triple_to_blocks[ti]:
                                    gains[bi] += 1.0
                        best_add = int(np.argmax(gains))
                        if gains[best_add] == 0:
                            break
                        for ti in block_triples[best_add]:
                            coverage_count[ti] += 1
                        current_blocks.append(best_add)
                        uncovered_mask = (coverage_count == 0)
                        n_unc = np.sum(uncovered_mask)
                    
                    no_improve = 0
                continue
            
            # Find candidate blocks that cover all critical triples
            # For efficiency, check candidates from the blocks that cover the first critical triple
            first_critical = critical_triples[0]
            candidates = triple_to_blocks[first_critical]
            
            best_candidate = -1
            best_gain = -1
            
            # Sample if too many candidates
            if len(candidates) > 100:
                candidates_sample = rng.sample(candidates, 100)
            else:
                candidates_sample = candidates
            
            for cand_idx in candidates_sample:
                if cand_idx == old_bidx:
                    continue
                cand_triples = block_triples[cand_idx]
                
                # Check if candidate covers all critical triples
                covers_all = True
                for ct in critical_triples:
                    if ct not in cand_triples:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Calculate gain: how many new triples does this block cover?
                new_coverage = 0
                for ct in cand_triples:
                    if coverage_count[ct] == 0:
                        new_coverage += 1
                
                if new_coverage > best_gain:
                    best_gain = new_coverage
                    best_candidate = cand_idx
            
            if best_candidate >= 0:
                # Perform the swap
                for ti in block_triples[old_bidx]:
                    coverage_count[ti] -= 1
                for ti in block_triples[best_candidate]:
                    coverage_count[ti] += 1
                
                current_blocks[i] = best_candidate
                no_improve = 0
            else:
                no_improve += 1
        
        return best_blocks

    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Strategy 1: Deterministic greedy
    sel = greedy_cover(use_weighted=False, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Strategy 2: Weighted greedy
    sel = greedy_cover(use_weighted=True, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Strategy 3+: Random restarts with weighted greedy + local search
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        # Vary the weighting: use different random perturbations
        sel = greedy_cover(use_weighted=True, random_tiebreak=True, rng=rng)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply local search
            if time.time() < deadline - 10.0:
                ls_result = local_search(peeled, rng, max_iters=400)
                ls_size = len(ls_result)
                if ls_size < size:
                    size = ls_size
                    peeled = ls_result
            
            if size < best_size:
                best_size = size
                best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Final polish: try local search on the best solution found
    if best_solution is not None and time.time() < deadline - 5.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        current_indices = []
        for b in best_solution:
            t = tuple(b)
            if t in block_to_idx:
                current_indices.append(block_to_idx[t])
        
        if current_indices:
            ls_result = local_search(current_indices, rng, max_iters=300)
            if len(ls_result) < len(best_solution):
                best_solution = [list(all_blocks[i]) for i in ls_result]
                best_size = len(ls_result)
    
    if best_solution is None:
        best_solution = []
    
    return best_solution
