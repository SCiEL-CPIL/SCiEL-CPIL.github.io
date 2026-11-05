import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # 1. Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage structure
    # For each block, which triples does it cover?
    block_triples = [[] for _ in range(num_blocks)]
    # For each triple, which blocks cover it?
    triple_blocks = [[] for _ in range(num_triples)]
    
    # Bitsets for fast coverage checks
    block_bitsets = [0] * num_blocks
    
    for i, b in enumerate(all_blocks):
        tris = []
        bs = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_blocks[idx].append(i)
            bs |= (1 << idx)
        block_triples[i] = tris
        block_bitsets[i] = bs
    
    # Helper: Greedy Set Cover with weighting
    def greedy_cover(uncovered_bitset, weights=None):
        if weights is None:
            weights = [1.0] * num_blocks
        selected = []
        current_uncovered = uncovered_bitset
        while current_uncovered:
            best_gain = -1
            best_idx = -1
            # Scan all blocks
            for i in range(num_blocks):
                gain = (block_bitsets[i] & current_uncovered).bit_count()
                if gain > 0:
                    # Weighted gain: prefer blocks that cover more uncovered triples
                    # and have lower weight (less "cost")
                    score = gain / (weights[i] + 1e-9)
                    if score > best_gain:
                        best_gain = score
                        best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            current_uncovered &= ~block_bitsets[best_idx]
        return selected
    
    # Initial Solution: Strong Greedy
    initial_sol = greedy_cover((1 << num_triples) - 1)
    
    if not initial_sol:
        return [list(all_blocks[i]) for i in range(100)]
    
    current_sol = list(initial_sol)
    best_sol = list(current_sol)
    best_size = len(current_sol)
    
    # Helper: Compute coverage counts for a solution
    def get_coverage_counts(sol):
        counts = [0] * num_triples
        for b in sol:
            for tri_idx in block_triples[b]:
                counts[tri_idx] += 1
        return counts
    
    # Helper: Check if solution is valid
    def is_valid(sol):
        counts = get_coverage_counts(sol)
        return all(c > 0 for c in counts)
    
    # Helper: Try to remove redundant blocks
    def remove_redundant(sol):
        if not sol:
            return sol
        counts = get_coverage_counts(sol)
        new_sol = []
        for b in sol:
            # Check if b is essential
            essential = False
            for tri_idx in block_triples[b]:
                if counts[tri_idx] == 1:
                    essential = True
                    break
            if not essential:
                # Redundant: remove it
                for tri_idx in block_triples[b]:
                    counts[tri_idx] -= 1
            else:
                new_sol.append(b)
        return new_sol
    
    # Helper: Find best swap: replace block b_old with block b_new
    # Gain = (uncovered triples covered by b_new) - (uncovered triples lost from b_old)
    # We want to maximize gain while maintaining validity
    def try_swap(sol, b_old_idx, counts):
        # Find which triples are exclusively covered by b_old
        exclusive_to_old = []
        for tri_idx in block_triples[b_old_idx]:
            if counts[tri_idx] == 1:
                exclusive_to_old.append(tri_idx)
        
        if not exclusive_to_old:
            # b_old is redundant, just remove it
            return [b for b in sol if b != b_old_idx], counts, True
        
        # Find candidate blocks that cover all exclusive_to_old
        # and have high coverage of other triples
        exclusive_set = set(exclusive_to_old)
        best_new = -1
        best_score = -1
        
        # Only consider blocks that cover all exclusive triples
        for b_new in triple_blocks[exclusive_to_old[0]]:
            if b_new == b_old_idx:
                continue
            # Check if b_new covers all exclusive triples
            covers_all = True
            for tri_idx in exclusive_to_old:
                if tri_idx not in block_triples[b_new]:
                    covers_all = False
                    break
            if not covers_all:
                continue
            
            # Score: how many additional triples does b_new cover that are currently under-covered?
            score = 0
            for tri_idx in block_triples[b_new]:
                if tri_idx in exclusive_set:
                    continue
                if counts[tri_idx] <= 2:
                    score += (3 - counts[tri_idx])
            
            if score > best_score:
                best_score = score
                best_new = b_new
        
        if best_new != -1:
            new_sol = [b for b in sol if b != b_old_idx] + [best_new]
            # Update counts
            for tri_idx in block_triples[b_old_idx]:
                counts[tri_idx] -= 1
            for tri_idx in block_triples[best_new]:
                counts[tri_idx] += 1
            return new_sol, counts, True
        
        return sol, counts, False
    
    # Helper: 2-opt exchange: replace two blocks with two others
    def try_2opt(sol, counts):
        n = len(sol)
        if n < 2:
            return sol, counts, False
        
        # Try all pairs of blocks to remove
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = sol[i], sol[j]
                
                # Find triples exclusively covered by b1 or b2
                exclusive_to_pair = []
                for tri_idx in block_triples[b1]:
                    if counts[tri_idx] == 1:
                        exclusive_to_pair.append(tri_idx)
                for tri_idx in block_triples[b2]:
                    if counts[tri_idx] == 1:
                        exclusive_to_pair.append(tri_idx)
                
                if not exclusive_to_pair:
                    # Both are redundant
                    new_sol = [b for b in sol if b != b1 and b != b2]
                    for tri_idx in block_triples[b1]:
                        counts[tri_idx] -= 1
                    for tri_idx in block_triples[b2]:
                        counts[tri_idx] -= 1
                    return new_sol, counts, True
                
                exclusive_set = set(exclusive_to_pair)
                
                # Find two blocks that cover all exclusive triples
                # This is expensive, so limit search
                best_pair = None
                best_score = -1
                
                # Sample candidate blocks
                candidates = set()
                for tri_idx in exclusive_to_pair:
                    for b in triple_blocks[tri_idx]:
                        if b != b1 and b != b2:
                            candidates.add(b)
                
                # Limit candidates to top 50 by some heuristic
                if len(candidates) > 50:
                    # Prefer blocks that cover many exclusive triples
                    cand_list = list(candidates)
                    cand_list.sort(key=lambda b: sum(1 for tri in block_triples[b] if tri in exclusive_set), reverse=True)
                    candidates = set(cand_list[:50])
                
                cand_list = list(candidates)
                for idx1 in range(len(cand_list)):
                    for idx2 in range(idx1 + 1, len(cand_list)):
                        b_new1, b_new2 = cand_list[idx1], cand_list[idx2]
                        
                        # Check if pair covers all exclusive triples
                        covers_all = True
                        for tri_idx in exclusive_to_pair:
                            if tri_idx not in block_triples[b_new1] and tri_idx not in block_triples[b_new2]:
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        
                        # Score
                        score = 0
                        for tri_idx in block_triples[b_new1]:
                            if tri_idx in exclusive_set:
                                continue
                            if counts[tri_idx] <= 2:
                                score += (3 - counts[tri_idx])
                        for tri_idx in block_triples[b_new2]:
                            if tri_idx in exclusive_set:
                                continue
                            if counts[tri_idx] <= 2:
                                score += (3 - counts[tri_idx])
                        
                        if score > best_score:
                            best_score = score
                            best_pair = (b_new1, b_new2)
                
                if best_pair:
                    b_new1, b_new2 = best_pair
                    new_sol = [b for b in sol if b != b1 and b != b2] + [b_new1, b_new2]
                    for tri_idx in block_triples[b1]:
                        counts[tri_idx] -= 1
                    for tri_idx in block_triples[b2]:
                        counts[tri_idx] -= 1
                    for tri_idx in block_triples[b_new1]:
                        counts[tri_idx] += 1
                    for tri_idx in block_triples[b_new2]:
                        counts[tri_idx] += 1
                    return new_sol, counts, True
        
        return sol, counts, False
    
    # Main Search Loop
    stagnation = 0
    max_stagnation = 100
    
    while time.time() < deadline - 5:
        # Local improvement: remove redundant
        current_sol = remove_redundant(current_sol)
        
        if len(current_sol) < best_size:
            best_sol = list(current_sol)
            best_size = len(current_sol)
            stagnation = 0
        else:
            stagnation += 1
        
        if time.time() > deadline - 10:
            break
        
        # Compute coverage counts
        counts = get_coverage_counts(current_sol)
        
        # Try 1-opt swaps
        improved = False
        for b in current_sol:
            new_sol, new_counts, success = try_swap(current_sol, b, counts)
            if success and len(new_sol) < len(current_sol):
                current_sol = new_sol
                counts = new_counts
                improved = True
                break
        
        if improved:
            continue
        
        # Try 2-opt exchanges
        new_sol, new_counts, success = try_2opt(current_sol, counts)
        if success and len(new_sol) < len(current_sol):
            current_sol = new_sol
            counts = new_counts
            continue
        
        # Perturbation: random restart with diversity
        if stagnation > max_stagnation:
            # Generate a new initial solution with randomization
            # Use random ordering of blocks in greedy
            random.shuffle(all_blocks)
            # Re-index? No, just use random greedy
            uncovered = (1 << num_triples) - 1
            new_initial = []
            while uncovered:
                best_gain = -1
                best_idx = -1
                for i in range(num_blocks):
                    gain = (block_bitsets[i] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                new_initial.append(best_idx)
                uncovered &= ~block_bitsets[best_idx]
            
            if new_initial:
                current_sol = new_initial
                stagnation = 0
        
        # Small perturbation: remove 1-2 random blocks and repair
        if len(current_sol) > 2 and random.random() < 0.3:
            k_remove = random.choice([1, 2])
            removed = random.sample(current_sol, k_remove)
            remaining = [b for b in current_sol if b not in removed]
            
            # Compute uncovered
            covered = 0
            for b in remaining:
                covered |= block_bitsets[b]
            needed = ((1 << num_triples) - 1) & ~covered
            
            # Repair with greedy
            repair = greedy_cover(needed)
            new_sol = remaining + repair
            new_sol = remove_redundant(new_sol)
            
            if len(new_sol) <= len(current_sol):
                current_sol = new_sol
    
    # Final polish
    if time.time() < deadline - 2:
        best_sol = remove_redundant(best_sol)
    
    # Ensure validity
    if not is_valid(best_sol):
        if is_valid(initial_sol):
            best_sol = initial_sol
        else:
            best_sol = initial_sol
    
    result_blocks = [list(all_blocks[i]) for i in best_sol]
    return result_blocks
