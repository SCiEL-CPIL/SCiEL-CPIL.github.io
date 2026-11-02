import itertools
import random
import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_PRECOMP_CACHE = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _PRECOMP_CACHE:
        return _PRECOMP_CACHE[key]
        
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = []
    block_triples = []
    blocks_covering_triple = [[] for _ in range(num_triples)]
    
    for block in itertools.combinations(points, k):
        block_id = len(all_blocks)
        all_blocks.append(block)
        
        block_triple_indices = []
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            block_triple_indices.append(idx)
            blocks_covering_triple[idx].append(block_id)
            
        block_triples.append(block_triple_indices)
        
    _PRECOMP_CACHE[key] = (all_blocks, block_triples, blocks_covering_triple, num_triples)
    return _PRECOMP_CACHE[key]

def _solve_residual_milp(residual_triples, all_blocks, block_triples, blocks_covering_triple, num_triples, time_limit):
    """Solve the residual covering problem using MILP."""
    if not residual_triples:
        return []
    
    residual_set = set(residual_triples)
    num_residual = len(residual_set)
    
    # Find candidate blocks that cover at least one residual triple
    candidate_blocks = set()
    for t_idx in residual_set:
        for b_idx in blocks_covering_triple[t_idx]:
            candidate_blocks.add(b_idx)
    
    candidate_blocks = sorted(candidate_blocks)
    num_candidates = len(candidate_blocks)
    
    if num_candidates == 0:
        return []
    
    # Build incidence matrix: rows = residual triples, cols = candidate blocks
    # A[i, j] = 1 if candidate block j covers residual triple i
    A = np.zeros((num_residual, num_candidates), dtype=np.int8)
    
    residual_to_local = {t_idx: i for i, t_idx in enumerate(sorted(residual_set))}
    
    for j, b_idx in enumerate(candidate_blocks):
        for t_idx in block_triples[b_idx]:
            if t_idx in residual_to_local:
                A[residual_to_local[t_idx], j] = 1
    
    # Objective: minimize sum of x_j
    c = np.ones(num_candidates)
    
    # Constraints: A @ x >= 1 for each residual triple
    constraints = LinearConstraint(A, 1, np.inf)
    
    # Bounds: x_j in {0, 1}
    bounds = Bounds(0, 1)
    
    # Integrality: all variables integer
    integrality = np.ones(num_candidates)
    
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                      options={'time_limit': time_limit, 'mip_gap': 0.0})
        
        if result.status == 0 and result.x is not None:
            selected = [candidate_blocks[j] for j in range(num_candidates) if result.x[j] > 0.5]
            return selected
    except Exception:
        pass
    
    return []

def _greedy_construct(v, k, t, all_blocks, block_triples, blocks_covering_triple, num_triples, rng, time_limit):
    """Run a single greedy construction attempt."""
    uncovered = set(range(num_triples))
    selected_blocks = []
    used_blocks = set()
    
    while uncovered and time.time() < time_limit:
        # Choose a random uncovered triple
        chosen_triple = rng.choice(list(uncovered))
        
        candidates = blocks_covering_triple[chosen_triple]
        
        best_block_idx = -1
        best_score = -1
        best_tie_breakers = []
        
        for b_idx in candidates:
            if b_idx in used_blocks:
                continue
            
            score = 0
            for t_idx in block_triples[b_idx]:
                if t_idx in uncovered:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_block_idx = b_idx
                best_tie_breakers = [b_idx]
            elif score == best_score:
                if best_score > 0:
                    best_tie_breakers.append(b_idx)
        
        if best_block_idx == -1:
            break
            
        if len(best_tie_breakers) > 1:
            best_block_idx = rng.choice(best_tie_breakers)
        
        selected_blocks.append(best_block_idx)
        used_blocks.add(best_block_idx)
        
        for t_idx in block_triples[best_block_idx]:
            if t_idx in uncovered:
                uncovered.remove(t_idx)
    
    return selected_blocks, uncovered

def _polish(selected_blocks, block_triples, num_triples):
    """Remove redundant blocks from the solution."""
    polish_improved = True
    while polish_improved:
        polish_improved = False
        
        coverage = [0] * num_triples
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                coverage[t_idx] += 1
        
        to_remove = []
        for i, b_idx in enumerate(selected_blocks):
            essential = False
            for t_idx in block_triples[b_idx]:
                if coverage[t_idx] == 1:
                    essential = True
                    break
            if not essential:
                to_remove.append(i)
        
        if to_remove:
            for i in reversed(to_remove):
                selected_blocks.pop(i)
            polish_improved = True
    
    return selected_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_blocks, block_triples, blocks_covering_triple, num_triples = _get_precomputed(v, k, t)
    
    best_solution = None
    best_size = float('inf')
    
    # Phase 1: Multiple greedy attempts with MILP completion
    greedy_attempts = 0
    while time.time() < deadline - 30.0:
        rng = random.Random()
        
        # Run greedy to get near-complete solution
        greedy_time_limit = time.time() + 5.0
        selected_blocks, uncovered = _greedy_construct(
            v, k, t, all_blocks, block_triples, blocks_covering_triple, 
            num_triples, rng, greedy_time_limit
        )
        
        # Polish the greedy solution
        selected_blocks = _polish(selected_blocks, block_triples, num_triples)
        
        # If there are still uncovered triples, try MILP completion
        if uncovered:
            milp_time_limit = min(10.0, deadline - time.time() - 5.0)
            if milp_time_limit > 1.0:
                milp_solution = _solve_residual_milp(
                    list(uncovered), all_blocks, block_triples, 
                    blocks_covering_triple, num_triples, milp_time_limit
                )
                
                if milp_solution:
                    # Combine greedy and MILP solutions
                    combined = selected_blocks + [b for b in milp_solution if b not in selected_blocks]
                    combined = _polish(combined, block_triples, num_triples)
                    
                    if len(combined) < len(selected_blocks):
                        selected_blocks = combined
        
        # Final polish
        selected_blocks = _polish(selected_blocks, block_triples, num_triples)
        
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = selected_blocks[:]
            greedy_attempts += 1
            
            if best_size <= 69:
                break
    
    # Phase 2: Try a few more pure greedy attempts with different seeds
    while time.time() < deadline - 5.0:
        rng = random.Random()
        
        greedy_time_limit = time.time() + 8.0
        selected_blocks, uncovered = _greedy_construct(
            v, k, t, all_blocks, block_triples, blocks_covering_triple,
            num_triples, rng, greedy_time_limit
        )
        
        selected_blocks = _polish(selected_blocks, block_triples, num_triples)
        
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = selected_blocks[:]
            
            if best_size <= 69:
                break
    
    if best_solution is None:
        return []
        
    return [list(all_blocks[b_idx]) for b_idx in best_solution]
