import itertools
import random
import time
import numpy as np
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    if v != 24 or k != 7 or t != 3:
        return _simple_greedy(v, k, t)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Initial greedy solution
    best_blocks = _greedy_fast(v, k, t, random_seed=42)
    best_blocks = _reduce_blocks(best_blocks, v, t)
    best_count = len(best_blocks)
    
    def get_block_triples_indices(block):
        return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
    
    # Pool of candidate blocks
    pool = [tuple(b) for b in best_blocks]
    pool_set = set(pool)
    
    # Precompute triples for initial pool
    # We will rebuild the matrix each iteration, so we just need the pool list
    
    while time.time() < deadline - 2.0:
        # Generate new candidates
        # Calculate coverage in current best solution
        coverage = [0] * n_triples
        for b in best_blocks:
            for tri_idx in get_block_triples_indices(b):
                coverage[tri_idx] += 1
        
        critical_triples = [i for i, c in enumerate(coverage) if c == 1]
        
        new_blocks = set()
        
        # 1. Random blocks
        for _ in range(300):
            b = tuple(sorted(random.sample(range(v), k)))
            if b not in pool_set:
                new_blocks.add(b)
                
        # 2. Blocks covering critical triples
        if critical_triples:
            for _ in range(300):
                tri_idx = random.choice(critical_triples)
                tri = all_triples[tri_idx]
                remaining = [p for p in range(v) if p not in tri]
                need = k - t
                if need > 0:
                    extra = random.sample(remaining, need)
                    b = tuple(sorted(list(tri) + extra))
                    if b not in pool_set:
                        new_blocks.add(b)
                        
        # Add to pool
        for b in new_blocks:
            pool.append(b)
            pool_set.add(b)
            
        # Cap pool size
        if len(pool) > 12000:
            # Keep best blocks
            best_set = set(best_blocks)
            # Keep random sample of others
            others = [b for b in pool if b not in best_set]
            random.shuffle(others)
            pool = list(best_set) + others[:10000]
            pool_set = set(pool)
            
        # Build ILP
        n_vars = len(pool)
        if n_vars == 0:
            break
            
        c = np.ones(n_vars)
        
        rows = []
        cols = []
        data = []
        
        for j, block in enumerate(pool):
            for tri_idx in get_block_triples_indices(block):
                rows.append(tri_idx)
                cols.append(j)
                data.append(-1.0)
                
        A = csr_matrix((data, (rows, cols)), shape=(n_triples, n_vars))
        b_ub = -np.ones(n_triples)
        
        integrality = np.ones(n_vars)
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, -np.inf, b_ub),
                integrality=integrality,
                bounds=Bounds(lb, ub)
            )
            
            if res.status == 0:
                sol_blocks = []
                for j in range(n_vars):
                    if res.x[j] > 0.5:
                        sol_blocks.append(pool[j])
                        
                sol_blocks = _reduce_blocks(sol_blocks, v, t)
                
                if len(sol_blocks) < best_count:
                    best_count = len(sol_blocks)
                    best_blocks = sol_blocks
                    
        except Exception:
            pass
            
    # Final reduction
    best_blocks = _reduce_blocks(best_blocks, v, t)
    
    return [list(b) for b in best_blocks]

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        target = min(uncovered)
        best_block = None
        best_gain = -1
        target_set = set(target)
        remaining_points = [p for p in range(v) if p not in target_set]
        need = k - t
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block
        blocks.append(best_block)
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks

def _greedy_fast(v, k, t, random_seed):
    rng = random.Random(random_seed)
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        unc_list = list(uncovered)
        target = unc_list[rng.randrange(len(unc_list))]
        
        best_block = None
        best_gain = -1
        
        target_set = set(target)
        remaining_points = [p for p in range(v) if p not in target_set]
        need = k - t
        
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
            elif gain == best_gain:
                if rng.random() < 0.5:
                    best_block = block
                    
        blocks.append(best_block)
        
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
            
    return blocks

def _reduce_blocks(blocks, v, t):
    if not blocks:
        return []
        
    triple_coverage = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            triple_coverage[sub] += 1
    
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            block = blocks[i]
            redundant = True
            for sub in itertools.combinations(block, t):
                if triple_coverage[sub] == 1:
                    redundant = False
                    break
            if redundant:
                for sub in itertools.combinations(block, t):
                    triple_coverage[sub] -= 1
                blocks.pop(i)
                changed = True
                break
    
    return blocks
