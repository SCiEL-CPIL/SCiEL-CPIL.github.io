import numpy as np
import itertools
import time
import random
from math import comb
import scipy.sparse as sp

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Build sparse incidence matrix A (blocks x triples)
    row_indices = []
    col_indices = []
    for i, block in enumerate(all_blocks):
        for triple in itertools.combinations(block, t):
            j = triple_to_idx[triple]
            row_indices.append(i)
            col_indices.append(j)

    A = sp.csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(n_blocks, n_triples))

    # Precompute triple lists for each block for fast local search
    block_triples = [A.indices[A.indptr[i]:A.indptr[i+1]] for i in range(n_blocks)]

    # Precompute triple_to_blocks
    AT = A.T.tocsr()
    triple_to_blocks = [AT.indices[AT.indptr[i]:AT.indptr[i+1]] for i in range(n_triples)]

    # Initial coverage counts
    triple_counts = np.array(A.sum(axis=0)).flatten().astype(np.int32)

    def greedy_cover(random_tiebreak=False, weight_power=1.0):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        weights = 1.0 / (triple_counts + 1).astype(np.float64)
        if weight_power != 1.0:
            weights = weights ** weight_power
            
        active_blocks = np.ones(n_blocks, dtype=bool)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            w_active = weights * uncovered
            gains = A @ w_active
            gains[~active_blocks] = 0
            
            if random_tiebreak:
                max_gain = np.max(gains)
                if max_gain <= 1e-12:
                    break
                candidates = np.where(gains >= max_gain - 1e-12)[0]
                candidates = candidates[active_blocks[candidates]]
                if len(candidates) == 0:
                    break
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] <= 1e-12:
                    break
            
            selected.append(best_idx)
            active_blocks[best_idx] = False
            uncovered[block_triples[best_idx]] = False
            
        return selected

    def remove_redundant(block_indices):
        if len(block_indices) == 0:
            return block_indices
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline:
                    return result
                bi = result[i]
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    continue
                coverage[triples_in_block] -= 1
                result.pop(i)
                changed = True
                break
                
        return result

    def local_search_v2(block_indices, max_iterations=50):
        if len(block_indices) == 0:
            return block_indices
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                block_triples_bi = block_triples[bi]
                
                # 1. Check for redundancy
                if np.all(coverage[block_triples_bi] > 1):
                    coverage[block_triples_bi] -= 1
                    current.pop(i)
                    improved = True
                    break
                    
                # 2. Try to swap
                critical_mask = coverage[block_triples_bi] == 1
                if not np.any(critical_mask):
                    continue
                
                critical_triples = block_triples_bi[critical_mask]
                
                # Find candidate blocks that cover ALL critical_triples
                min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
                candidates = triple_to_blocks[min_ct]
                
                other_critical = critical_triples[critical_triples != min_ct]
                
                valid_candidates = []
                for cand in candidates:
                    if cand == bi:
                        continue
                    if len(other_critical) == 0:
                        valid_candidates.append(cand)
                    else:
                        all_covered = True
                        for ct in other_critical:
                            if ct not in block_triples[cand]:
                                all_covered = False
                                break
                        if all_covered:
                            valid_candidates.append(cand)
                
                if valid_candidates:
                    best_cand = valid_candidates[0]
                    coverage[block_triples_bi] -= 1
                    coverage[block_triples[best_cand]] += 1
                    current[i] = best_cand
                    improved = True
                    break
            
            if not improved:
                break
                
        return current

    def local_search_remove_two(block_indices, max_iterations=20):
        if len(block_indices) < 2:
            return block_indices
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(current)):
                    bi1 = current[i]
                    bi2 = current[j]
                    both_triples = np.union1d(block_triples[bi1], block_triples[bi2])
                    if np.all(coverage[both_triples] > 2):
                        coverage[both_triples] -= 2
                        current.pop(j)
                        current.pop(i)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        return current

    # Helper for simulated annealing: build block from seed points
    def build_block_from_seed(seed_points, rng):
        seed_set = set(seed_points)
        if len(seed_set) > k:
            return None
        
        current_block = list(seed_set)
        current_set = set(current_block)
        
        # Greedily add points to maximize coverage of low-count triples
        # We don't have access to global coverage here easily, so we just fill randomly or greedily based on static weights
        # For simplicity in SA, let's just fill randomly from remaining points
        remaining = [x for x in range(v) if x not in current_set]
        rng.shuffle(remaining)
        
        for x in remaining:
            if len(current_block) >= k:
                break
            current_block.append(x)
            current_set.add(x)
        
        if len(current_block) < k:
            return None
            
        return tuple(sorted(current_block))

    def simulated_annealing(initial_blocks, T=10.0, T_min=0.01, cooling=0.9995):
        if len(initial_blocks) == 0:
            return []
        
        blocks = [tuple(b) for b in initial_blocks]
        block_set = set(blocks)
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for b in blocks:
            bi = all_blocks.index(b) # This is slow, better to use a map
            # Let's use a precomputed map from block tuple to index
            coverage[block_triples[bi]] += 1
            
        # Actually, all_blocks.index is O(N). We need a dict.
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        
        # Recalculate coverage properly
        coverage = np.zeros(n_triples, dtype=np.int32)
        for b in blocks:
            bi = block_to_idx[b]
            coverage[block_triples[bi]] += 1
            
        current_cost = len(blocks)
        best_blocks = [list(b) for b in blocks]
        best_cost = current_cost
        
        rng = random.Random(42)
        
        while time.time() < deadline and T > T_min:
            # 1. Try to remove a redundant block
            if len(blocks) > 0:
                i = rng.randrange(len(blocks))
                b = blocks[i]
                bi = block_to_idx[b]
                b_triples = block_triples[bi]
                
                if np.all(coverage[b_triples] > 1):
                    coverage[b_triples] -= 1
                    blocks.pop(i)
                    current_cost -= 1
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = [list(b) for b in blocks]
                    T *= cooling
                    continue
            
            # 2. Try to swap a block
            if len(blocks) > 0:
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                old_bi = block_to_idx[old_block]
                old_triples = block_triples[old_bi]
                
                # Identify critical triples
                critical_mask = coverage[old_triples] == 1
                if np.any(critical_mask):
                    critical_triples = old_triples[critical_mask]
                    # Points needed
                    points_needed = set()
                    for ct in critical_triples:
                        points_needed.update(triples[ct])
                    
                    if len(points_needed) <= k:
                        # Build new block
                        new_block = build_block_from_seed(list(points_needed), rng)
                        if new_block and new_block != old_block:
                            new_bi = block_to_idx[new_block]
                            new_triples = block_triples[new_bi]
                            
                            # Check validity: no triple loses its last cover
                            valid = True
                            for ct in old_triples:
                                if ct not in new_triples and coverage[ct] == 1:
                                    valid = False
                                    break
                            
                            if valid:
                                # Apply swap
                                coverage[old_triples] -= 1
                                coverage[new_triples] += 1
                                blocks[i] = new_block
                                # current_cost unchanged
                                T *= cooling
                                continue
                
                # If no critical triples, we can try to replace with a "better" block?
                # Or just skip.
            
            # 3. Try to add a block? No, we want to minimize.
            
            # 4. Try to merge? (Remove 2, add 1) - complex, skip for now.
            
            T *= cooling
            
        return best_blocks

    # Run multiple restarts
    best_solution = None
    best_size = float('inf')
    
    np.random.seed(42)
    random.seed(42)
    
    # Use a subset of time for greedy restarts, rest for SA on best greedy
    greedy_time_limit = deadline - 40.0
    
    num_restarts = 0
    while time.time() < greedy_time_limit:
        num_restarts += 1
        wp = random.choice([1.0, 1.5, 2.0])
        selected = greedy_cover(random_tiebreak=True, weight_power=wp)
        selected = remove_redundant(selected)
        selected = local_search_v2(selected, max_iterations=30)
        selected = local_search_remove_two(selected, max_iterations=10)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(all_blocks[i]) for i in selected]
            
    # If we have a good solution, try to improve it with SA
    if best_solution is not None and time.time() < deadline - 5.0:
        sa_blocks = simulated_annealing(best_solution)
        if len(sa_blocks) < best_size:
            best_size = len(sa_blocks)
            best_solution = sa_blocks

    return best_solution
