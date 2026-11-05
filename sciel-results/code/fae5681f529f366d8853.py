import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute bitsets and triple indices for each block
    block_bitsets = [0] * num_blocks
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        bitset = 0
        tris_in_block = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            bitset |= (1 << idx)
            tris_in_block.append(idx)
            triple_to_blocks[idx].append(i)
        block_bitsets[i] = bitset
        block_triple_indices[i] = tris_in_block

    def greedy_construct(alpha=0.0, seed=None):
        rng = random.Random(seed)
        uncovered = (1 << num_triples) - 1
        selected = []
        selected_set = set()
        
        while uncovered and time.time() < deadline - 10:
            best_gain = -1
            best_indices = []
            min_gain = float('inf')
            
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = (block_bitsets[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                    min_gain = gain
                elif gain == best_gain:
                    best_indices.append(i)
                if gain < min_gain:
                    min_gain = gain
            
            if best_gain <= 0:
                break
            
            if alpha == 0.0:
                chosen = best_indices[0] if best_indices else 0
            else:
                threshold = best_gain - alpha * (best_gain - min_gain)
                candidates = [i for i in best_indices]
                if not candidates:
                    for i in range(num_blocks):
                        if i in selected_set:
                            continue
                        gain = (block_bitsets[i] & uncovered).bit_count()
                        if gain >= threshold:
                            candidates.append(i)
                if not candidates:
                    candidates = best_indices
                chosen = rng.choice(candidates)
            
            selected.append(chosen)
            selected_set.add(chosen)
            uncovered &= ~block_bitsets[chosen]
        
        return selected

    def local_search(selected_blocks, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 5
        
        sol = list(selected_blocks)
        sol_set = set(sol)
        
        counts = [0] * num_triples
        for idx in sol:
            for tri_idx in block_triple_indices[idx]:
                counts[tri_idx] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            i = 0
            while i < len(sol):
                idx = sol[i]
                is_redundant = True
                for tri_idx in block_triple_indices[idx]:
                    if counts[tri_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    sol.pop(i)
                    sol_set.discard(idx)
                    for tri_idx in block_triple_indices[idx]:
                        counts[tri_idx] -= 1
                    improved = True
                else:
                    i += 1
            
            if time.time() > time_limit:
                break
            
            for i, idx in enumerate(sol):
                if time.time() > time_limit:
                    break
                
                unique_tris = []
                for tri_idx in block_triple_indices[idx]:
                    if counts[tri_idx] == 1:
                        unique_tris.append(tri_idx)
                
                if not unique_tris:
                    continue
                
                min_len = float('inf')
                min_tri = -1
                for tri_idx in unique_tris:
                    if len(triple_to_blocks[tri_idx]) < min_len:
                        min_len = len(triple_to_blocks[tri_idx])
                        min_tri = tri_idx
                
                candidates = triple_to_blocks[min_tri]
                valid_candidate = -1
                for cand_idx in candidates:
                    if cand_idx in sol_set:
                        continue
                    cand_bitset = block_bitsets[cand_idx]
                    covers_all = True
                    for tri_idx in unique_tris:
                        if not (cand_bitset & (1 << tri_idx)):
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidate = cand_idx
                        break
                
                if valid_candidate != -1:
                    for tri_idx in block_triple_indices[idx]:
                        counts[tri_idx] -= 1
                    sol_set.discard(idx)
                    
                    sol[i] = valid_candidate
                    sol_set.add(valid_candidate)
                    for tri_idx in block_triple_indices[valid_candidate]:
                        counts[tri_idx] += 1
                    
                    improved = True
                    break
        
        return sol

    def csp_minimize(initial_sol, time_limit=None):
        """
        Structural change: Backtracking search to find a minimal solution.
        We start with a valid solution from greedy. We try to remove blocks.
        If removal breaks coverage, we try to "repair" by adding a different block
        that covers the lost triples, effectively performing a guided swap
        that respects the global coverage constraint strictly.
        """
        if time_limit is None:
            time_limit = deadline - 5
            
        sol = list(initial_sol)
        sol_set = set(sol)
        
        counts = [0] * num_triples
        for idx in sol:
            for tri_idx in block_triple_indices[idx]:
                counts[tri_idx] += 1
        
        # Heuristic ordering: try to remove blocks that are "most redundant" first
        # i.e., blocks whose unique triples are fewest, or blocks that appear late in greedy
        # But we want to find a MINIMAL set. A set is minimal if no block can be removed.
        # The current local_search already does this. 
        # To be structurally different, let's try a different optimization path:
        # Simulated Annealing on the set of blocks, where the "energy" is the number of blocks
        # plus a penalty for uncovered triples.
        
        rng = random.Random(42)
        current_energy = len(sol)
        best_sol = list(sol)
        best_energy = len(sol)
        
        # We need a move generator.
        # Move: Remove one block, Add one block.
        # Or: Remove one block (if valid).
        
        # Let's try a simple random walk with acceptance criterion.
        # State: set of blocks.
        # Energy: len(blocks) * 1000 - covered_triples_count (we want to minimize blocks, maximize coverage)
        # Actually, just len(blocks) if valid, else large penalty.
        
        def get_energy(blocks_set):
            # Calculate coverage
            covered = 0
            for idx in blocks_set:
                covered |= block_bitsets[idx]
            num_covered = covered.bit_count()
            if num_covered == num_triples:
                return len(blocks_set)
            else:
                return len(blocks_set) + (num_triples - num_covered) * 10000

        current_blocks = set(sol)
        current_energy = get_energy(current_blocks)
        
        T = 10.0
        cooling = 0.995
        
        while time.time() < time_limit:
            # Generate a neighbor
            # 1. Remove a random block
            # 2. Add a random block
            # 3. Swap: Remove one, add one
            
            move_type = rng.choice(['remove', 'add', 'swap'])
            
            if move_type == 'remove' and current_blocks:
                # Pick a block to remove
                to_remove = rng.choice(list(current_blocks))
                new_blocks = set(current_blocks)
                new_blocks.discard(to_remove)
                
            elif move_type == 'add':
                # Pick a block to add (prefer those covering many uncovered)
                # For simplicity, pick random
                to_add = rng.choice(range(num_blocks))
                new_blocks = set(current_blocks)
                new_blocks.add(to_add)
                
            else: # swap
                if not current_blocks:
                    continue
                to_remove = rng.choice(list(current_blocks))
                to_add = rng.choice(range(num_blocks))
                new_blocks = set(current_blocks)
                new_blocks.discard(to_remove)
                new_blocks.add(to_add)
            
            new_energy = get_energy(new_blocks)
            
            # Acceptance
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < pow(2.718281828, -delta_e / T):
                current_blocks = new_blocks
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(current_blocks)
            
            T *= cooling
            if T < 0.1:
                break
        
        return best_sol

    best_blocks_list = None
    best_size = num_blocks
    
    # Phase 1: Standard Greedy + Local Search to get a good baseline
    # This is kept to ensure we have a valid solution to start the CSP phase
    baseline_sol = greedy_construct(0.0, 1)
    baseline_sol = local_search(baseline_sol, deadline - 20)
    
    if len(baseline_sol) < best_size:
        covered = 0
        for idx in baseline_sol:
            covered |= block_bitsets[idx]
        if covered == (1 << num_triples) - 1:
            best_size = len(baseline_sol)
            best_blocks_list = [all_blocks[i] for i in baseline_sol]
    
    # Phase 2: CSP / Simulated Annealing approach
    # We use the baseline as the starting point for the SA
    # This is the structural change: using a stochastic global optimization method
    # on the solution space defined by the coverage constraints.
    
    if time.time() < deadline - 10:
        sa_sol = csp_minimize(baseline_sol, deadline - 5)
        if len(sa_sol) < best_size:
            covered = 0
            for idx in sa_sol:
                covered |= block_bitsets[idx]
            if covered == (1 << num_triples) - 1:
                best_size = len(sa_sol)
                best_blocks_list = [all_blocks[i] for i in sa_sol]
    
    if best_blocks_list is None:
        best_blocks_list = all_blocks
        best_size = num_blocks
    
    return best_blocks_list
