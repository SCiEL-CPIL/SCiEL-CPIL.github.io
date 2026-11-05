import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all blocks and their triples
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # block_triples[b] is a list of triple indices
    block_triples = []
    for b in all_blocks:
        trs = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(trs)
        
    triples_per_block = len(block_triples[0])
    block_triples_np = np.array(block_triples, dtype=np.int32)
    
    # Inverse index: for each triple, which blocks contain it?
    triple_to_blocks = [[] for _ in range(num_triples)]
    for b_idx in range(num_blocks):
        for t_idx in block_triples_np[b_idx]:
            triple_to_blocks[t_idx].append(b_idx)
    
    # Convert triple_to_blocks to numpy arrays for faster intersection
    triple_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in triple_to_blocks]
    
    def greedy_construct_idx(seed, sample_size=None):
        rng = random.Random(seed)
        
        if sample_size is None:
            sample_size = min(num_blocks, 300)
        
        uncovered = set(range(num_triples))
        selected_indices = []
        
        while uncovered:
            # Find block covering most uncovered triples
            best_block = -1
            best_count = -1
            
            if num_blocks > sample_size:
                candidates = rng.sample(range(num_blocks), sample_size)
            else:
                candidates = range(num_blocks)
            
            for b_idx in candidates:
                count = 0
                for t_idx in block_triples_np[b_idx]:
                    if t_idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected_indices.append(best_block)
            for t_idx in block_triples_np[best_block]:
                uncovered.discard(t_idx)
                
        return selected_indices

    def prune_indices(indices):
        if not indices:
            return [], np.zeros(num_triples, dtype=np.int32)
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        
        for b_idx in indices:
            cover_count[block_triples_np[b_idx]] += 1
            
        new_indices = []
        new_cover_count = np.zeros(num_triples, dtype=np.int32)
        
        for b_idx in indices:
            trs = block_triples_np[b_idx]
            # Check if any triple in this block is exclusively covered by it
            is_redundant = True
            for t_idx in trs:
                if cover_count[t_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                cover_count[trs] -= 1
            else:
                new_indices.append(b_idx)
                new_cover_count[trs] += 1
                
        return new_indices, new_cover_count

    def try_remove(indices, cover, b_pos):
        """Try to remove block at b_pos and see if coverage is still valid."""
        b_idx = indices[b_pos]
        trs = block_triples_np[b_idx]
        
        # Check if any triple would become uncovered
        for t_idx in trs:
            if cover[t_idx] == 1:
                return None  # Can't remove
        
        # Safe to remove
        new_indices = indices[:b_pos] + indices[b_pos+1:]
        new_cover = cover.copy()
        new_cover[trs] -= 1
        return new_indices, new_cover

    def try_swap(indices, cover, b_pos, rng, temperature=1.0):
        """Try to replace block at b_pos with a better one using simulated annealing."""
        b_idx = indices[b_pos]
        trs = block_triples_np[b_idx]
        
        # Find exclusive triples (only covered by this block)
        exclusive_triples = []
        for t_idx in trs:
            if cover[t_idx] == 1:
                exclusive_triples.append(t_idx)
                
        if not exclusive_triples:
            # Block is redundant, just remove it
            return try_remove(indices, cover, b_pos)
        
        # Find candidate blocks that cover at least some exclusive triples
        # Use a softer constraint: prefer blocks covering many exclusive triples
        candidate_scores = {}
        for t_idx in exclusive_triples:
            for b in triple_to_blocks_np[t_idx].tolist():
                if b != b_idx:
                    candidate_scores[b] = candidate_scores.get(b, 0) + 1
        
        if not candidate_scores:
            return None
        
        # Sort candidates by number of exclusive triples they cover (descending)
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        
        # Take top candidates
        max_candidates = 30
        if len(sorted_candidates) > max_candidates:
            sorted_candidates = sorted_candidates[:max_candidates]
        
        best_candidate = None
        best_score = float('-inf')
        
        for c_idx, exc_count in sorted_candidates:
            c_trs = block_triples_np[c_idx]
            
            # Score: 
            # 1. Number of exclusive triples covered (primary)
            # 2. Overlap with existing coverage (secondary - creates redundancy)
            # 3. Random perturbation for exploration
            overlap_score = np.sum(cover[c_trs])
            score = exc_count * 10 + overlap_score + rng.random() * temperature
            
            if score > best_score:
                best_score = score
                best_candidate = c_idx
        
        if best_candidate is None:
            return None
            
        # Perform the swap
        new_indices = indices.copy()
        new_indices[b_pos] = best_candidate
        new_cover = cover.copy()
        new_cover[trs] -= 1
        new_cover[block_triples_np[best_candidate]] += 1
        
        # Validate: check that no triple is now uncovered
        for t_idx in block_triples_np[best_candidate]:
            if new_cover[t_idx] < 0:
                return None  # Shouldn't happen but safety check
        
        return new_indices, new_cover

    def try_add(indices, cover, rng, temperature=1.0):
        """Try to add a new block that helps with future pruning."""
        # Find triples with lowest coverage (most vulnerable)
        min_cover = np.min(cover)
        if min_cover > 1:
            # All triples have coverage >= 2, try to find a block that adds redundancy
            # to enable future removals
            pass
        
        # Find a random triple with low coverage
        low_cover_triples = np.where(cover <= 2)[0]
        if len(low_cover_triples) == 0:
            return None
        
        t_idx = rng.choice(low_cover_triples)
        blocks_with_t = triple_to_blocks_np[t_idx].tolist()
        
        if not blocks_with_t:
            return None
        
        # Pick a random block containing this triple
        candidate = rng.choice(blocks_with_t)
        
        # Check if this block is already in indices
        if candidate in indices:
            return None
        
        # Check if adding this block creates useful redundancy
        c_trs = block_triples_np[candidate]
        # Count how many triples this block would increase coverage for
        redundancy_gain = np.sum(cover[c_trs] >= 1)
        
        if redundancy_gain < 3:
            return None
        
        new_indices = indices + [candidate]
        new_cover = cover.copy()
        new_cover[c_trs] += 1
        
        return new_indices, new_cover

    # Initial solution: try multiple greedy runs with different seeds
    best_indices = None
    best_cover = None
    best_size = float('inf')
    
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        if time.time() > deadline - 20:
            break
        indices = greedy_construct_idx(seed, sample_size=400)
        indices, cover = prune_indices(indices)
        if len(indices) < best_size:
            best_size = len(indices)
            best_indices = indices
            best_cover = cover
    
    if best_indices is None:
        best_indices = greedy_construct_idx(42, sample_size=300)
        best_indices, best_cover = prune_indices(best_indices)
        best_size = len(best_indices)
    
    current_indices = best_indices.copy()
    current_cover = best_cover.copy()
    
    stuck_count = 0
    iteration = 0
    temperature = 5.0
    cooling_rate = 0.995
    
    rng = random.Random(12345)
    
    while time.time() < deadline - 5:
        iteration += 1
        improved = False
        
        # 1. Prune
        prev_size = len(current_indices)
        current_indices, current_cover = prune_indices(current_indices)
        if len(current_indices) < prev_size:
            improved = True
            if len(current_indices) < best_size:
                best_size = len(current_indices)
                best_indices = current_indices.copy()
                best_cover = current_cover.copy()
            stuck_count = 0
            continue
        
        n_blocks = len(current_indices)
        if n_blocks == 0:
            break
        
        # 2. Try multiple random moves with simulated annealing
        attempts = 0
        max_attempts = 30
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            idx_pos = rng.randint(0, n_blocks - 1)
            
            # First try simple removal
            result = try_remove(current_indices, current_cover, idx_pos)
            if result is not None:
                current_indices, current_cover = result
                if len(current_indices) < best_size:
                    best_size = len(current_indices)
                    best_indices = current_indices.copy()
                    best_cover = current_cover.copy()
                improved = True
                stuck_count = 0
                break
            
            # Then try swap
            result = try_swap(current_indices, current_cover, idx_pos, rng, temperature)
            if result is not None:
                new_indices, new_cover = result
                # Prune after swap
                new_indices, new_cover = prune_indices(new_indices)
                
                delta = len(new_indices) - len(current_indices)
                
                if delta < 0:
                    # Better solution
                    current_indices = new_indices
                    current_cover = new_cover
                    if len(current_indices) < best_size:
                        best_size = len(current_indices)
                        best_indices = current_indices.copy()
                        best_cover = current_cover.copy()
                    improved = True
                    stuck_count = 0
                    break
                elif delta == 0:
                    # Same size - accept with probability based on temperature
                    if rng.random() < 0.5:
                        current_indices = new_indices
                        current_cover = new_cover
                    stuck_count += 1
                else:
                    # Worse solution - accept with probability exp(-delta/temperature)
                    prob = np.exp(-delta / max(temperature, 0.1))
                    if rng.random() < prob:
                        current_indices = new_indices
                        current_cover = new_cover
                    stuck_count += 1
            
            # Occasionally try adding a block
            if rng.random() < 0.1:
                add_result = try_add(current_indices, current_cover, rng, temperature)
                if add_result is not None:
                    new_indices, new_cover = add_result
                    new_indices, new_cover = prune_indices(new_indices)
                    if len(new_indices) <= len(current_indices):
                        current_indices = new_indices
                        current_cover = new_cover
            
            if stuck_count > 30:
                # Restart from best
                current_indices = best_indices.copy()
                current_cover = best_cover.copy()
                stuck_count = 0
                break
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1
        
        if not improved:
            stuck_count += 1
            if stuck_count > 80:
                # Full restart with new seed
                seed = rng.randint(0, 100000)
                new_indices = greedy_construct_idx(seed, sample_size=400)
                new_indices, new_cover = prune_indices(new_indices)
                if len(new_indices) < best_size:
                    best_size = len(new_indices)
                    best_indices = new_indices.copy()
                    best_cover = new_cover.copy()
                current_indices = best_indices.copy()
                current_cover = best_cover.copy()
                stuck_count = 0
                temperature = 3.0

    # Final prune
    best_indices, best_cover = prune_indices(best_indices)
    
    return [list(all_blocks[b]) for b in best_indices]
