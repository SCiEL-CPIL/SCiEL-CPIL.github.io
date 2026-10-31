import time
import random
import itertools
import numpy as np

# Global cache for precomputed data
_PREPARE_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _PREPARE_CACHE:
        return _PREPARE_CACHE[key]
    
    # Generate all t-subsets
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    n_t = len(t_subs)
    index = {sub: i for i, sub in enumerate(t_subs)}
    
    # Generate all k-blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Precompute NumPy coverage matrix (n_blocks x n_t)
    # Using uint8 to save memory and speed up operations
    cover_matrix = np.zeros((n_blocks, n_t), dtype=np.uint8)
    
    # Also build sparse index: subset_index -> list of block_indices
    # This is crucial for fast swap search (finding candidates that cover specific subsets)
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, b in enumerate(blocks):
        for sub in itertools.combinations(b, t):
            s_idx = index[sub]
            cover_matrix[b_idx, s_idx] = 1
            sub_to_blocks[s_idx].append(b_idx)
        
    # Precompute a "degree" or "connectivity" score for each block?
    # The directive suggests spectral clustering / eigenvector centrality.
    # A simple surrogate for centrality in this context is the number of "rare" subsets a block covers.
    # Subsets with low frequency (few blocks cover them) are "peripheral" or "hard".
    # Blocks covering many "hard" subsets are more "central" or critical.
    
    # Count frequency of each subset
    subset_freq = np.sum(cover_matrix, axis=0)
    
    # We might use this in local search to prioritize swaps.
    
    _PREPARE_CACHE[key] = (blocks, cover_matrix, n_t, index, sub_to_blocks, n_blocks, subset_freq)
    return _PREPARE_CACHE[key]

def _greedy_select(cover_matrix, n_blocks, n_t, deadline):
    """
    Fast greedy construction using NumPy.
    """
    uncovered = np.ones(n_t, dtype=np.uint8)
    selected = []
    selected_set = set()
    
    while np.sum(uncovered) > 0 and time.time() < deadline:
        # Gains for each block: number of newly covered subsets
        gains = cover_matrix @ uncovered
        
        # Find the block with max gain
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        selected.append(best_idx)
        selected_set.add(best_idx)
        
        # Update uncovered: subsets covered by best_idx become 0
        # uncovered = uncovered & (cover_matrix[best_idx] == 0)
        # This creates a new array. In-place is faster if we manage it carefully, 
        # but numpy boolean indexing is usually fast enough.
        uncovered[cover_matrix[best_idx] == 1] = 0
            
    return selected

def _local_search(selected, cover_matrix, n_blocks, n_t, sub_to_blocks, subset_freq, deadline, rng):
    """
    Local search with deletions and swaps.
    Uses NumPy for count tracking.
    """
    if not selected:
        return selected
    
    # Initialize counts
    # counts[i] = number of selected blocks covering subset i
    counts = np.zeros(n_t, dtype=np.int32)
    for idx in selected:
        counts += cover_matrix[idx]
        
    selected_set = set(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try deletions
        # A block is removable if all subsets it covers have count > 1
        to_remove = -1
        for pos, i in enumerate(selected):
            m = cover_matrix[i]
            # Find indices where m is 1
            subs = np.nonzero(m)[0]
            if len(subs) == 0:
                continue
            # Check if any count is 1
            if np.min(counts[subs]) > 1:
                to_remove = pos
                break
        
        if to_remove != -1:
            i = selected[to_remove]
            counts -= cover_matrix[i]
            selected.pop(to_remove)
            selected_set.discard(i)
            improved = True
            continue
        
        # 2. Try swaps
        # For each block i in selected, find its "unique" or "critical" subsets (count == 1)
        # If no unique subsets, it would have been removed.
        # So we look for a block j not in selected that covers all unique subsets of i.
        
        # Optimization: Prioritize blocks with fewer unique subsets (easier to satisfy)
        # Or blocks covering "rare" subsets? 
        # The "Spectral/Connectivity" idea: Blocks covering rare subsets (low freq) are critical.
        # If a block covers many rare subsets, it's hard to replace.
        # Let's just iterate and try to find a valid swap.
        
        # To speed up, we can randomize the order of selected blocks we try to swap
        order = list(range(len(selected)))
        rng.shuffle(order)
        
        for pos in order:
            if time.time() > deadline:
                break
                
            i = selected[pos]
            m_i = cover_matrix[i]
            
            # Find unique subsets (count == 1)
            unique_mask = (m_i == 1) & (counts == 1)
            unique_indices = np.nonzero(unique_mask)[0]
            
            if len(unique_indices) == 0:
                # Should have been removed, but maybe counts changed? 
                # If no unique, it's redundant.
                # Try removing again? 
                # Actually, if min(counts[subs]) > 1 was false, it means some count is 1.
                # So unique_indices should not be empty.
                continue
                
            # Find candidates j not in selected that cover ALL unique_indices
            # Use the rarest unique subset to minimize candidate list
            # Find the unique subset with the fewest covering blocks
            min_freq = float('inf')
            min_idx = unique_indices[0]
            for u_idx in unique_indices:
                # subset_freq is global frequency, but we need current candidate pool size?
                # sub_to_blocks[u_idx] is the list of ALL blocks covering it.
                # We can filter by not in selected_set.
                # Approximation: use global freq to pick the most constrained one
                if len(sub_to_blocks[u_idx]) < min_freq:
                    min_freq = len(sub_to_blocks[u_idx])
                    min_idx = u_idx
            
            candidates = sub_to_blocks[min_idx]
            
            found_swap = False
            for j in candidates:
                if j in selected_set:
                    continue
                
                m_j = cover_matrix[j]
                
                # Check if j covers all unique_indices
                # m_j[unique_indices] should be all 1
                if np.all(m_j[unique_indices] == 1):
                    # Valid swap candidate
                    # Heuristic: Prefer j that covers more "rare" subsets? 
                    # Or just take the first one?
                    # Let's check if this swap is "beneficial" in terms of connectivity.
                    # A simple metric: Does j cover subsets with lower average frequency?
                    # If yes, it's "more central" or "better".
                    
                    # For now, accept the first valid swap found to keep it fast.
                    # We can randomize candidates if we want diversity.
                    
                    # Perform swap
                    counts -= m_i
                    counts += m_j
                    selected_set.discard(i)
                    selected_set.add(j)
                    selected[pos] = j
                    improved = True
                    found_swap = True
                    break # Restart local search loop after swap
                    
            if found_swap:
                break
                
    return selected

def _perturb(selected, cover_matrix, n_blocks, n_t, sub_to_blocks, rng, num_moves=2):
    """
    Perturb a solution by performing random valid swaps/deletions.
    """
    if not selected:
        return selected
        
    counts = np.zeros(n_t, dtype=np.int32)
    for idx in selected:
        counts += cover_matrix[idx]
        
    selected_set = set(selected)
    sel_list = list(selected)
    
    for _ in range(num_moves):
        # Try a random swap
        # Pick a random block in selected
        pos = rng.randint(0, len(sel_list) - 1)
        i = sel_list[pos]
        m_i = cover_matrix[i]
        
        unique_mask = (m_i == 1) & (counts == 1)
        unique_indices = np.nonzero(unique_mask)[0]
        
        if len(unique_indices) == 0:
            # Can remove?
            subs = np.nonzero(m_i)[0]
            if len(subs) > 0 and np.min(counts[subs]) > 1:
                # Remove
                counts -= m_i
                sel_list.pop(pos)
                selected_set.discard(i)
                continue
            else:
                # Cannot remove, try swap
                # If no unique, it's redundant? No, if min count > 1 is false, some count is 1.
                # So unique_indices should not be empty.
                pass
        
        if len(unique_indices) > 0:
            # Pick random unique subset
            u_idx = rng.choice(unique_indices)
            candidates = [j for j in sub_to_blocks[u_idx] if j not in selected_set]
            
            if candidates:
                # Pick random candidate
                j = rng.choice(candidates)
                m_j = cover_matrix[j]
                
                # Check if j covers all unique_indices
                if np.all(m_j[unique_indices] == 1):
                    counts -= m_i
                    counts += m_j
                    selected_set.discard(i)
                    selected_set.add(j)
                    sel_list[pos] = j
                    
    return sel_list

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 140.0 # Safety margin
    
    blocks, cover_matrix, n_t, index, sub_to_blocks, n_blocks, subset_freq = _prepare(v, k, t)
    
    best_size = float('inf')
    best_selected = []
    
    # Strategy:
    # 1. Initial Greedy
    # 2. Local Search
    # 3. ILS: Perturb and Local Search again
    
    num_restarts = 20 # Adjust based on time
    time_per_restart = (deadline - t0) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        rng = random.Random(42 + r)
        
        if r == 0:
            # Initial greedy
            sel = _greedy_select(cover_matrix, n_blocks, n_t, deadline - 5.0)
        else:
            # Perturb the best solution found so far
            # Or perturb the previous solution? 
            # Perturbing the best is safer.
            if not best_selected:
                sel = _greedy_select(cover_matrix, n_blocks, n_t, deadline - 5.0)
            else:
                sel = _perturb(best_selected, cover_matrix, n_blocks, n_t, sub_to_blocks, rng, num_moves=3)
                
        # Local Search
        sel = _local_search(sel, cover_matrix, n_blocks, n_t, sub_to_blocks, subset_freq, deadline - 5.0, rng)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_selected = list(sel)
            
        # If we found a very good solution (e.g., close to lower bound), we can stop early?
        # Lower bound is 131. If we hit 135, we might want to keep going.
        
    # Final validity check and fallback
    if best_selected:
        # Verify coverage
        covered = np.zeros(n_t, dtype=np.uint8)
        for i in best_selected:
            covered |= cover_matrix[i]
            
        if np.sum(covered) != n_t:
            # Fallback: add any missing blocks greedily
            uncovered = (covered == 0)
            selected_set = set(best_selected)
            while np.sum(uncovered) > 0 and time.time() < deadline - 1.0:
                gains = cover_matrix @ uncovered
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                best_selected.append(best_idx)
                selected_set.add(best_idx)
                uncovered[cover_matrix[best_idx] == 1] = 0

    # Return list of blocks (as lists of ints)
    result = [list(blocks[i]) for i in best_selected]
    return result

