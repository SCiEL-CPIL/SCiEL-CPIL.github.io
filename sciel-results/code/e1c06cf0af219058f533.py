"""Combinatorial Holography covering via spectral residual energy over Z19."""
import itertools
import numpy as np
import random
import time

# Precomputation for v=19, k=7, t=4
_V = 19
_K = 7
_T = 4

# Build list of all 4-subsets
_all_4subs = list(itertools.combinations(range(_V), _T))
_n4 = len(_all_4subs)  # 3876

# Map each 4-subset to its index
_subset_index = {s: i for i, s in enumerate(_all_4subs)}

# Build list of all 7-blocks
_all_blocks = list(itertools.combinations(range(_V), _K))
_nb = len(_all_blocks)  # 50388

# For each block, compute which 4-subsets it covers (as a bitset using Python int)
_block_masks = []
# Also store the list of subset indices for each block for fast count updates
_block_subset_indices = []
for blk in _all_blocks:
    mask = 0
    indices = []
    for quad in itertools.combinations(blk, _T):
        idx = _subset_index[quad]
        mask |= (1 << idx)
        indices.append(idx)
    _block_masks.append(mask)
    _block_subset_indices.append(indices)

# For each 4-subset, list of block indices that contain it
_subset_to_blocks = [[] for _ in range(_n4)]
for bi, blk in enumerate(_all_blocks):
    for quad in itertools.combinations(blk, _T):
        _subset_to_blocks[_subset_index[quad]].append(bi)

# Spectral Holography Components
# We use the DFT of the cyclic group Z19 to analyze the "shape" of the uncovered set.
# The idea is that certain configurations of uncovered 4-subsets have high spectral energy
# at specific frequencies. We prefer blocks that cover subsets corresponding to these
# high-energy components, as they may break up "hard" clusters more effectively.
omega = np.exp(2j * np.pi / _V)
# Precompute DFT of all possible 4-subsets? No, that's too much.
# Instead, we compute the DFT of the *indicator vector* of the uncovered set on the fly
# or use a proxy.
# Actually, for v=19, we can map each point to a frequency component.
# A 4-subset S covers points p in S.
# The "Spectral Energy" of a set of subsets U can be approximated by the energy of the
# sum of their point indicators.
# Let's precompute the point indicators for each 4-subset in the frequency domain?
# No, let's keep it simpler:
# We will compute the DFT of the *uncovered point count* vector (length 19).
# If a point is in many uncovered 4-subsets, it has high "pressure".
# We want to cover points with high pressure.
# This is a "Point-Pressure" heuristic, which is a time-domain approximation of spectral energy.

# Precompute for each block, the sum of "pressures" it covers?
# Pressure changes dynamically.

def _compute_point_pressures(uncovered_mask):
    """
    Computes how many uncovered 4-subsets contain each point.
    Returns a list of 19 integers.
    """
    pressures = [0] * _V
    # Iterate over all 4-subsets in the uncovered mask
    # This is expensive if done naively.
    # Optimization: We can precompute for each point, the mask of 4-subsets containing it.
    # point_to_subsets_mask[p] is a bitset of 4-subsets containing p.
    # Then pressure[p] = bit_count(uncovered_mask & point_to_subsets_mask[p])
    return pressures

# Precompute point-to-subset masks for fast pressure calculation
_point_to_subset_masks = []
for p in range(_V):
    mask = 0
    for quad in _all_4subs:
        if p in quad:
            idx = _subset_index[quad]
            mask |= (1 << idx)
    _point_to_subset_masks.append(mask)

def _get_point_pressures(uncovered_mask):
    """Fast calculation of point pressures using precomputed masks."""
    pressures = [0] * _V
    for p in range(_V):
        pressures[p] = (uncovered_mask & _point_to_subset_masks[p]).bit_count()
    return pressures

def _spectral_score_block(bi, pressures):
    """
    Scores a block based on the sum of pressures of the points it contains.
    Higher score = better choice to reduce "hard" uncovered clusters.
    """
    blk = _all_blocks[bi]
    score = sum(pressures[p] for p in blk)
    return score

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Uses Randomized Greedy with Spectral-Pressure Heuristic and Local Search.
    """
    if v != _V or k != _K or t != _T:
        return _simple_greedy(v, k, t)
    
    best_solution = None
    best_size = float('inf')
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Number of restarts to try.
    # We will run as many as possible within the time limit.
    # Each restart: Greedy (~1s) + Local Search (~0.5s) = 1.5s.
    # ~70 restarts possible.
    
    restart_count = 0
    while time.time() - start_time < time_limit:
        restart_count += 1
        uncovered = (1 << _n4) - 1
        selected_blocks = []
        cov_counts = [0] * _n4
        
        # --- Phase 1: Greedy with Spectral-Pressure Tie-Breaking ---
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            # Calculate point pressures for tie-breaking
            # This is O(V * bit_count) which is fast (19 * ~20 ops)
            pressures = _get_point_pressures(uncovered)
            
            # Find max gain
            for bi in range(_nb):
                gain = (uncovered & _block_masks[bi]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [bi]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(bi)
            
            if not best_candidates:
                break
                
            # Tie-breaking: Choose the block with the highest spectral-pressure score
            # among the max-gain candidates.
            # If there are many candidates, sampling is faster.
            if len(best_candidates) > 20:
                # Sample 20 candidates and pick best pressure
                sampled = random.sample(best_candidates, 20)
                chosen_bi = max(sampled, key=lambda b: _spectral_score_block(b, pressures))
            else:
                chosen_bi = max(best_candidates, key=lambda b: _spectral_score_block(b, pressures))
                
            selected_blocks.append(chosen_bi)
            
            # Update state
            mask = _block_masks[chosen_bi]
            uncovered &= ~mask
            
            for idx in _block_subset_indices[chosen_bi]:
                cov_counts[idx] += 1
        
        # --- Phase 2: Local Search (Deletion + Swap) ---
        # First, aggressive deletion
        changed = True
        while changed:
            changed = False
            random.shuffle(selected_blocks)
            new_selected = []
            for bi in selected_blocks:
                redundant = True
                for idx in _block_subset_indices[bi]:
                    if cov_counts[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for idx in _block_subset_indices[bi]:
                        cov_counts[idx] -= 1
                    changed = True
                else:
                    new_selected.append(bi)
            selected_blocks = new_selected
        
        # Second, try to swap blocks to reduce size (if we have "slack")
        # Actually, deletion already minimizes size for the current set.
        # To find a *smaller* set, we need to replace a block with a different one
        # that covers the same unique parts but allows other blocks to be removed.
        # This is complex. Instead, we rely on multiple restarts with different
        # random choices in the greedy phase.
        
        # --- Phase 3: Simulated Annealing on Block Removal ---
        # Try to remove more blocks by temporarily allowing "uncovered" sets
        # and then re-adding better blocks? 
        # This is essentially a larger search.
        
        # Simple improvement: If we have time, try a few random swaps.
        # Pick a block A, remove it. If uncovered increases, try to find a block B
        # that covers the new uncovered sets better than A did?
        # This is heuristic.
        
        # Let's stick to the deletion result as the candidate.
        
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = [list(_all_blocks[bi]) for bi in selected_blocks]
            
            if best_size <= 131:
                break

    return best_solution

def _simple_greedy(v, k, t):
    """Fallback simple greedy for non-standard parameters."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
