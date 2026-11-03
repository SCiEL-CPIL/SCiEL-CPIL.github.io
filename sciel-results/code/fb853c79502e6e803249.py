"""Covering design C(19,7,4) via greedy + local search with NumPy bitsets."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from scipy.special import comb

# Precompute all 4-subsets of 19 points as indices
_V = 19
_K = 7
_T = 4

# Build the mapping from 4-subset to index
_all_t_subsets = list(itertools.combinations(range(_V), _T))
_T_COUNT = len(_all_t_subsets)  # C(19,4) = 3876
_t_subset_to_idx = {s: i for i, s in enumerate(_all_t_subsets)}

# Precompute for each 7-block, which 4-subsets it covers
# Represent as a bitset (array of uint64)
_NUM_BLOCKS_C19_7 = int(comb(19, 7, exact=True))  # 50388

# We'll build the incidence matrix lazily or use a function
# For speed, precompute the bitsets for all 7-blocks

def _build_block_bitsets():
    """Precompute bitsets for all C(19,7) blocks."""
    bitsets = np.zeros((_NUM_BLOCKS_C19_7, (_T_COUNT + 63) // 64), dtype=np.uint64)
    block_list = list(itertools.combinations(range(_V), _K))
    for bi, block in enumerate(block_list):
        for t_sub in itertools.combinations(block, _T):
            idx = _t_subset_to_idx[t_sub]
            bitsets[bi, idx // 64] |= (1 << (idx % 64))
    return bitsets, block_list

# Precompute
_BLOCK_BITSETS, _BLOCK_LIST = _build_block_bitsets()

# Also build a reverse map: for each 4-subset, which blocks cover it
# This is expensive (3876 * 50388), so we'll compute on the fly or use a different approach

def _block_index(block):
    """Get the index of a block (tuple of 7 sorted ints) in _BLOCK_LIST."""
    # Use a dictionary for O(1) lookup
    return _block_to_idx.get(tuple(block), -1)

# Build the lookup
_block_to_idx = {b: i for i, b in enumerate(_BLOCK_LIST)}


def _popcount_u64(arr):
    """Count set bits in an array of uint64."""
    return np.sum(arr.view(np.uint64).astype(np.int32), axis=None) if arr.size else 0


def _bitset_popcount(arr):
    """Count set bits in a numpy array of uint64."""
    # Use a lookup table for bytes
    if arr.size == 0:
        return 0
    # Convert to bytes and count
    b = arr.tobytes()
    # Use a precomputed table
    return _BYTE_POPCOUNT[b].sum()


# Precompute byte popcount table
_BYTE_POPCOUNT = np.zeros(256, dtype=np.int32)
for i in range(256):
    _BYTE_POPCOUNT[i] = bin(i).count('1')


def _count_uncovered_in_block(block_indices, uncovered_bits):
    """Count how many uncovered 4-subsets are covered by a given block."""
    if not block_indices:
        return 0
    # OR all the block bitsets
    combined = np.zeros_like(uncovered_bits[:1])
    for bi in block_indices:
        combined |= _BLOCK_BITSETS[bi]
    # AND with uncovered
    masked = combined & uncovered_bits
    # Count bits
    b = masked.tobytes()
    return int(_BYTE_POPCOUNT[b].sum())


def _greedy_construct(v, k, t, time_limit=100.0):
    """Greedy covering with NumPy bitsets."""
    start_time = time.time()
    
    # Initialize uncovered bitset
    uncovered = np.ones((_T_COUNT + 63) // 64, dtype=np.uint64)
    # Set all bits to 1
    for i in range(_T_COUNT):
        uncovered[i // 64] |= (1 << (i % 64))
    # Clear extra bits in the last word
    if _T_COUNT % 64 != 0:
        uncovered[-1] &= ((1 << (_T_COUNT % 64)) - 1)
    
    blocks = []
    used_blocks = set()
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Check if all covered
        if not np.any(uncovered):
            break
        
        # Find the best block to add
        # For each unused block, compute how many uncovered t-subsets it covers
        # We'll do this in chunks for efficiency
        
        best_gain = -1
        best_block = None
        
        # Iterate over all blocks
        for bi in range(_NUM_BLOCKS_C19_7):
            if bi in used_blocks:
                continue
            # Compute gain
            combined = _BLOCK_BITSETS[bi] & uncovered
            b = combined.tobytes()
            gain = int(_BYTE_POPCOUNT[b].sum())
            if gain > best_gain:
                best_gain = gain
                best_block = bi
                if gain == 0:
                    break
        
        if best_block is None or best_gain <= 0:
            # Should not happen if uncovered is non-empty
            break
        
        used_blocks.add(best_block)
        blocks.append(_BLOCK_LIST[best_block])
        
        # Update uncovered
        uncovered &= ~_BLOCK_BITSETS[best_block]
    
    return blocks


def _local_search(blocks, v, k, t, time_limit=100.0):
    """Local search: try to remove or swap blocks to reduce count."""
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    def _is_valid(block_list):
        """Check if the block list covers all t-subsets."""
        if time.time() - start_time > time_limit:
            return False
        # Build the union of all block bitsets
        if not block_list:
            return False
        union = np.zeros((_T_COUNT + 63) // 64, dtype=np.uint64)
        for b in block_list:
            bi = _block_to_idx.get(tuple(b), -1)
            if bi >= 0:
                union |= _BLOCK_BITSETS[bi]
            else:
                # Manually compute
                for t_sub in itertools.combinations(b, t):
                    idx = _t_subset_to_idx[t_sub]
                    union[idx // 64] |= (1 << (idx % 64))
        # Check if all bits are set
        for i in range(_T_COUNT):
            if not (union[i // 64] >> (i % 64)) & 1:
                return False
        return True
    
    def _try_remove(block_list):
        """Try to remove a block."""
        for i in range(len(block_list)):
            if time.time() - start_time > time_limit:
                return block_list, False
            new_list = block_list[:i] + block_list[i+1:]
            if _is_valid(new_list):
                return new_list, True
        return block_list, False
    
    def _try_swap(block_list):
        """Try to swap one block for another."""
        for i in range(len(block_list)):
            if time.time() - start_time > time_limit:
                return block_list, False
            for bi in range(_NUM_BLOCKS_C19_7):
                if bi in [tuple(b) for b in block_list]:
                    continue
                new_list = block_list[:i] + [list(_BLOCK_LIST[bi])] + block_list[i+1:]
                if _is_valid(new_list):
                    return new_list, True
        return block_list, False
    
    # Iteratively try to improve
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try removals
        blocks, removed = _try_remove(blocks)
        if removed:
            improved = True
            continue
        # Try swaps (only if we have time)
        if time.time() - start_time < time_limit * 0.8:
            blocks, swapped = _try_swap(blocks)
            if swapped:
                improved = True
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t)."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Phase 1: Greedy construction
    time_for_greedy = min(60.0, (deadline - start_time) * 0.4)
    blocks = _greedy_construct(v, k, t, time_limit=time_for_greedy)
    
    if time.time() > deadline:
        return blocks
    
    # Phase 2: Local search
    time_for_ls = deadline - time.time() - 5.0  # 5s safety margin
    if time_for_ls > 10:
        blocks = _local_search(blocks, v, k, t, time_limit=time_for_ls)
    
    return blocks
# EVOLVE-BLOCK-END
