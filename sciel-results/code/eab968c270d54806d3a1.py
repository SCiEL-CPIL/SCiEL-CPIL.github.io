"""Rewritten covering design program: vectorized greedy + local search polish."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
from collections import Counter

# Deadline guard: kill before the 150s wall-clock limit with safety margin.
_DEADLINE = 150.0
_MARGIN = 15.0  # seconds to leave before the hard kill


def _elapsed():
    return time.time() - _START


def _time_left():
    return _DEADLINE - _MARGIN - _elapsed()


def construct(v, k, t):
    global _START
    _START = time.time()

    # Precompute all t-subsets and map them to integer indices.
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    # Map each t-subset to an index for fast lookup.
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute which t-subsets it participates in.
    # point_tsubs[p] = list of indices of t-subsets containing p
    point_tsubs = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].append(idx)

    # Coverage array: 1 if covered, 0 otherwise
    covered = np.zeros(n_tsubs, dtype=np.uint8)
    # Count how many blocks cover each t-subset (for redundancy removal)
    cover_count = np.zeros(n_tsubs, dtype=np.int32)

    blocks = []  # list of frozensets
    block_to_tsubs = []  # for each block, the set of t-subset indices it covers

    # --- Greedy construction with vectorized gain ---
    while n_tsubs - covered.sum() > 0 and _time_left() > 5.0:
        # Pick the first uncovered t-subset to seed the block
        uncovered_mask = 1 - covered
        first_uncovered_idx = int(np.argmax(uncovered_mask))
        if uncovered_mask[first_uncovered_idx] == 0:
            break
        seed_tsub = all_tsubs[first_uncovered_idx]
        block = list(seed_tsub)

        # Greedily add points to maximize new coverage
        while len(block) < k and _time_left() > 3.0:
            block_set = set(block)
            # Compute gain for each candidate point not in block
            best_gain = -1
            best_point = -1
            # We can vectorize: for each candidate, count how many currently-uncovered
            # t-subsets would be covered if we add it.
            # A t-subset s is covered by block∪{x} if s ⊆ block∪{x} and s is uncovered.
            # This means s ⊆ block or (s\block == {x}) i.e., s has exactly one element outside block and that element is x.
            # Actually: s ⊆ block∪{x} means all elements of s are in block or equal to x.
            # So s must be a subset of block∪{x}. Since |s|=t and |block|=b, we need t-b <= 1 new elements from x.
            # More precisely: s ⊆ block∪{x} iff s∩(complement of block∪{x}) = empty.
            
            # Faster approach: for each candidate x, the new t-subsets covered are those
            # uncovered t-subsets s such that s ⊆ block ∪ {x}.
            # Since s has t elements and block has b elements, s can have at most 1 element
            # outside block (namely x). So s must be: (t-1 elements from block) ∪ {x},
            # or (t elements from block) [already covered if block has >= t elements and they form a t-subset].
            
            # Let's just compute directly for candidates
            candidates = [x for x in range(v) if x not in block_set]
            if not candidates:
                break
            
            # For each candidate, compute gain
            gains = np.zeros(len(candidates), dtype=np.int32)
            for ci, x in enumerate(candidates):
                # t-subsets covered by adding x: those s where s ⊆ block∪{x} and s not yet covered
                # s must contain x and (t-1) elements from block, OR s ⊆ block (but those are already handled)
                # Actually we need s ⊆ block ∪ {x}. Since x ∉ block, this means s contains x
                # and the remaining t-1 elements are from block. But also s could be entirely in block
                # (if |block| >= t), but those would already be in the "covered by block" set.
                # Wait - we're building the block incrementally. The t-subsets that will be newly
                # covered by the FINAL block are all t-subsets of the final block. But during
                # construction, we want to maximize incremental gain.
                # 
                # Simpler: gain(x) = number of currently uncovered t-subsets s such that s ⊆ block∪{x}
                # and s is not a subset of block (i.e., s contains x).
                # These are: choose (t-1) elements from block, combine with x.
                # But only if |block| >= t-1.
                if len(block) >= t - 1:
                    # Count uncovered t-subsets of form {x} ∪ C where C ⊆ block, |C| = t-1
                    # We can check each such t-subset
                    count = 0
                    # Generate (t-1)-subsets of block
                    for comb in itertools.combinations(block, t-1):
                        s = tuple(sorted(comb + (x,)))
                        idx = tsub_to_idx.get(s)
                        if idx is not None and not covered[idx]:
                            count += 1
                    gains[ci] = count
                else:
                    gains[ci] = 0
            
            # Pick best
            max_idx = int(np.argmax(gains))
            if gains[max_idx] <= 0:
                # No gain from any candidate; just pick the first
                best_point = candidates[0]
            else:
                best_point = candidates[max_idx]
            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        block_frozen = frozenset(block)
        if block_frozen in blocks:
            # Duplicate; skip
            continue
        blocks.append(block_frozen)
        
        # Update coverage
        new_tsubs_idx = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            if not covered[idx]:
                covered[idx] = 1
                new_tsubs_idx.append(idx)
            cover_count[idx] += 1
        block_to_tsubs.append(set(range(len(block_to_tsubs))))  # placeholder
        
        # Store the actual t-subset indices for this block
        block_tsub_indices = set()
        for ts in itertools.combinations(block, t):
            block_tsub_indices.add(tsub_to_idx[ts])
        block_to_tsubs[-1] = block_tsub_indices

    # --- Local search: redundant block removal ---
    if _time_left() > 2.0:
        # Try to remove blocks that don't uniquely cover any t-subset
        changed = True
        while changed and _time_left() > 2.0:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if len(blocks) <= 1:
                    break
                # Check if block i is the only cover for any t-subset
                redundant = True
                for idx in block_to_tsubs[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for idx in block_to_tsubs[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_to_tsubs.pop(i)
                    changed = True

    # --- 1-exchange local search: try replacing a block with a better one ---
    if _time_left() > 5.0:
        # For each block, try to find a replacement that covers more currently-uniquely-covered t-subsets
        # This is expensive, so do a limited number of passes
        for _pass in range(3):
            if _time_left() < 3.0:
                break
            improved = False
            for i in range(len(blocks)):
                if _time_left() < 3.0:
                    break
                # Find t-subsets uniquely covered by block i
                unique_to_i = [idx for idx in block_to_tsubs[i] if cover_count[idx] == 1]
                if not unique_to_i:
                    continue
                # Try to find a new block that covers all of unique_to_i
                # The new block must contain the union of all points in those t-subsets
                required_points = set()
                for idx in unique_to_i:
                    required_points.update(all_tsubs[idx])
                if len(required_points) > k:
                    continue
                # Can we form a valid block containing required_points?
                if len(required_points) <= k:
                    # Try a few random completions
                    remaining = [p for p in range(v) if p not in required_points]
                    need = k - len(required_points)
                    if need <= 0:
                        new_block = frozenset(required_points)
                    elif need <= len(remaining):
                        # Try a few random choices
                        best_new_block = None
                        best_gain = 0
                        for _trial in range(5):
                            chosen = set(np.random.choice(remaining, size=need, replace=False).tolist())
                            candidate = frozenset(required_points | chosen)
                            if candidate in blocks:
                                continue
                            # Compute how many new t-subsets this would cover
                            new_covered = 0
                            for ts in itertools.combinations(candidate, t):
                                idx = tsub_to_idx[ts]
                                if not covered[idx]:
                                    new_covered += 1
                            if new_covered > best_gain:
                                best_gain = new_covered
                                best_new_block = candidate
                        if best_new_block is not None and best_gain > 0:
                            # Replace block i with best_new_block
                            # Remove old coverage
                            for idx in block_to_tsubs[i]:
                                cover_count[idx] -= 1
                                if cover_count[idx] == 0:
                                    covered[idx] = 0
                            blocks[i] = best_new_block
                            # Add new coverage
                            new_tsubs = set()
                            for ts in itertools.combinations(best_new_block, t):
                                idx = tsub_to_idx[ts]
                                new_tsubs.add(idx)
                                if not covered[idx]:
                                    covered[idx] = 1
                                cover_count[idx] += 1
                            block_to_tsubs[i] = new_tsubs
                            improved = True
                            break  # Restart the loop after a change

    # Convert frozensets to sorted lists
    result = [sorted(b) for b in blocks]
    return result

# EVOLVE-BLOCK-END
