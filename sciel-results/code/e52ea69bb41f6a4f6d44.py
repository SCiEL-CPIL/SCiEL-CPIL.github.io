"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    
    Strategy:
    1. Use a greedy algorithm with a smart point selection heuristic.
    2. Use a local search / improvement phase to try to reduce the number of blocks.
    3. Use a time guard to ensure we finish within the deadline.
    
    Returns: list of blocks (each block is a list of k distinct integers in 0..v-1)
    """
    start_time = time.time()
    deadline = 115.0  # Leave 5 seconds safety margin
    
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    total_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Track which t-subsets are covered
    # Use a bitset-like approach with a list of booleans
    covered = [False] * total_tsubsets
    
    def is_covered(ts):
        return covered[tsubset_to_idx[ts]]
    
    def mark_covered(block):
        for ts in itertools.combinations(block, t):
            covered[tsubset_to_idx[ts]] = True
    
    # Greedy construction with smart point selection
    def greedy_construct():
        blocks = []
        uncovered_count = total_tsubsets
        
        # Precompute for each point, the t-subsets that contain it
        # This helps in quickly computing gain
        
        while uncovered_count > 0:
            if time.time() > deadline:
                return blocks
            
            # Find the first uncovered t-subset
            first_uncovered = None
            for i, is_cov in enumerate(covered):
                if not is_cov:
                    first_uncovered = all_tsubsets[i]
                    break
            
            if first_uncovered is None:
                break
            
            # Start a new block with the elements of the first uncovered t-subset
            block = set(first_uncovered)
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                if time.time() > deadline:
                    break
                
                best_point = None
                best_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Compute gain: number of uncovered t-subsets that would be covered
                    # by adding x to the block
                    gain = 0
                    # For each (t-1)-subset of the current block, check if adding x
                    # creates a t-subset that is uncovered
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if not covered[tsubset_to_idx[ts]]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is not None and best_gain > 0:
                    block.add(best_point)
                elif best_point is not None:
                    # If no gain, just add any available point
                    for x in range(v):
                        if x not in block:
                            block.add(x)
                            break
                else:
                    break
            
            if len(block) == k:
                block = sorted(block)
                blocks.append(block)
                # Mark all t-subsets in this block as covered
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx[ts]
                    if not covered[idx]:
                        covered[idx] = True
                        uncovered_count -= 1
        
        return blocks
    
    # Local search: try to replace a block with a better one
    def local_search_improve(blocks):
        """Try to improve the covering by replacing blocks."""
        if time.time() > deadline:
            return blocks
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to remove redundant blocks
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                
                # Temporarily remove block i
                saved_block = blocks.pop(i)
                
                # Check if all t-subsets in saved_block are still covered
                redundant = True
                for ts in itertools.combinations(saved_block, t):
                    if not is_covered(ts):
                        redundant = False
                        break
                
                if redundant:
                    improved = True
                else:
                    blocks.insert(i, saved_block)
        
        return blocks
    
    # Try to construct a good initial covering
    blocks = greedy_construct()
    
    # Apply local search improvements
    if time.time() < deadline:
        blocks = local_search_improve(blocks)
    
    # If we still have time, try a second greedy pass with different ordering
    if time.time() < deadline - 10:
        # Reset covered array
        covered = [False] * total_tsubsets
        
        # Try a different greedy strategy: prioritize t-subsets with fewer candidates
        blocks2 = []
        uncovered_count = total_tsubsets
        
        # Precompute: for each t-subset, how many k-subsets contain it
        # This is C(v-t, k-t) for all t-subsets (they're all the same)
        
        while uncovered_count > 0 and time.time() < deadline:
            # Find an uncovered t-subset
            first_uncovered = None
            for i, is_cov in enumerate(covered):
                if not is_cov:
                    first_uncovered = all_tsubsets[i]
                    break
            
            if first_uncovered is None:
                break
            
            block = set(first_uncovered)
            
            # Add points greedily
            while len(block) < k and time.time() < deadline:
                best_point = None
                best_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if not covered[tsubset_to_idx[ts]]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is not None and best_gain > 0:
                    block.add(best_point)
                elif best_point is not None:
                    for x in range(v):
                        if x not in block:
                            block.add(x)
                            break
                else:
                    break
            
            if len(block) == k:
                block = sorted(block)
                blocks2.append(block)
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx[ts]
                    if not covered[idx]:
                        covered[idx] = True
                        uncovered_count -= 1
        
        # Use the better of the two coverings
        if len(blocks2) < len(blocks):
            blocks = blocks2
    
    # Final cleanup: remove any duplicate blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks
# EVOLVE-BLOCK-END
