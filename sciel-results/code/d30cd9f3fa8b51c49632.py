import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Precomputation
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute coverage lists for each block
    block_triple_indices = [[] for _ in range(num_blocks)]
    for i, b in enumerate(all_blocks):
        indices = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triple_indices[i] = indices

    # Reverse index: triple -> list of blocks covering it
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, indices in enumerate(block_triple_indices):
        for idx in indices:
            triple_to_blocks[idx].append(i)

    def greedy_construct(random_seed):
        """
        Constructs a covering using a weighted greedy algorithm with efficient gain updates.
        Randomness is introduced by shuffling the order of blocks with equal gain.
        """
        rng = random.Random(random_seed)
        
        # gain[i] is the number of currently uncovered triples covered by block i
        # Initially, all triples are uncovered, so gain[i] = len(block_triple_indices[i])
        gain = np.array([len(indices) for indices in block_triple_indices], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_indices = []
        
        while is_uncovered.any():
            if time.time() > deadline - 10:
                break
            
            max_gain = np.max(gain)
            if max_gain == 0:
                # Should not happen if is_uncovered.any() is true and blocks are valid
                break
            
            # Find indices with max gain
            # np.where returns an array of indices
            candidates = np.where(gain == max_gain)[0]
            
            # Randomly select from candidates
            selected_idx = rng.choice(candidates.tolist())
            selected_indices.append(selected_idx)
            
            # Update gain and is_uncovered
            # For each triple in the selected block:
            for tri_idx in block_triple_indices[selected_idx]:
                if is_uncovered[tri_idx]:
                    is_uncovered[tri_idx] = False
                    # For every block that covers this triple, decrease its gain
                    # Because this triple is no longer "newly covered" by them
                    for other_b_idx in triple_to_blocks[tri_idx]:
                        gain[other_b_idx] -= 1
                        
        return selected_indices

    def prune_solution(block_indices):
        """
        Removes redundant blocks from the solution.
        A block is redundant if every triple it covers is also covered by another block in the set.
        """
        if not block_indices:
            return []
        
        # Convert to list for modification
        current_set = list(block_indices)
        
        # Count coverage for each triple
        covered_counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in current_set:
            for tri_idx in block_triple_indices[b_idx]:
                covered_counts[tri_idx] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(current_set)):
                b_idx = current_set[i]
                # Check if b_idx is redundant
                redundant = True
                for tri_idx in block_triple_indices[b_idx]:
                    if covered_counts[tri_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove b_idx
                    current_set.pop(i)
                    for tri_idx in block_triple_indices[b_idx]:
                        covered_counts[tri_idx] -= 1
                    changed = True
                    if time.time() > deadline - 5:
                        return current_set
                    # Restart loop because indices changed
                    break
        
        return current_set

    def local_search_improve(blocks, time_limit=5.0):
        """
        Attempts to improve the solution by swapping or removing blocks.
        This is a simple local search to refine the greedy result.
        """
        if not blocks:
            return blocks
        
        current_blocks = list(blocks)
        current_size = len(current_blocks)
        
        # Precompute coverage counts
        covered_counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in current_blocks:
            for tri_idx in block_triple_indices[b_idx]:
                covered_counts[tri_idx] += 1
        
        # Check if already valid (should be, but just in case)
        if not np.all(covered_counts >= 1):
             return current_blocks

        start_ls = time.time()
        
        # 1. Try to remove redundant blocks again (just in case prune missed something due to order)
        changed = True
        while changed:
            if time.time() - start_ls > time_limit:
                break
            changed = False
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                redundant = True
                for tri_idx in block_triple_indices[b_idx]:
                    if covered_counts[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    current_blocks.pop(i)
                    for tri_idx in block_triple_indices[b_idx]:
                        covered_counts[tri_idx] -= 1
                    changed = True
                    break
        
        # 2. Try to replace one block with two? No, that increases size.
        # 3. Try to swap a block for another that covers the same "essential" triples but allows removal of others?
        #    This is complex. Let's stick to simple removals and maybe random restarts of pruning.
        
        # A more effective heuristic: Randomly remove a block, if coverage breaks, find a replacement?
        # Or: For each block, try to replace it with a block that has higher "efficiency" (covers more uncovered by others?)
        # Actually, since we are minimizing count, the main gain comes from finding a smaller set.
        
        # Let's try a simple hill climbing: 
        # Pick a random block B. 
        # Remove it. 
        # If coverage is broken, find a block B' that covers the most lost triples.
        # If B' is not already in the set, swap B with B'.
        # This doesn't reduce size, but might lead to a state where a block becomes removable.
        
        # Given time constraints, simple pruning is often the biggest win.
        # Let's just return the pruned set.
        
        return current_blocks

    best_solution = None
    best_size = float('inf')
    
    # Run multiple greedy constructions with different random seeds
    # The randomness comes from tie-breaking.
    
    num_starts = 30 # Increase starts as gain update is O(degree) not O(N)
    
    for seed in range(num_starts):
        if time.time() > deadline - 15:
            break
            
        raw_solution = greedy_construct(seed)
        
        if not raw_solution:
            continue
            
        # Prune
        pruned_solution = prune_solution(raw_solution)
        
        # Local search polish (lightweight)
        # Note: local_search_improve currently just does more pruning.
        # We could add more sophisticated logic here if time permits, 
        # but for now, robust pruning is key.
        polished_solution = local_search_improve(pruned_solution, time_limit=2.0)
        
        # Verify validity
        if len(polished_solution) < best_size:
            covered = np.zeros(num_triples, dtype=np.int32)
            for b_idx in polished_solution:
                for tri_idx in block_triple_indices[b_idx]:
                    covered[tri_idx] += 1
            
            if np.all(covered >= 1):
                best_solution = polished_solution
                best_size = len(polished_solution)
                
                # If we hit the lower bound (69), we can stop
                if best_size <= 69:
                    break

    if best_solution is None:
        # Fallback: Return a large valid covering
        # Use the first greedy result without pruning? Or just generate one.
        # If greedy failed, something is wrong. 
        # Let's just do a simple greedy without time limit check for fallback.
        raw_solution = greedy_construct(0)
        if not raw_solution:
             # Absolute fallback: return all blocks (valid but huge score)
             return [all_blocks[i] for i in range(num_blocks)]
        best_solution = raw_solution

    # Convert indices to blocks
    result_blocks = [all_blocks[i] for i in best_solution]
    
    return result_blocks

if __name__ == "__main__":
    # Test with small parameters
    # v=7, k=3, t=2 -> C(7,3,2) = 7 blocks (Fano plane)
    # blocks = construct(7, 3, 2)
    # print(len(blocks))
    pass
