import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmasks for each block
    # mask[i] is an integer where bit j is set if block i covers t-subset j
    block_masks = [0] * num_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    full_mask = (1 << num_tsubs) - 1

    # 2. Build Conflict/Compatibility Graph for Clique Search
    # We want to find a set of blocks (a clique in a compatibility graph) 
    # that covers all triples.
    # Compatibility: Two blocks A and B are compatible if adding B to a set containing A 
    # doesn't "conflict" in a way that makes it harder to cover. 
    # Actually, for a pure clique problem, we usually define edges based on pairwise constraints.
    # Here, the constraint is global coverage. 
    # A better heuristic for "Clique-like" search in covering:
    # We search for a maximal set of blocks such that no block is redundant.
    # However, standard Max Clique finds a set where every pair is connected.
    # Let's define "Connection" as: Block A and Block B are connected if 
    # (mask[A] | mask[B]) > mask[A] and (mask[A] | mask[B]) > mask[B]? No, that's always true unless one is subset of other.
    # 
    # Alternative Structural Approach: 
    # Use a "Greedy Clique" style selection but on a graph where edges represent 
    # "Non-Redundancy" or "High Overlap Efficiency".
    # 
    # Let's try a different structural change: **Iterated Local Search (ILS) with a 
    # specialized "Swap-Remove" move that uses pre-computed candidate lists.**
    # The previous SA was too slow because it picked random blocks from 346k.
    # We will restrict the search space to a "candidate pool" of blocks that cover 
    # "hard" triples (triples with fewest covering blocks).
    
    # Calculate frequency of each triple in the entire universe of blocks
    triple_freq = [0] * num_tsubs
    for mask in block_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_freq[idx] += 1
            m ^= lsb
            
    # Identify "Hard" triples: those covered by the fewest blocks
    # Sort triples by frequency
    sorted_triples = sorted(range(num_tsubs), key=lambda x: triple_freq[x])
    
    # Build inverted index: triple_idx -> list of block indices covering it
    triple_to_blocks = [[] for _ in range(num_tsubs)]
    for bi, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(bi)
            m ^= lsb

    # 3. Initial Solution: Greedy with Priority on Hard Triples
    def run_greedy_hard():
        covered = 0
        selected = []
        used_blocks = set()
        
        while covered != full_mask and time.time() < deadline - 60:
            # Find the uncovered triple with the minimum frequency (hardest to cover)
            # To do this efficiently, we iterate through sorted_triples and check if uncovered
            target_idx = -1
            for t_idx in sorted_triples:
                if (covered >> t_idx) & 1:
                    continue
                target_idx = t_idx
                break
            
            if target_idx == -1:
                break
                
            candidates = triple_to_blocks[target_idx]
            best_gain = -1
            best_block = -1
            
            # Randomize candidates to break ties and allow diversity
            random.shuffle(candidates)
            
            for bi in candidates:
                if bi in used_blocks:
                    continue
                gain = (block_masks[bi] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
                    
            if best_block == -1:
                # Fallback: pick any from candidates
                for bi in candidates:
                    if bi not in used_blocks:
                        best_block = bi
                        break
                if best_block == -1:
                    break
                    
            selected.append(best_block)
            used_blocks.add(best_block)
            covered |= block_masks[best_block]
            
        return selected

    current_blocks_set = set(run_greedy_hard())
    
    # Ensure validity
    current_mask = 0
    for bi in current_blocks_set:
        current_mask |= block_masks[bi]
        
    if current_mask != full_mask:
        # Complete with simple greedy
        while current_mask != full_mask and time.time() < deadline - 50:
            uncovered = full_mask ^ current_mask
            # Pick a random uncovered triple
            t_idx = -1
            for _ in range(100):
                t_idx = random.randint(0, num_tsubs - 1)
                if (uncovered >> t_idx) & 1:
                    break
            if t_idx == -1:
                break
            candidates = triple_to_blocks[t_idx]
            random.shuffle(candidates)
            for bi in candidates:
                if bi not in current_blocks_set:
                    current_blocks_set.add(bi)
                    current_mask |= block_masks[bi]
                    break

    best_solution = list(current_blocks_set)
    best_count = len(best_solution)

    # 4. Iterated Local Search (ILS)
    # Move: Pick a block to remove (if possible) or swap.
    # Key Improvement: Pre-compute "Critical" blocks. 
    # A block is critical if removing it breaks coverage.
    # We maintain a count of how many times each triple is covered by the current set.
    
    def get_counts(block_set):
        counts = [0] * num_tsubs
        for bi in block_set:
            m = block_masks[bi]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    def is_redundant(bi, counts):
        m = block_masks[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if counts[idx] == 1:
                return False
            m ^= lsb
        return True

    # ILS Loop
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.995
    
    # Initialize counts
    current_counts = get_counts(current_blocks_set)
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to remove a redundant block
            # Find all redundant blocks
            redundant = []
            for bi in current_blocks_set:
                if is_redundant(bi, current_counts):
                    redundant.append(bi)
            
            if redundant:
                bi = random.choice(redundant)
                # Remove it
                current_blocks_set.discard(bi)
                # Update counts
                m = block_masks[bi]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    current_counts[idx] -= 1
                    m ^= lsb
                    
                if len(current_blocks_set) < best_count:
                    best_count = len(current_blocks_set)
                    best_solution = list(current_blocks_set)
            else:
                # No redundant block, try a swap? Or just cool down faster?
                T *= 0.99 # Cool down faster if stuck
                
        else:
            # Try a Swap: Remove one, Add one
            if not current_blocks_set:
                continue
                
            bi_old = random.choice(list(current_blocks_set))
            
            # We want to replace bi_old with a block that covers the triples 
            # that bi_old uniquely covers (if any) or just improves density.
            
            # Get triples uniquely covered by bi_old
            unique_mask = 0
            m = block_masks[bi_old]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if current_counts[idx] == 1:
                    unique_mask |= (1 << idx)
                m ^= lsb
            
            if unique_mask == 0:
                # bi_old is redundant, we should have removed it. 
                # But maybe we missed it? Let's just try to remove it.
                current_blocks_set.discard(bi_old)
                m = block_masks[bi_old]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    current_counts[idx] -= 1
                    m ^= lsb
                if len(current_blocks_set) < best_count:
                    best_count = len(current_blocks_set)
                    best_solution = list(current_blocks_set)
                continue
            
            # We need a new block bi_new that covers all of unique_mask.
            # Find candidates for bi_new from the intersection of blocks covering the unique triples.
            # Pick a random triple from unique_mask
            m = unique_mask
            lsb = m & -m
            t_idx = lsb.bit_length() - 1
            
            candidates = triple_to_blocks[t_idx]
            random.shuffle(candidates)
            
            found_new = False
            for bi_new in candidates:
                if bi_new == bi_old:
                    continue
                if bi_new in current_blocks_set:
                    continue
                    
                # Check if bi_new covers all unique_mask
                if (block_masks[bi_new] & unique_mask) == unique_mask:
                    # Check if the swap is valid (i.e., after removing bi_old and adding bi_new, 
                    # no triple is left uncovered)
                    # Since bi_new covers all unique triples of bi_old, and all other triples 
                    # were covered by >=2 blocks (or 1 block not bi_old), the coverage is maintained.
                    # Wait, if a triple was covered by bi_old and one other block, and we remove bi_old,
                    # it's still covered by the other.
                    # If a triple was covered ONLY by bi_old, it's in unique_mask, and bi_new covers it.
                    # So the swap is always valid if bi_new covers unique_mask.
                    
                    # Perform swap
                    current_blocks_set.discard(bi_old)
                    current_blocks_set.add(bi_new)
                    
                    # Update counts
                    m_old = block_masks[bi_old]
                    while m_old:
                        lsb = m_old & -m_old
                        idx = lsb.bit_length() - 1
                        current_counts[idx] -= 1
                        m_old ^= lsb
                        
                    m_new = block_masks[bi_new]
                    while m_new:
                        lsb = m_new & -m_new
                        idx = lsb.bit_length() - 1
                        current_counts[idx] += 1
                        m_new ^= lsb
                        
                    found_new = True
                    break
                    
            if not found_new:
                # No valid swap found, maybe try to add a block?
                # For now, just cool down
                pass

        T *= cooling_rate

    # Final Cleanup
    # Remove any remaining redundant blocks
    current_counts = get_counts(current_blocks_set)
    redundant = [bi for bi in current_blocks_set if is_redundant(bi, current_counts)]
    for bi in redundant:
        current_blocks_set.discard(bi)
        
    if len(current_blocks_set) < best_count:
        best_count = len(current_blocks_set)
        best_solution = list(current_blocks_set)

    # Verify final solution
    final_mask = 0
    for bi in best_solution:
        final_mask |= block_masks[bi]
        
    if final_mask != full_mask:
        # Fallback to greedy if SA broke it
        return [list(all_blocks[bi]) for bi in run_greedy_hard()]

    return [list(all_blocks[bi]) for bi in best_solution]

if __name__ == "__main__":
    # Test with small case
    start = time.time()
    blocks = construct(7, 4, 3)
    end = time.time()
    print(f"Time: {end-start:.2f}s, Blocks: {len(blocks)}")
    
    # Verify
    all_tsubs = set(itertools.combinations(range(7), 3))
    covered = set()
    for b in blocks:
        for tsub in itertools.combinations(b, 3):
            covered.add(tsub)
    if all_tsubs == covered:
        print("Valid")
    else:
        print("Invalid")
