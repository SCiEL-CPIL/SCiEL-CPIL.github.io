"""Different design family: pairwise-greedy set cover over all k-subsets,
with NumPy-vectorized marginal-gain scoring and a hard deadline + repair."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin
    rng = random.Random(98765)

    # ---------- Precomputation ----------
    # 1. Enumerate t-subsets and k-subsets
    tsub_list = list(itertools.combinations(range(v), t))
    ksub_list = list(itertools.combinations(range(v), k))
    
    n_tsub = len(tsub_list)
    n_ksub = len(ksub_list)
    
    # Map t-subset tuple to index
    idx_of_tsub = {sub: i for i, sub in enumerate(tsub_list)}
    
    # 2. For each k-subset, find which t-subsets it covers
    # cover_indices[j] is a list of t-subset indices covered by ksub j
    cover_indices = []
    for ksub in ksub_list:
        kset = set(ksub)
        covered_tsubs = []
        for tsub in tsub_list:
            if set(tsub).issubset(kset):
                covered_tsubs.append(idx_of_tsub[tsub])
        cover_indices.append(covered_tsubs)
        
    # 3. For each t-subset, find which k-subsets cover it
    # tsub_to_blocks[i] is a list of k-subset indices that cover tsub i
    tsub_to_blocks = [[] for _ in range(n_tsub)]
    for j in range(n_ksub):
        for s in cover_indices[j]:
            tsub_to_blocks[s].append(j)
            
    # 4. State
    gains = np.full(n_ksub, len(cover_indices[0]), dtype=np.int32) # Initially all cover 35
    is_covered = np.zeros(n_tsub, dtype=bool)
    chosen_set = set()
    chosen_blocks = []
    
    # ---------- Greedy Loop ----------
    while True:
        if time.time() > deadline:
            break
            
        # Find best block: max gain, not chosen
        valid_mask = np.ones(n_ksub, dtype=bool)
        for j in chosen_set:
            valid_mask[j] = False
            
        # Get gains for valid blocks
        valid_gains = np.where(valid_mask, gains, -1)
        max_gain = int(np.max(valid_gains))
        
        if max_gain <= 0:
            break
        
        # Find candidates with max_gain
        candidates = np.where((valid_gains == max_gain) & valid_mask)[0]
        
        if len(candidates) == 0:
            break
            
        # Tie-breaking: Randomly pick from candidates
        best_j = int(rng.choice(candidates))
        
        # Commit
        chosen_set.add(best_j)
        chosen_blocks.append(ksub_list[best_j])
        gains[best_j] = -1 # Mark as used
        
        # Update gains for blocks sharing t-subsets with best_j
        for s in cover_indices[best_j]:
            if not is_covered[s]:
                is_covered[s] = True
                # Decrement gain for all blocks covering s
                for j in tsub_to_blocks[s]:
                    if j not in chosen_set:
                        gains[j] -= 1
                        
        # Check if done
        if np.all(is_covered):
            break

    # ---------- Repair ----------
    # If not all covered, add blocks for remaining uncovered t-subsets
    uncovered_tsubs = np.where(~is_covered)[0]
    chosen_set_tuples = set(tuple(sorted(b)) for b in chosen_blocks)
    
    for s in uncovered_tsubs:
        if is_covered[s]:
            continue # Already covered by a previous repair block
            
        # Pick a block covering s
        candidates = tsub_to_blocks[s]
        chosen_candidate = None
        for j in candidates:
            b_tuple = tuple(sorted(ksub_list[j]))
            if b_tuple not in chosen_set_tuples:
                chosen_candidate = ksub_list[j]
                break
        if chosen_candidate is None:
            # All blocks covering s are already in the set?
            # This means s is covered. Contradiction.
            # Fallback: pick any
            chosen_candidate = ksub_list[candidates[0]]
            b_tuple = tuple(sorted(chosen_candidate))
            
        chosen_blocks.append(chosen_candidate)
        chosen_set_tuples.add(b_tuple)
        
        # Update is_covered for all tsubs in this block
        bset = set(chosen_candidate)
        for t_idx in range(n_tsub):
            if not is_covered[t_idx]:
                if set(tsub_list[t_idx]).issubset(bset):
                    is_covered[t_idx] = True
                    
    # Final deduplication
    final_blocks = list(set(tuple(sorted(b)) for b in chosen_blocks))
    result = [list(b) for b in final_blocks]
    return result
# EVOLVE-BLOCK-END
