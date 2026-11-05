import itertools
import time
import random

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

all_masks = [0] * N_BLOCKS
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for i, block in enumerate(all_blocks):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(i)
    all_masks[i] = mask

# Precompute gain tables for faster greedy selection
# For each block, store its mask as a list of set bits for faster iteration
block_bits = []
for i in range(N_BLOCKS):
    bits = []
    mm = all_masks[i]
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bits.append(bits)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    best_len = float('inf')
    best_blocks = []

    # Helper functions for managing a cover
    def create_cover_state():
        """Create a new cover state with counts and block set."""
        return {
            'blocks': [],
            'block_set': set(),
            'counts': [0] * N_SUBSETS
        }

    def add_block(state, idx):
        state['blocks'].append(idx)
        state['block_set'].add(idx)
        for bit_idx in block_bits[idx]:
            state['counts'][bit_idx] += 1

    def remove_block(state, idx):
        state['block_set'].remove(idx)
        # Remove from list efficiently
        pos = state['blocks'].index(idx)
        state['blocks'][pos] = state['blocks'][-1]
        state['blocks'].pop()
        for bit_idx in block_bits[idx]:
            state['counts'][bit_idx] -= 1

    def is_redundant(state, idx):
        for bit_idx in block_bits[idx]:
            if state['counts'][bit_idx] == 1:
                return False
        return True

    def prune_redundant(state):
        """Remove all redundant blocks. Returns number removed."""
        removed = 0
        changed = True
        while changed:
            changed = False
            for i in range(len(state['blocks']) - 1, -1, -1):
                idx = state['blocks'][i]
                if is_redundant(state, idx):
                    remove_block(state, idx)
                    removed += 1
                    changed = True
        return removed

    def greedy_fill(state, uncov_mask, time_limit):
        """Fill uncovered subsets using greedy."""
        while uncov_mask and time.time() < time_limit:
            best_idx = -1
            best_gain = 0
            for i in range(N_BLOCKS):
                if i in state['block_set']:
                    continue
                gain = (all_masks[i] & uncov_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            add_block(state, best_idx)
            uncov_mask &= ~all_masks[best_idx]
        return uncov_mask

    def try_swap(state, time_limit):
        """Try a 1-for-1 swap. Returns True if swap was made."""
        if not state['blocks']:
            return False
        
        # Try multiple random blocks
        for _ in range(min(30, len(state['blocks']))):
            if time.time() > time_limit:
                return False
            i = random.randint(0, len(state['blocks']) - 1)
            idx = state['blocks'][i]
            
            # Find unique subsets covered by idx
            unique_bits = []
            for bit_idx in block_bits[idx]:
                if state['counts'][bit_idx] == 1:
                    unique_bits.append(bit_idx)
            
            if not unique_bits:
                # It's redundant
                remove_block(state, idx)
                return True
            
            # Limit search if too many unique bits
            if len(unique_bits) <= 10:
                # Sort by constraint tightness (rarest first)
                unique_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                
                # Intersection of candidate lists
                candidates = set(blocks_containing_subset[unique_bits[0]])
                for b in unique_bits[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
                
                # Filter out blocks already in cover
                valid_candidates = [c for c in candidates if c not in state['block_set']]
                
                if valid_candidates:
                    # Choose the candidate that covers the most currently uncovered or weakly covered subsets
                    best_cand = valid_candidates[0]
                    best_gain = -1
                    for c in valid_candidates[:min(20, len(valid_candidates))]:
                        # Gain = number of subsets in c that have count <= 2 (would be improved)
                        gain = 0
                        for bit in block_bits[c]:
                            cnt = state['counts'][bit]
                            if cnt <= 2:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = c
                    remove_block(state, idx)
                    add_block(state, best_cand)
                    return True
        
        return False

    def try_swap_2_for_2(state, time_limit):
        """Try to replace 2 blocks with 2 different blocks (reduces redundancy)."""
        n_blocks = len(state['blocks'])
        if n_blocks < 2:
            return False
        
        max_attempts = 15
        for _ in range(max_attempts):
            if time.time() > time_limit:
                return False
            
            i, j = random.sample(range(n_blocks), 2)
            idx1 = state['blocks'][i]
            idx2 = state['blocks'][j]
            
            # Find subsets that would become uncovered if we remove both
            lost_bits = set()
            for s in block_bits[idx1]:
                c = state['counts'][s]
                if c == 1:
                    lost_bits.add(s)
                elif c == 2 and (all_masks[idx2] >> s) & 1:
                    lost_bits.add(s)
            for s in block_bits[idx2]:
                c = state['counts'][s]
                if c == 1:
                    lost_bits.add(s)
                elif c == 2 and (all_masks[idx1] >> s) & 1:
                    lost_bits.add(s)
            
            if not lost_bits:
                continue
            if len(lost_bits) > 12:
                continue
            
            cand_list = list(lost_bits)
            cand_list.sort(key=lambda b: len(blocks_containing_subset[b]))
            
            common_candidates = set(blocks_containing_subset[cand_list[0]])
            for s in cand_list[1:]:
                if not common_candidates:
                    break
                common_candidates &= set(blocks_containing_subset[s])
            
            # Need at least 2 candidates that can cover all lost bits
            valid_candidates = [c for c in common_candidates if c not in state['block_set']]
            if len(valid_candidates) < 2:
                continue
            
            # Try to find two candidates whose union covers all lost bits
            valid_candidates = valid_candidates[:min(30, len(valid_candidates))]
            found = False
            for a in range(len(valid_candidates)):
                for b in range(a+1, len(valid_candidates)):
                    ca, cb = valid_candidates[a], valid_candidates[b]
                    # Check if ca and cb together cover all lost bits
                    mask_union = all_masks[ca] | all_masks[cb]
                    ok = True
                    for s in lost_bits:
                        if not (mask_union >> s) & 1:
                            ok = False
                            break
                    if ok:
                        remove_block(state, idx1)
                        remove_block(state, idx2)
                        add_block(state, ca)
                        add_block(state, cb)
                        found = True
                        break
                if found:
                    break
            if found:
                return True
        
        return False

    def try_swap_2_for_1(state, time_limit):
        """Try to replace 2 blocks with 1 block."""
        n_blocks = len(state['blocks'])
        if n_blocks < 2:
            return False
        
        # Sample a few pairs
        max_attempts = 25
        for _ in range(max_attempts):
            if time.time() > time_limit:
                return False
            
            i, j = random.sample(range(n_blocks), 2)
            idx1 = state['blocks'][i]
            idx2 = state['blocks'][j]
            
            # Find subsets that would become uncovered if we remove idx1 and idx2
            candidates_unique = set()
            for s in block_bits[idx1]:
                c = state['counts'][s]
                if c == 1:
                    candidates_unique.add(s)
                elif c == 2 and (all_masks[idx2] >> s) & 1:
                    candidates_unique.add(s)
            
            for s in block_bits[idx2]:
                c = state['counts'][s]
                if c == 1:
                    candidates_unique.add(s)
                elif c == 2 and (all_masks[idx1] >> s) & 1:
                    candidates_unique.add(s)
            
            if not candidates_unique:
                continue
                
            if len(candidates_unique) > 10:
                continue
                
            cand_list = list(candidates_unique)
            cand_list.sort(key=lambda b: len(blocks_containing_subset[b]))
            
            common_candidates = set(blocks_containing_subset[cand_list[0]])
            for s in cand_list[1:]:
                if not common_candidates:
                    break
                common_candidates &= set(blocks_containing_subset[s])
                
            valid_candidates = [c for c in common_candidates if c not in state['block_set']]
            
            if valid_candidates:
                new_idx = random.choice(valid_candidates)
                remove_block(state, idx1)
                remove_block(state, idx2)
                add_block(state, new_idx)
                return True
                
        return False

    # Strategy 1: Probabilistic Method with Alteration
    # Select random blocks with probability p, then remove redundant ones
    # Expected size of cover: C(19,7) * p * (1 - (1-1/C(19,4))^p) approx
    # We want to find good p values to try
    
    # Theoretical: each 4-subset is in C(19-4, 7-4) = C(15,3) = 455 blocks
    # out of C(19,7) = 50388 total blocks
    # P(not covered by one random block) = 1 - 455/50388 ≈ 1 - 0.00903
    # P(not covered by p fraction) = (1 - 0.00903)^p
    # We need this to be very small, so p should be around ln(50388*455)/455 ≈ ... 
    # Let's just try various p values
    
    p_values = [0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025]
    
    for p in p_values:
        if time.time() > deadline - 30:
            break
        
        state = create_cover_state()
        
        # Randomly select blocks with probability p
        for i in range(N_BLOCKS):
            if random.random() < p:
                add_block(state, i)
        
        # Check if we have a valid cover; if not, fill in greedily
        uncov_mask = 0
        for s in range(N_SUBSETS):
            if state['counts'][s] == 0:
                uncov_mask |= 1 << s
        
        if uncov_mask:
            greedy_fill(state, uncov_mask, deadline - 15)
        
        # Now prune redundant blocks (alteration step)
        prune_redundant(state)
        
        # Try some local search improvements
        local_search_time = min(deadline - 10, start_time + 115)
        no_improve = 0
        max_no_improve = 30
        
        while time.time() < local_search_time:
            # Prune
            if prune_redundant(state) > 0:
                no_improve = 0
                continue
            
            # Try swap
            if try_swap(state, local_search_time):
                no_improve = 0
                continue
            
            # Try 2-for-1 swap
            if try_swap_2_for_1(state, local_search_time):
                no_improve = 0
                continue
            
            # Try 2-for-2 swap
            if try_swap_2_for_2(state, local_search_time):
                no_improve = 0
                continue
            
            no_improve += 1
            if no_improve > max_no_improve:
                # Perturbation: remove some random blocks and re-greedy
                n_remove = random.randint(2, 5)
                to_remove = random.sample(state['blocks'], min(n_remove, len(state['blocks'])))
                for idx in to_remove:
                    remove_block(state, idx)
                
                # Compute uncovered
                uncov_mask = 0
                for s in range(N_SUBSETS):
                    if state['counts'][s] == 0:
                        uncov_mask |= 1 << s
                
                greedy_fill(state, uncov_mask, local_search_time)
                prune_redundant(state)
                
                no_improve = 0
                max_no_improve = min(100, max_no_improve * 2)
        
        # Update best
        if len(state['blocks']) < best_len:
            best_len = len(state['blocks'])
            best_blocks = state['blocks'].copy()
        
        # Early exit if we're doing well
        if best_len <= 145:
            break

    # Strategy 2: Greedy initialization with local search (backup)
    if time.time() < deadline - 20:
        state = create_cover_state()
        uncov_mask = (1 << N_SUBSETS) - 1
        greedy_fill(state, uncov_mask, deadline - 10)
        prune_redundant(state)
        
        # Local search
        no_improve = 0
        max_no_improve = 50
        while time.time() < deadline - 5:
            if prune_redundant(state) > 0:
                no_improve = 0
                continue
            
            if try_swap(state, deadline - 5):
                no_improve = 0
                continue
            
            if try_swap_2_for_1(state, deadline - 5):
                no_improve = 0
                continue
            
            if try_swap_2_for_2(state, deadline - 5):
                no_improve = 0
                continue
            
            no_improve += 1
            if no_improve > max_no_improve:
                n_remove = random.randint(3, 6)
                to_remove = random.sample(state['blocks'], min(n_remove, len(state['blocks'])))
                for idx in to_remove:
                    remove_block(state, idx)
                
                uncov_mask = 0
                for s in range(N_SUBSETS):
                    if state['counts'][s] == 0:
                        uncov_mask |= 1 << s
                
                greedy_fill(state, uncov_mask, deadline - 5)
                prune_redundant(state)
                
                no_improve = 0
                max_no_improve = min(200, max_no_improve * 2)
        
        if len(state['blocks']) < best_len:
            best_len = len(state['blocks'])
            best_blocks = state['blocks'].copy()

    # Final cleanup
    if best_blocks:
        state = create_cover_state()
        for idx in best_blocks:
            add_block(state, idx)
        prune_redundant(state)
        if len(state['blocks']) < best_len:
            best_len = len(state['blocks'])
            best_blocks = state['blocks'].copy()

    # Final polish: dedicated local search on best solution with more iterations
    if best_blocks and time.time() < deadline - 5:
        state = create_cover_state()
        for idx in best_blocks:
            add_block(state, idx)
        prune_redundant(state)
        
        polish_deadline = deadline - 3
        no_improve = 0
        max_no_improve = 40
        while time.time() < polish_deadline:
            if prune_redundant(state) > 0:
                no_improve = 0
                continue
            if try_swap(state, polish_deadline):
                no_improve = 0
                continue
            if try_swap_2_for_1(state, polish_deadline):
                no_improve = 0
                continue
            if try_swap_2_for_2(state, polish_deadline):
                no_improve = 0
                continue
            no_improve += 1
            if no_improve > max_no_improve:
                # Smaller perturbation for polish
                n_remove = random.randint(1, 3)
                to_remove = random.sample(state['blocks'], min(n_remove, len(state['blocks'])))
                for idx in to_remove:
                    remove_block(state, idx)
                uncov_mask = 0
                for s in range(N_SUBSETS):
                    if state['counts'][s] == 0:
                        uncov_mask |= 1 << s
                greedy_fill(state, uncov_mask, polish_deadline)
                prune_redundant(state)
                no_improve = 0
                max_no_improve = min(150, max_no_improve * 2)
        
        if len(state['blocks']) < best_len:
            best_len = len(state['blocks'])
            best_blocks = state['blocks'].copy()

    return [list(all_blocks[i]) for i in best_blocks]
