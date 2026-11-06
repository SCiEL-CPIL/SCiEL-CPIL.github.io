import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Leave margin for 60s limit
    
    # Precompute masks for distances 0 to 5
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)
    
    all_masks = []
    for d in range(MIN_DIST):
        all_masks.extend(masks_by_dist[d])
        
    # blocked_count[i] is the number of codewords within distance < MIN_DIST of i
    blocked_count = np.zeros(1 << N, dtype=np.int32)
    
    code = []
    
    def add_word(w):
        code.append(w)
        for mask in all_masks:
            blocked_count[w ^ mask] += 1
            
    def remove_word(w):
        code.remove(w)
        for mask in all_masks:
            blocked_count[w ^ mask] -= 1

    # Phase 1: Greedy Construction
    # This typically finds the 256-word linear code if we iterate in order
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if blocked_count[w] == 0:
            add_word(w)
            
    best_code = list(code)
    best_size = len(code)
    
    random.seed(42)
    
    # Phase 2: Local Search
    # We try to increase the size.
    # Moves:
    # 1. Add a word if possible (blocked_count[w] == 0)
    # 2. Swap 1 word for 2 words (remove c, add u, v such that d(u,v)>=6 and compatible with rest)
    
    no_improvement_count = 0
    max_no_improvement = 100 # Reset perturbation strategy after this many failed improvement attempts
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Check if we can just add a word
        # We can scan the whole space or sample. Scanning 131k is fast.
        # But if we are in a local optimum, no word is free.
        # Let's check if any word is free.
        # Instead of scanning all, we can keep a list of "candidates" or just scan periodically.
        # For now, a full scan is ~0.01s, so it's fine.
        
        # Optimization: Only scan if we haven't checked recently or if code changed.
        # Since we restart on improvement, we can scan every iteration.
        
        free_word_found = -1
        # We can vectorize the check: np.where(blocked_count == 0)[0]
        # But we need to exclude words already in the code?
        # blocked_count[w] for w in code is at least 1 (from itself, dist 0).
        # So blocked_count[w] == 0 implies w is NOT in code and is compatible.
        
        free_indices = np.where(blocked_count == 0)[0]
        
        if len(free_indices) > 0:
            # Pick a random free word to add
            w = int(random.choice(free_indices))
            add_word(w)
            improved = True
            if len(code) > best_size:
                best_size = len(code)
                best_code = list(code)
            # Continue loop to check for more additions
            continue
        
        # No free word found. Try Swap 1 for 2.
        # Shuffle codewords to avoid bias
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        swap_success = False
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            c = code[idx]
            
            # Temporarily remove c
            for mask in all_masks:
                blocked_count[c ^ mask] -= 1
            
            # Find newly free words
            # These are words w such that blocked_count[w] == 0 now,
            # but were > 0 before (blocked by c).
            # Actually, any word with blocked_count == 0 is a candidate.
            # We need to find TWO compatible words among them.
            
            free_indices = np.where(blocked_count == 0)[0]
            
            if len(free_indices) >= 2:
                # Try to find a pair (u, v) in free_indices with d(u,v) >= MIN_DIST
                # free_indices can be large.
                # We can sample or iterate.
                # Let's iterate through free_indices and check against others.
                
                # To speed up, we can pick a candidate u, then check all v > u.
                # If free_indices is very large, this might be slow.
                # But typically, removing one word from a tight code unblocks a small number of words.
                
                found_pair = False
                # Shuffle free indices to randomize
                free_list = list(free_indices)
                random.shuffle(free_list)
                
                for i in range(len(free_list)):
                    u = free_list[i]
                    # Check compatibility with other free words
                    # We only need to find ONE v.
                    for j in range(i + 1, len(free_list)):
                        v = free_list[j]
                        if np.bit_count(u ^ v) >= MIN_DIST:
                            # Found pair u, v
                            # Remove c
                            code.remove(c)
                            
                            # Add u
                            code.append(u)
                            for mask in all_masks:
                                blocked_count[u ^ mask] += 1
                                
                            # Add v
                            code.append(v)
                            for mask in all_masks:
                                blocked_count[v ^ mask] += 1
                                
                            improved = True
                            if len(code) > best_size:
                                best_size = len(code)
                                best_code = list(code)
                                
                            swap_success = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
            if swap_success:
                break
            else:
                # Restore c
                for mask in all_masks:
                    blocked_count[c ^ mask] += 1
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturbation: Swap 1 for 1 (Move)
                # Pick a random codeword c and a random free word w (after removing c)
                # such that d(c,w) >= MIN_DIST is not required for the swap itself,
                # but w must be compatible with the rest.
                
                c_idx = random.randint(0, len(code) - 1)
                c = code[c_idx]
                
                # Remove c
                for mask in all_masks:
                    blocked_count[c ^ mask] -= 1
                    
                # Find a free word w
                free_indices = np.where(blocked_count == 0)[0]
                
                if len(free_indices) > 0:
                    w = int(random.choice(free_indices))
                    # Add w
                    code.append(w)
                    for mask in all_masks:
                        blocked_count[w ^ mask] += 1
                    # Note: c is already removed from blocked_count, 
                    # but we need to remove it from the code list.
                    code.remove(c)
                    
                    # Reset no_improvement counter
                    no_improvement_count = 0
                else:
                    # No free word, restore c
                    for mask in all_masks:
                        blocked_count[c ^ mask] += 1
                    # If stuck, break to save time
                    break

    return best_code
