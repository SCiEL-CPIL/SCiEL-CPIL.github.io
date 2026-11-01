import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    if n <= 1:
        return "1" if n == 1 else ""

    # Build the standard recursive superpermutation
    # S_1 = [1]
    # S_m from S_{m-1}:
    #   Let W = list of distinct (m-1)-permutations in order of appearance in S_{m-1}
    #   S_m = w_1, m, w_1, m, w_2, m, w_2, ..., m, w_k, m, w_k
    #   where consecutive blocks overlap where possible (they share the trailing symbols)
    #   
    # Actually the correct construction is:
    # S_m = w_1, m, w_2, m, w_3, m, ..., w_k, m, w_k
    # where w_i are the distinct (m-1)-perms in S_{m-1}, and the last w_k is repeated.
    # Length = sum_{i=1}^{m} i! 
    # S_1: [1], len 1
    # S_2: [1,2,1], len 3 = 1+2
    # S_3: [1,2,3,1,2,1,3,2,1], len 9 = 1+2+6
    # S_4: len 33 = 1+2+6+24
    # ...
    # S_7: len 1+2+6+24+120+720+5040 = 5913

    def build_superperm(n):
        if n == 1:
            return [1]
        prev = build_superperm(n - 1)
        # Get distinct (n-1)-length windows in order of first appearance
        windows = []
        seen = set()
        for i in range(len(prev) - (n - 1) + 1):
            w = tuple(prev[i:i + n - 1])
            if w not in seen:
                seen.add(w)
                windows.append(w)
        
        # Build new word: w_1, n, w_2, n, w_3, ..., n, w_k, n, w_k
        # The key insight: consecutive windows in prev overlap by n-2 symbols
        # when they are consecutive in the superpermutation.
        # So we can just append: w_1, then for each next window w_i, 
        # append n and the part of w_i that doesn't overlap with current end.
        
        result = []
        for i, w in enumerate(windows):
            if i == 0:
                result.extend(w)
            else:
                # Append n
                result.append(n)
                # Append w with maximum overlap with current end
                # The current end is the previous window (which ends with w[i-1])
                # w[i] shares n-2 symbols with w[i-1] at the end of w[i-1] / start of w[i]
                # Actually, let's just find max overlap
                max_ov = 0
                for k in range(min(len(result), len(w)), 0, -1):
                    if result[-k:] == list(w[:k]):
                        max_ov = k
                        break
                result.extend(w[max_ov:])
            # After each window except the last, we don't append n yet
            # Wait, the pattern is w_1, n, w_2, n, ..., n, w_k, n, w_k
            # So after the last window, we append n and w_k again
        
        # Append n and last window at the end
        result.append(n)
        # Append last window with overlap
        w_last = windows[-1]
        max_ov = 0
        for k in range(min(len(result), len(w_last)), 0, -1):
            if result[-k:] == list(w_last[:k]):
                max_ov = k
                break
        result.extend(w_last[max_ov:])
        
        return result

    word = build_superperm(n)
    L = len(word)
    
    # For n <= 4, the recursive construction is already optimal
    # n=1: 1, n=2: 3, n=3: 9, n=4: 33
    if n <= 4:
        return "".join(map(str, word))

    # For n >= 5, apply local search deletion
    # Convert to list of integers for efficiency
    
    # Map all permutations to IDs
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    total_perms = len(all_perms)  # n!
    
    def compute_window_ids(w, L, n):
        """Compute permutation ID for each window of length n"""
        ids = [0] * (L - n + 1)
        for i in range(L - n + 1):
            ids[i] = perm_to_id[tuple(w[i:i+n])]
        return ids
    
    def count_windows(ids):
        """Count occurrences of each permutation"""
        counts = [0] * total_perms
        for pid in ids:
            counts[pid] += 1
        return counts
    
    def try_delete(word, idx, n, window_ids, counts, L):
        """Try to delete position idx. Returns True if deletion is valid."""
        # Affected old windows: those that include position idx
        # Window j covers positions [j, j+n-1]
        # Position idx is in window j iff j <= idx <= j+n-1
        # So j ranges from max(0, idx-n+1) to min(idx, L-n)
        
        start_w = max(0, idx - n + 1)
        end_w = min(idx, L - n)
        
        if start_w > end_w:
            return False
        
        # Save old counts and decrement
        old_pids = {}
        for j in range(start_w, end_w + 1):
            pid = window_ids[j]
            old_pids[pid] = old_pids.get(pid, 0) + 1
            counts[pid] -= 1
        
        # After deletion, new windows form at positions [start_w, idx-1]
        # (the last old window at position idx is gone, and windows shift)
        # New window at position j (for j in [start_w, idx-1]) consists of:
        #   word[j:idx] + word[idx+1:idx+1+(n-(idx-j))]
        # But only if idx+1+(n-(idx-j)) <= L, i.e., n-(idx-j) <= L-idx-1, i.e., j >= idx+n-L-1... 
        # Actually the new word has length L-1, so new window at j needs j+n-1 <= L-2, i.e., j <= L-n-1
        
        new_pids = {}
        valid = True
        for j in range(start_w, min(idx, L - n)):
            # New window at position j in the new word
            # It spans: old word[j:idx] + old word[idx+1:idx+1+(n-(idx-j))]
            prefix_len = idx - j
            suffix_len = n - prefix_len
            new_start = idx + 1
            new_end = new_start + suffix_len
            if new_end > L:
                # This window would extend beyond the old word
                # Actually after deletion, the word is shorter
                # The new window at j in new word corresponds to:
                # new_word[j:j+n] = old_word[j:idx] + old_word[idx+1:idx+1+suffix_len]
                # We need old_word[idx+1:idx+1+suffix_len] to be valid
                # idx+1+suffix_len = idx+1+n-(idx-j) = n+1+j
                # Need n+1+j <= L => j <= L-n-1
                # We already have j < L-n from the range, so j <= L-n-1
                # So this is fine
                pass
            
            # Build the new permutation
            # Use a precomputed lookup or build tuple
            new_p = tuple(word[j:idx]) + tuple(word[idx+1:idx+1+suffix_len])
            
            # Check it's a valid permutation
            if new_p not in perm_to_id:
                valid = False
                break
            new_pid = perm_to_id[new_p]
            new_pids[new_pid] = new_pids.get(new_pid, 0) + 1
            counts[new_pid] += 1
        
        if not valid:
            # Revert counts
            for pid, cnt in old_pids.items():
                counts[pid] += cnt
            for pid, cnt in new_pids.items():
                counts[pid] -= cnt
            return False
        
        # Check: for every permutation that had count 1 in old windows and is affected,
        # it must still have count >= 1 in new windows
        for pid, cnt in old_pids.items():
            if cnt == 1:
                # This permutation only appeared in affected old windows
                # Check if it appears in new windows or in unaffected old windows
                # Its total old count was counts[pid] + cnt (before we decremented)
                # Actually, we already decremented, so counts[pid] is the count after removing affected windows
                # We need counts[pid] + new_pids.get(pid, 0) > 0
                # i.e., counts[pid] + new_pids.get(pid, 0) >= 1
                if counts[pid] + new_pids.get(pid, 0) < 1:
                    # This permutation is lost!
                    # Revert
                    for pid2, cnt2 in old_pids.items():
                        counts[pid2] += cnt2
                    for pid2, cnt2 in new_pids.items():
                        counts[pid2] -= cnt2
                    return False
        
        # Valid deletion! Commit
        word.pop(idx)
        # Update window_ids
        # Old windows [start_w, end_w] are removed/changed
        # New windows [start_w, idx-1] replace them
        # Windows after idx shift left by 1
        
        # Rebuild window_ids for affected region
        # New length is L-1, so there are (L-1)-n+1 = L-n windows
        # Old had L-n+1 windows
        # Windows at positions < start_w are unchanged
        # Windows at positions >= idx in old word become positions >= idx-1 in new word
        # But we need to rebuild the affected region
        
        # Simplest: rebuild all window_ids (expensive but safe)
        # Or rebuild just the affected portion
        # For now, let's rebuild for correctness, and optimize later if needed
        
        return True

    # Better approach: rebuild window_ids incrementally
    def commit_deletion(word, idx, n, window_ids, L):
        """After deleting word[idx], update window_ids. L is old length."""
        new_L = L - 1
        # Old window_ids has L-n+1 entries
        # New window_ids will have new_L-n+1 = L-n entries
        # Windows at positions [0, idx-n] are unchanged (if valid)
        # Windows at positions [max(0,idx-n+1), idx-1] are new
        # Windows at positions [idx, L-n-1] in new word correspond to [idx+1, L-n] in old word
        
        # Rebuild the whole thing for simplicity first
        new_window_ids = compute_window_ids(word, new_L, n)
        return new_window_ids

    # Initialize
    window_ids = compute_window_ids(word, L, n)
    counts = count_windows(window_ids)
    
    # Verify initial word is valid
    assert min(counts) >= 1, "Initial superpermutation is invalid!"
    
    # Local search: try deletions
    # Strategy: scan positions, try to delete each one
    # If successful, delete and restart scan (or continue from same position)
    # Repeat until no more deletions are possible or time runs out
    
    # Use a randomized approach: try deletions in random order, 
    # and use multiple passes
    
    best_length = L
    
    # Fast deletion check using precomputed structures
    # For each position, we need to quickly determine if deletion is valid
    
    # Precompute for each position the set of windows that contain it
    # This is expensive, so let's just do the check on the fly
    
    # Optimized deletion:
    def can_delete(word, idx, n, window_ids, counts, L):
        """Check if deleting position idx is valid. Returns (valid, old_pids, new_pids)"""
        start_w = max(0, idx - n + 1)
        end_w = min(idx, L - n)
        
        if start_w > end_w:
            return False, {}, {}
        
        # Count old affected windows
        old_pids = {}
        for j in range(start_w, end_w + 1):
            pid = window_ids[j]
            old_pids[pid] = old_pids.get(pid, 0) + 1
        
        # Build new windows
        new_pids = {}
        for j in range(start_w, min(idx, L - n)):
            prefix_len = idx - j
            suffix_len = n - prefix_len
            # new_p = word[j:idx] + word[idx+1:idx+1+suffix_len]
            # Build tuple efficiently
            new_p = tuple(word[j:idx]) + tuple(word[idx+1:idx+1+suffix_len])
            if new_p not in perm_to_id:
                return False, old_pids, {}
            new_pid = perm_to_id[new_p]
            new_pids[new_pid] = new_pids.get(new_pid, 0) + 1
        
        # Check validity: no permutation loses all its occurrences
        for pid, cnt in old_pids.items():
            # Total old count = current count + cnt (we haven't decremented yet)
            # After deletion, count = current_count - cnt + new_pids.get(pid, 0)
            # Need this >= 1
            if counts[pid] - cnt + new_pids.get(pid, 0) < 1:
                return False, old_pids, new_pids
        
        return True, old_pids, new_pids
    
    def perform_deletion(word, idx, n, window_ids, counts, L):
        """Perform deletion at idx. Updates word, window_ids, counts in place. Returns new L."""
        old_pids = {}
        start_w = max(0, idx - n + 1)
        end_w = min(idx, L - n)
        
        # Decrement old counts
        for j in range(start_w, end_w + 1):
            pid = window_ids[j]
            old_pids[pid] = old_pids.get(pid, 0) + 1
            counts[pid] -= 1
        
        # Increment new counts
        for j in range(start_w, min(idx, L - n)):
            prefix_len = idx - j
            suffix_len = n - prefix_len
            new_p = tuple(word[j:idx]) + tuple(word[idx+1:idx+1+suffix_len])
            new_pid = perm_to_id[new_p]
            counts[new_pid] += 1
        
        # Delete from word
        word.pop(idx)
        new_L = L - 1
        
        # Rebuild window_ids for affected region
        # For simplicity, rebuild all
        window_ids[:] = compute_window_ids(word, new_L, n)
        
        return new_L

    # Main search loop
    # Phase 1: Greedy scan from left to right
    # Phase 2: Randomized restarts
    # Phase 3: Multi-pass with different random seeds
    
    current_L = L
    
    # Phase 1: Left-to-right scan
    while time.time() - start_time < deadline:
        if current_L <= best_length:
            best_length = current_L
        
        made_progress = False
        for idx in range(current_L):
            if time.time() - start_time > deadline:
                break
            valid, old_pids, new_pids = can_delete(word, idx, n, window_ids, counts, current_L)
            if valid:
                current_L = perform_deletion(word, idx, n, window_ids, counts, current_L)
                made_progress = True
                # After deletion, indices shift; restart scan from 0
                break
        
        if not made_progress:
            # No more deletions in left-to-right scan
            # Try randomized approach
            break
    
    # Phase 2: Randomized deletion attempts
    # Try random positions, and if stuck, try different strategies
    iterations = 0
    while time.time() - start_time < deadline - 5:
        iterations += 1
        made_progress = False
        
        # Try a batch of random positions
        batch_size = min(1000, current_L)
        positions = random.sample(range(current_L), batch_size)
        
        for idx in positions:
            if time.time() - start_time > deadline - 5:
                break
            valid, old_pids, new_pids = can_delete(word, idx, n, window_ids, counts, current_L)
            if valid:
                current_L = perform_deletion(word, idx, n, window_ids, counts, current_L)
                made_progress = True
                # After deletion, some positions in our batch may be invalid
                # Just continue with the remaining positions (they may be out of range)
                # Actually, after deletion, indices >= idx shift left
                # So positions in our list that are >= idx need adjustment
                # For simplicity, break and restart
                break
        
        if made_progress:
            if current_L < best_length:
                best_length = current_L
        else:
            # No progress in this batch, try a different strategy
            # Do a full scan in reverse
            for idx in range(current_L - 1, -1, -1):
                if time.time() - start_time > deadline - 5:
                    break
                valid, old_pids, new_pids = can_delete(word, idx, n, window_ids, counts, current_L)
                if valid:
                    current_L = perform_deletion(word, idx, n, window_ids, counts, current_L)
                    made_progress = True
                    break
            
            if not made_progress:
                # Try bidirectional: scan from random middle
                mid = random.randint(0, current_L)
                for offset in range(max(mid, current_L - mid)):
                    for idx in [mid + offset, mid - offset] if mid - offset >= 0 else [mid + offset]:
                        if time.time() - start_time > deadline - 5:
                            break
                        if 0 <= idx < current_L:
                            valid, old_pids, new_pids = can_delete(word, idx, n, window_ids, counts, current_L)
                            if valid:
                                current_L = perform_deletion(word, idx, n, window_ids, counts, current_L)
                                made_progress = True
                                break
                    if made_progress:
                        break
                
                if not made_progress:
                    # Stuck. Try a more aggressive approach: 
                    # temporarily allow some "bad" deletions and repair?
                    # For now, just break
                    break
    
    # Final: try a few more random passes
    for _ in range(10):
        if time.time() - start_time > deadline:
            break
        made_progress = False
        for idx in random.sample(range(current_L), min(500, current_L)):
            if time.time() - start_time > deadline:
                break
            valid, old_pids, new_pids = can_delete(word, idx, n, window_ids, counts, current_L)
            if valid:
                current_L = perform_deletion(word, idx, n, window_ids, counts, current_L)
                made_progress = True
        if not made_progress:
            break
    
    return "".join(map(str, word))
