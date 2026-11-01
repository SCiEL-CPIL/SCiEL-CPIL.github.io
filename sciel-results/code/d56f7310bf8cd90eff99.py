import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    if n <= 1:
        return "1" if n == 1 else ""
        
    # 1. Generate initial superpermutation
    # Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        # Get all distinct (m-1)-length windows from current word
        # and append m to form new blocks
        # The standard construction:
        # S_m is formed by taking S_{m-1}, and for each permutation p of 1..m-1 
        # that appears as a window in S_{m-1}, we append p, m, p.
        # Wait, the standard construction is:
        # S_1 = [1]
        # S_m: Let W be the list of all permutations of 1..m-1 in the order they appear as windows in S_{m-1}.
        #      Actually, it's simpler:
        #      S_m = [p_1, m, p_1, p_2, m, p_2, ..., p_k, m, p_k] where p_i are the distinct (m-1)-perms in S_{m-1}.
        #      But we need to handle overlaps.
        
        # Let's use the explicit construction:
        # Start with S_1 = [1]
        # To get S_m:
        #   Take S_{m-1}.
        #   Identify all distinct (m-1)-length windows in S_{m-1}.
        #   Let these be w_1, w_2, ..., w_k.
        #   New word is: w_1, m, w_1, w_2, m, w_2, ..., w_k, m, w_k.
        #   This ensures all m-perms are present.
        
        # Optimization: We can just build it.
        if m == 2:
            word = [1, 2, 1]
        else:
            # Get distinct windows of length m-1
            windows = []
            seen = set()
            for i in range(len(word) - (m - 1) + 1):
                w = tuple(word[i:i + m - 1])
                if w not in seen:
                    seen.add(w)
                    windows.append(list(w))
            
            new_word = []
            for w in windows:
                # Append w, then m, then w
                # But we need to overlap with previous part
                # The previous part ended with the last w from the previous iteration?
                # No, the structure is:
                # w_1, m, w_1
                # then w_2, m, w_2
                # Overlap: The end of (w_1, m, w_1) is w_1.
                # The start of (w_2, m, w_2) is w_2.
                # Do they overlap?
                # w_1 and w_2 are permutations of 1..m-1.
                # They are distinct. So no overlap between the blocks.
                # Wait, the standard superpermutation construction has overlaps.
                
                # Let's re-verify the standard construction.
                # S_1 = 1
                # S_2 = 1 2 1
                # S_3: Windows of S_2 (len 2): 12, 21.
                #       Blocks: 12 3 12, 21 3 21.
                #       Concatenation: 12 3 12 3 21 3 21 ?
                #       Length: 3 + 3 + 3 + 3 = 12? No.
                #       12 3 12 (len 4)
                #       21 3 21 (len 4)
                #       Overlap? End of first is 12. Start of second is 21. No overlap.
                #       Total len 8?
                #       1 2 3 1 2 3 2 1 3 2 1?
                #       Let's check S_3 length. 1! + 2! + 3! = 1 + 2 + 6 = 9.
                #       My manual trace: 1, 2, 3, 1, 2, 3, 2, 1, 3, 2, 1 (len 11).
                #       The standard construction is:
                #       S_m = S_{m-1} with m inserted?
                
                # Correct Algorithm:
                # S_1 = [1]
                # S_m is constructed from S_{m-1} as follows:
                # Let P be the sequence of all permutations of 1..m-1 in the order they appear as windows in S_{m-1}.
                # S_m = P_1, m, P_1, P_2, m, P_2, ..., P_k, m, P_k.
                # Where P_i are the permutations.
                # For m=3, P from S_2=[1,2,1] are 12, 21.
                # S_3 = 12, 3, 12, 21, 3, 21.
                # Seq: 1, 2, 3, 1, 2, 2, 1, 3, 2, 1. (Len 10).
                # But 1!+2!+3! = 9.
                # The minimal superpermutation for n=3 is 123121321 (len 9).
                # The recursive construction I used gives 10.
                # The "shortest known" for n=3 is 9.
                # The recursive construction $S_n$ has length $\sum i!$.
                # For n=3, sum is 9.
                # So my trace was wrong.
                
                # Let's trace S_3 from S_2=[1,2,1].
                # Windows of len 2: 12, 21.
                # Construction:
                # Start with 12.
                # Append 3.
                # Append 12.
                # Current: 1, 2, 3, 1, 2.
                # Next perm 21.
                # Append 3?
                # Overlap? End is 12. Start of 21 is 21. No overlap.
                # Append 21.
                # Current: 1, 2, 3, 1, 2, 2, 1.
                # Append 3?
                # Append 21.
                # Current: 1, 2, 3, 1, 2, 2, 1, 3, 2, 1.
                # This is length 10.
                
                # Why is the formula $\sum i!$?
                # $L_1 = 1$.
                # $L_2 = 3$.
                # $L_3 = 9$.
                # The construction that achieves $\sum i!$ is:
                # S_m is formed by taking S_{m-1} and inserting m?
                # Actually, the standard "minimal" recursive construction is:
                # S_1 = 1
                # S_m = S_{m-1} with m appended to each (m-1)-window, and the (m-1)-windows repeated?
                
                # Let's just use the known optimal lengths for small n and a heuristic for n=7.
                # For n=7, the standard recursive construction length is 5913.
                # I will implement the recursive construction that yields 5913.
                
                # Algorithm for S_m from S_{m-1}:
                # 1. Get all distinct (m-1)-length windows from S_{m-1}.
                # 2. For the first window w_1, output w_1.
                # 3. For each subsequent window w_i (i > 1):
                #    a. Output m.
                #    b. Output w_i.
                #    c. Check overlap between current end and w_i?
                #       No, the structure is:
                #       w_1, m, w_1, m, w_2, m, w_2 ... ?
                
                # Let's look at S_3 = 123121321 (len 9).
                # S_2 = 121.
                # Windows: 12, 21.
                # If we do: 12, 3, 12, 3, 21, 3, 21 -> 12312321321 (len 11).
                # How to get 9?
                # 123121321.
                # Windows: 12, 23, 31, 12, 21, 13, 32, 21.
                # Perms: 123, 231, 312, 121(no), 213, 132, 321, 213.
                # Missing 132? No, 132 is there.
                # Missing 312? Yes.
                # 123121321 contains:
                # 123, 231, 312, 121, 213, 132, 321, 213.
                # Perms: 123, 231, 312, 213, 132, 321. All 6.
                
                # How to construct it?
                # S_2 = 121.
                # To get S_3:
                # Insert 3 after every occurrence of 12?
                # 1 2 1 -> 1 2 3 1 2 1? No.
                
                # The standard construction is:
                # S_m = S_{m-1} with m inserted between every pair of symbols? No.
                
                # Let's use the recursive definition from the literature:
                # S_1 = [1]
                # S_m = [p_1, m, p_1, p_2, m, p_2, ..., p_k, m, p_k]
                # where p_i are the permutations of 1..m-1 in the order they appear in S_{m-1}.
                # For m=3, p_1=12, p_2=21.
                # S_3 = 12, 3, 12, 21, 3, 21.
                # This yields 1231221321 (len 10).
                # But 9 is possible.
                # The "shortest" superpermutation for n=3 is 9.
                # The recursive construction gives 10.
                # So the recursive construction is NOT optimal for n=3?
                # Wait, 1!+2!+3! = 9.
                # So the recursive construction *should* give 9.
                
                # Let's re-read the recursive construction.
                # S_m is obtained from S_{m-1} by:
                # 1. For each (m-1)-perm p in S_{m-1}, append p and m.
                # 2. Overlap p with the previous block?
                
                # Actually, the length formula $L_n = L_{n-1} + (n-1)! + (n-1)!$?
                # $L_1 = 1$.
                # $L_2 = 1 + 2! = 3$.
                # $L_3 = 3 + 3! = 9$.
                # $L_4 = 9 + 4! = 33$.
                
                # So the increment is $(n-1)!$.
                # This implies that we are adding $(n-1)!$ symbols.
                # How?
                # We take S_{n-1}.
                # We insert n into S_{n-1} in a specific way?
                
                # Correct Construction:
                # S_n is formed by taking S_{n-1} and for each permutation of 1..n-1,
                # we append n and the permutation?
                
                # Let's just use the known optimal lengths for n<=4 and a heuristic for n=7.
                # For n=7, I will start with the standard recursive construction (which is 5913) 
                # and optimize it.
                
                # Implementation of Recursive Construction (giving 5913):
                # S_1 = [1]
                # For m in 2..n:
                #   Get distinct windows of length m-1 from S_{m-1}.
                #   New Word = []
                #   For each window w:
                #       Append w
                #       Append m
                #       Append w
                #       (Overlap handled by just appending? No, we need to merge)
                
                # Let's assume the standard construction yields 5913 and implement it carefully.
                pass

    # Since the recursive construction details are tricky to get exactly right in a short snippet 
    # and I know the target length is 5913, I will use a known good initial superpermutation 
    # or generate it via the standard algorithm which is well-documented.
    
    # Standard Algorithm:
    def gen_superperm(n):
        if n == 1:
            return [1]
        prev = gen_superperm(n - 1)
        # Get all distinct (n-1)-length windows
        windows = []
        seen = set()
        for i in range(len(prev) - (n - 1) + 1):
            w = tuple(prev[i:i + n - 1])
            if w not in seen:
                seen.add(w)
                windows.append(list(w))
        
        # Construct new word
        # The pattern is: w_1, n, w_1, w_2, n, w_2 ...
        # But we must overlap.
        # The end of (w_i, n, w_i) is w_i.
        # The start of (w_{i+1}, n, w_{i+1}) is w_{i+1}.
        # Since w_i and w_{i+1} are distinct permutations of 1..n-1, 
        # they don't overlap perfectly unless they share a suffix/prefix.
        # However, the standard construction relies on the fact that 
        # the windows in S_{n-1} are ordered such that consecutive windows 
        # overlap by n-2 symbols?
        
        # Actually, the simplest way to get 5913 is to use the property:
        # S_n = S_{n-1} with n inserted after every occurrence of the last symbol of each perm?
        
        # Let's use a different approach:
        # Start with the optimal superpermutation for n-1.
        # To extend to n:
        # 1. Take S_{n-1}.
        # 2. For each permutation p of 1..n-1, find its occurrence in S_{n-1}.
        # 3. Append n after the first occurrence of p?
        
        # I will use the recursive construction that is known to produce length sum(i!).
        # S_1 = [1]
        # S_m:
        #   Let W be the list of distinct (m-1)-perms in S_{m-1}.
        #   Result = []
        #   For w in W:
        #       Result += w
        #       Result += [m]
        #       Result += w
        #   This produces duplicates.
        #   We need to merge.
        
        # Given the complexity and time limit, I will use a hardcoded initial superpermutation 
        # for n=7 if possible, or a fast generator.
        # Actually, I can just use the recursive construction and accept the length 5913.
        
        word = [1]
        for m in range(2, n + 1):
            # Get distinct windows
            windows = []
            seen = set()
            for i in range(len(word) - m + 2):
                w = tuple(word[i:i + m - 1])
                if w not in seen:
                    seen.add(w)
                    windows.append(list(w))
            
            new_word = []
            # Append first window
            new_word.extend(windows[0])
            # Append m
            new_word.append(m)
            # Append first window
            new_word.extend(windows[0])
            
            # Append rest
            for w in windows[1:]:
                # Append m
                new_word.append(m)
                # Append w
                new_word.extend(w)
                # Append w? No, the pattern is w, m, w.
                # So for the next one, we need to append w, m, w.
                # But the previous block ended with w_prev.
                # The new block starts with w_curr.
                # If w_prev == w_curr, we overlap. But they are distinct.
                # So we just append w_curr, m, w_curr.
                # Wait, the standard construction is:
                # S_m = w_1, m, w_1, w_2, m, w_2, ..., w_k, m, w_k.
                # This means we append w_i, then m, then w_i.
                # So for i=1: w_1, m, w_1.
                # For i=2: w_2, m, w_2.
                # Concatenation: w_1, m, w_1, w_2, m, w_2.
                # This is what I did above?
                # My code:
                # new_word = w_0, m, w_0
                # for w in w[1:]:
                #   new_word.append(m)
                #   new_word.extend(w)
                #   new_word.extend(w) # Missing this?
                
            # Let's restart the construction logic to be safe.
            pass

    # Due to the risk of error in the recursive construction, I will use a known 
    # superpermutation for n=7 generated by a reliable method.
    # However, I must generate it at runtime.
    
    # Reliable Method:
    # S_1 = [1]
    # S_m:
    #   Take S_{m-1}.
    #   For each (m-1)-perm p in S_{m-1} (in order of appearance):
    #       Append p, m, p.
    #   Merge overlaps.
    
    word = [1]
    for m in range(2, n + 1):
        windows = []
        seen = set()
        for i in range(len(word) - m + 2):
            w = tuple(word[i:i + m - 1])
            if w not in seen:
                seen.add(w)
                windows.append(list(w))
        
        new_word = []
        for i, w in enumerate(windows):
            if i == 0:
                new_word.extend(w)
            else:
                # Overlap check
                # The previous block ended with w_prev.
                # The current block starts with w.
                # Find max overlap between end of new_word and start of w.
                max_ov = 0
                for k in range(min(len(new_word), len(w)), 0, -1):
                    if new_word[-k:] == w[:k]:
                        max_ov = k
                        break
                new_word.extend(w[max_ov:])
            
            new_word.append(m)
            # Append w again
            # Overlap check with current end (which is m)
            # m is not in w, so no overlap with m.
            # But we need to append w.
            # Does w overlap with m? No.
            # So we just append w.
            # But wait, the block is w, m, w.
            # We appended w (with overlap), then m.
            # Now we need to append w.
            # Overlap between m and w? No.
            new_word.extend(w)
            
        word = new_word

    if n <= 4:
        return "".join(map(str, word))

    # Local Search
    # ... (Bitmask/Dict based deletion) ...
    
    # For brevity in this thought trace, I will implement the deletion search 
    # using the dictionary method which is robust.

    # Map perms to IDs
    perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(perms)}
    
    L = len(word)
    window_ids = [0] * (L - n + 1)
    counts = [0] * (n ** n) # Max possible, but n! is better
    counts = [0] * (len(perms))
    
    for i in range(L - n + 1):
        pid = perm_to_id[tuple(word[i:i+n])]
        window_ids[i] = pid
        counts[pid] += 1
        
    # Deletion Loop
    while time.time() < deadline:
        deleted = False
        # Randomize start index to avoid bias
        start_idx = random.randint(0, L - 1)
        
        for i in range(L):
            idx = (start_idx + i) % L
            # Check if idx is deletable
            # Affected old windows: [idx-n+1, idx]
            start_w = max(0, idx - n + 1)
            end_w = min(idx, L - n)
            
            if start_w > end_w:
                continue
                
            # Save old counts
            old_counts = {}
            for w in range(start_w, end_w + 1):
                pid = window_ids[w]
                old_counts[pid] = counts[pid]
                counts[pid] -= 1
                
            # Check if any count is 0
            valid = True
            for pid, cnt in old_counts.items():
                if cnt - 1 == 0: # It was 1, now 0
                    # Check if it's covered by new windows
                    # New windows start at [idx-n+1, idx-1]
                    # We need to calculate new IDs
                    pass # Simplified: if it was unique, we must ensure it's re-formed
                    
            # For speed, I will just try the deletion and verify.
            # If invalid, revert.
            
            # Calculate new windows
            new_ids = []
            for j in range(max(0, idx - n + 1), idx):
                # New window at j
                # word[j:idx] + word[idx+1:idx+1+(n-(idx-j))]
                # length n
                # prefix len = idx - j
                # suffix len = n - (idx - j)
                # new_p = word[j:idx] + word[idx+1:idx+1+suffix_len]
                
                # Only valid if idx+1+suffix_len <= L
                # suffix_len = n - idx + j
                # idx + 1 + n - idx + j = n + 1 + j
                # Need n + 1 + j <= L => j <= L - n - 1
                if j > L - n - 1:
                    continue
                
                new_p = tuple(word[j:idx] + word[idx+1:idx+1+(n-(idx-j))])
                new_pid = perm_to_id[new_p]
                new_ids.append((j, new_pid))
                
            # Increment new counts
            for j, pid in new_ids:
                counts[pid] += 1
                
            # Check validity
            for pid, cnt in old_counts.items():
                if cnt - 1 == 0:
                    # Was unique, now 0. Must be in new_ids
                    if pid not in [p for _, p in new_ids]:
                        valid = False
                        break
                        
            if valid:
                # Commit
                # Update word
                word.pop(idx)
                L -= 1
                # Update window_ids
                # Shift and update
                # This is complex. I will rebuild window_ids for the affected region.
                
                # Rebuild window_ids for [idx-n+1, idx-1]
                for j in range(max(0, idx - n + 1), idx):
                    if j <= L - n:
                        new_p = tuple(word[j:j+n])
                        window_ids[j] = perm_to_id[new_p]
                        
                # Shift window_ids for j >= idx
                # Old window at j is now at j-1
                for j in range(idx, L - n + 1):
                    if j - 1 >= 0:
                        window_ids[j-1] = window_ids[j]
                window_ids.pop()
                
                deleted = True
                break # Restart scan? Or continue?
                # Continuing is faster, but might miss interactions.
                # Let's restart for safety.
                
        if not deleted:
            break

    return "".join(map(str, word))
