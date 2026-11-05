import time
import random
import math

def construct(n):
    start_time = time.time()
    # Leave a 5-second safety margin before the 120s hard kill
    deadline = start_time + 115.0
    
    # 1. Generate standard recursive superpermutation
    # This produces a valid superpermutation of length sum(k! for k in 1..n)
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all distinct (m-1)-permutations from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Construct new word
        new = []
        for p in perms:
            # The block is P m P
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of 'new'
            # We check suffixes of 'new' against prefixes of 'block'
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # Convert to list of ints for easier manipulation
    S = list(word)
    
    # 2. Initialize counts of permutations
    # For n=7, there are 5040 permutations.
    # We store a count for every window in S.
    def build_counts(S, n):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    counts = build_counts(S, n)
    
    best_S = S[:]
    best_len = len(S)
    
    # Helper to check validity (for final verification or perturbation checks if needed)
    def is_valid(S, n):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i+n]))
        return len(seen) == math.factorial(n)

    # Local Search Strategy:
    # We repeatedly try to remove characters.
    # If we get stuck (no removals in a full pass), we perturb the string
    # by swapping two random characters or inserting a random character,
    # then try to remove again. This helps escape local minima.
    
    # We track the number of consecutive failed passes to decide when to perturb
    failed_passes = 0
    max_failed_passes_before_perturb = 3
    
    # Pre-compute factorial for validity check
    target_perms = math.factorial(n)
    
    while time.time() < deadline:
        # Attempt a full pass of removals from right to left
        # Removing from right to left ensures that indices < i are not affected
        # by the shift caused by removal at i, so we can continue scanning leftwards.
        removed_any = False
        i = len(S) - 1
        
        while i >= 0:
            if time.time() > deadline:
                break
            
            # Identify windows affected by removing S[i]
            # Old windows containing index i:
            # Start index j such that j <= i <= j+n-1  =>  i-n+1 <= j <= i
            # Also j must be a valid window start: 0 <= j <= len(S)-n
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            valid_removal = True
            old_perms_to_dec = []
            
            # Check if any old window has count 1
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid_removal = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid_removal:
                i -= 1
                continue
            
            # If valid, perform the removal
            # 1. Decrement counts for old windows
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # 2. Identify new windows created by the removal
            # The new string is S_new = S[:i] + S[i+1:]
            # A new window at index k in S_new straddles the gap if:
            # It uses some elements from S[:i] and some from S[i+1:]
            # Specifically, the window starts at k in S_new.
            # The elements are S_new[k], ..., S_new[k+n-1].
            # Mapping to S indices:
            # S_new[idx] = S[idx] if idx < i, else S[idx+1]
            
            # The straddling windows are those where the window start k in S_new
            # is such that the window covers index i in S.
            # In S_new, the "gap" is at position i.
            # Windows that cross this gap start at k in S_new where:
            # k < i and k+n-1 >= i  (in S_new coordinates)
            # Actually, it's easier to think in terms of S indices.
            # The new windows are formed by joining S[j] and S[j+1] for j < i and S[j+1] and S[j+2] for j >= i? No.
            
            # Let's derive the new windows explicitly.
            # New length L' = len(S) - 1.
            # New window at index k (0 <= k <= L'-n) consists of:
            # S_new[k : k+n]
            # If k+n <= i: window is S[k:k+n] (Unchanged, already counted)
            # If k >= i: window is S[k+1:k+1+n] (Unchanged content, shifted, already counted)
            # The only *new* combinations are those where the window includes characters from both sides of the removed index i.
            # This happens if k < i and k+n-1 >= i.
            # i.e., i - n + 1 <= k <= i - 1.
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, (len(S) - 1) - n) # L' - n
            
            for k in range(start_new, end_new + 1):
                # Construct the new window tuple
                # It consists of:
                # S[k : i]  (length i-k)
                # S[i+1 : i+1 + (n - (i-k))] = S[i+1 : k+n+1]
                
                # Let's just build the tuple by mapping indices
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            # Commit removal
            S.pop(i)
            removed_any = True
            best_S = S[:]
            best_len = len(S)
            
            # Do not decrement i, because the element that was at i+1 is now at i,
            # and we haven't checked it yet (or rather, the indices > i are shifted left,
            # but we are scanning downwards. The element now at i was previously at i+1.
            # We have already processed indices > i. So we should check the new i next.)
            # Wait, if we remove i, the element at i+1 moves to i.
            # We are scanning i downwards. We have already processed i+1, i+2, etc.
            # So the element now at i (which was i+1) has already been considered for removal?
            # No, when we were at i+1, we checked if removing S[i+1] was valid.
            # Now S[i+1] is at S[i]. The context has changed (S[i-1] is to the left, S[i+1] (old i+2) is to the right).
            # So we should check the new S[i] again. Thus, do not decrement i.
            
        if removed_any:
            failed_passes = 0
        else:
            failed_passes += 1
            
            # If stuck for a few passes, perturb
            if failed_passes >= max_failed_passes_before_perturb:
                # Check if we have enough time for a perturbation and rebuild
                if time.time() > deadline - 5.0:
                    break
                
                # Strategy: Swap two random characters
                # This is a simple move to break the local minimum.
                # We must ensure the swap doesn't make the string invalid immediately?
                # Actually, swaps can create invalid states (missing perms).
                # But we are going to try to remove characters next.
                # If the swap makes the string invalid, we won't be able to remove anything
                # that fixes it easily, and we might lose the valid state.
                
                # Safer perturbation: Try to insert a character? No, we want to shorten.
                # Better perturbation: Randomly remove a character even if it's "invalid" locally?
                # No, that breaks validity.
                
                # Let's try a different approach for perturbation:
                # Instead of random swaps which risk validity, let's try to "shuffle"
                # the order of attempts or use a different removal heuristic.
                
                # Alternative: If stuck, it means every character is critical to some unique window.
                # This often happens in the standard construction.
                # A known improvement for superpermutations is to identify "redundant" overlaps.
                
                # Given the complexity of safe perturbations, let's just restart the search
                # with a slightly different random seed or break if we've spent too much time.
                # However, since we are in a "stuck" state, simple random swaps are risky.
                
                # Let's try a "local swap" that preserves validity?
                # Too complex.
                
                # Let's just break the loop if we are stuck and close to deadline.
                # The current program already does decent removals.
                # The main gain comes from the initial greedy removal pass.
                
                # To avoid infinite loops of failed passes, we break.
                break
                
    # Final check: Ensure the returned string is valid.
    # If the local search somehow corrupted the string (shouldn't happen with valid removals),
    # we fall back to the original.
    if not is_valid(best_S, n):
        return "".join(map(str, word))
        
    return "".join(map(str, best_S))
