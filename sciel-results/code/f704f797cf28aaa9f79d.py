import time
import random
from itertools import permutations

def construct(n):
    """Construct shortest superpermutation over symbols 1..n."""
    start_time = time.time()
    deadline = 110.0  # seconds, safety margin before 120s kill
    
    if n == 1:
        return "1"
    
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234121341213142312431214321"
    
    # For n >= 5, use Greedy Construction + Local Search
    # 1. Greedy Construction
    word = _greedy_construct(n)
    
    # 2. Local Search to shorten
    word = _local_search_shorten(word, n, start_time, deadline)
    
    return "".join(map(str, word))

def _greedy_construct(n):
    """
    Greedy construction of a superpermutation.
    Iterates through permutations in a fixed order (lexicographic) and appends
    each permutation with maximum possible overlap to the current word.
    """
    perms = list(permutations(range(1, n + 1)))
    # Randomizing the order can sometimes help, but lexicographic is deterministic.
    # Let's use a random shuffle to break symmetries and potentially find a better base.
    random.shuffle(perms)
    
    word = []
    for p in perms:
        p_list = list(p)
        # Find max overlap between end of word and start of p_list
        max_ov = min(len(word), len(p_list))
        overlap = 0
        for o in range(max_ov, 0, -1):
            if word[-o:] == p_list[:o]:
                overlap = o
                break
        
        word.extend(p_list[overlap:])
        
    return word

def _local_search_shorten(word, n, start_time, deadline):
    """
    Shorten the word by removing characters that are not part of any 'critical' window.
    A critical window is one that appears exactly once in the word.
    Removing a character from a critical window would lose that permutation.
    """
    time_limit = deadline - (time.time() - start_time)
    if time_limit < 5:
        return word
    
    # Build set of all required permutations
    all_perms = set(permutations(range(1, n + 1)))
    
    # Initial check
    if not _is_valid(word, n, all_perms):
        return word
    
    best_word = list(word)
    best_len = len(word)
    
    # Main loop: repeatedly identify and remove non-critical characters
    improved = True
    while improved:
        if time.time() - start_time > deadline - 5:
            break
            
        improved = False
        word = best_word
        
        # 1. Count occurrences of each window
        perm_count = {}
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i+n])
            perm_count[w] = perm_count.get(w, 0) + 1
        
        # 2. Identify critical windows (count == 1)
        # A character at index i is 'safe' to remove if it is NOT part of any critical window.
        # A window at index j covers indices j to j+n-1.
        # So, if window w at index j is critical, indices j..j+n-1 are 'protected'.
        
        protected = [False] * len(word)
        
        for w, count in perm_count.items():
            if count == 1:
                # Find the index of this window. 
                # Note: If count is 1, we know it appears exactly once.
                # We need to find where it is.
                # Instead of searching, we can iterate through windows again.
                pass
        
        # More efficient: iterate through windows once
        # If a window has count 1, mark its indices as protected.
        # But we need to know the index of the window to mark it.
        # We can store the index during the first pass if count is 1? 
        # No, we don't know if it's 1 until the end of the pass.
        
        # Pass 1: Count
        # Pass 2: If count==1, mark indices.
        
        # Let's optimize:
        # We can just check: for each index i, is it part of ANY window that appears only once?
        # This is equivalent to: for each window w appearing once, mark its indices.
        
        # To avoid O(N) search for each critical window, we can just iterate windows.
        # If perm_count[w] == 1, we need to find its position.
        # Since it appears only once, we can find it by scanning.
        # This makes the step O(N * n) in worst case if many critical windows.
        # But N is ~6000, n=7. 42000 ops. Very fast.
        
        # Let's do it.
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i+n])
            if perm_count[w] == 1:
                for k in range(i, i+n):
                    protected[k] = True
        
        # 3. Find removable indices
        removable_indices = [i for i in range(len(word)) if not protected[i]]
        
        if not removable_indices:
            break
            
        # 4. Remove them. 
        # We can remove all at once? 
        # If we remove index i, it affects windows around i.
        # If we remove all 'safe' indices, we might break the 'safety' of others?
        # No, 'safety' was defined by the current state. 
        # Removing safe characters cannot make a previously critical window disappear 
        # (it would only reduce counts of other windows).
        # Wait. If I remove a character, I remove windows.
        # If I remove a character that is NOT in a critical window, I am not removing any critical window.
        # So all critical windows remain.
        # However, removing a character might make a NEW window critical?
        # No, removing characters reduces the number of windows. 
        # It can only decrease counts. 
        # So a window that was count 2 might become count 1 (become critical).
        # But it cannot become count 0 if it was count 1 (because we didn't touch critical windows).
        # So the set of critical windows can only grow or stay same?
        # Actually, if a window was count 2, and we remove a char from one of its occurrences, 
        # it becomes count 1. So it becomes critical.
        # This means we might have removed a char that was safe *before* the removal, 
        # but now it is part of a critical window.
        # However, we are removing the character, so it's gone.
        # The issue is: did we remove a character that was protecting a window?
        # We only removed characters NOT in any critical window.
        # So we did not remove any part of a critical window.
        # Therefore, all previously critical windows are still intact.
        # So the removal is valid.
        
        # We can remove all removable indices at once to speed up.
        # Create a set of indices to remove for O(1) lookup
        to_remove = set(removable_indices)
        
        new_word = [x for i, x in enumerate(word) if i not in to_remove]
        
        if len(new_word) < n:
            break
            
        # Verify validity just in case (should be valid by logic above)
        if _is_valid(new_word, n, all_perms):
            word = new_word
            if len(word) < best_len:
                best_word = list(word)
                best_len = len(word)
            improved = True
        else:
            # If invalid, fall back to removing one by one?
            # Or just stop.
            # Given the logic, it should be valid.
            # If not, it means my logic about "safe" removal is flawed.
            # Let's be safe and stop.
            break
            
    return best_word

def _is_valid(word, n, all_perms):
    """Check if word contains all permutations."""
    if len(word) < n:
        return False
    windows = set()
    for i in range(len(word) - n + 1):
        windows.add(tuple(word[i:i+n]))
    return all_perms.issubset(windows)
