import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if n == 1:
        return "1"
    
    # 1. Generate classic superpermutation
    # S_1 = "1"
    # S_k = S_{k-1} -> (perm, k, perm) for each perm in S_{k-1}
    word = [1]
    for m in range(2, n + 1):
        # Extract unique (m-1)-length permutations from current word
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block is p + [m] + p
            # But we can overlap with the previous block
            block = list(p) + [m] + list(p)
            
            # Find max overlap with end of new_word
            overlap = 0
            max_ov = min(len(new_word), len(block))
            for o in range(max_ov, 0, -1):
                if new_word[-o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        word = new_word
        
    # 2. Prepare for Greedy Deletion
    target_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts
    perm_counts = {}
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in target_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    # Helper to check if deletion at idx is valid
    def can_delete(idx):
        # Windows containing idx: start from max(0, idx - n + 1) to min(idx, len(word) - n)
        start = max(0, idx - n + 1)
        end = min(idx, len(word) - n)
        
        if start > end:
            return True # No valid windows? Should not happen if 0 <= idx < len(word) and len(word)>=n
            
        affected_counts = {}
        for i in range(start, end + 1):
            p = tuple(word[i:i + n])
            # p is guaranteed to be in target_perms if we started with valid superperm
            # and only deleted validly? Yes.
            if p in perm_counts:
                affected_counts[p] = affected_counts.get(p, 0) + 1
                
        for p, cnt in affected_counts.items():
            if perm_counts[p] - cnt <= 0:
                return False
        return True

    def do_delete(idx):
        start = max(0, idx - n + 1)
        end = min(idx, len(word) - n)
        affected_counts = {}
        for i in range(start, end + 1):
            p = tuple(word[i:i + n])
            if p in perm_counts:
                affected_counts[p] = affected_counts.get(p, 0) + 1
                
        for p, cnt in affected_counts.items():
            perm_counts[p] -= cnt
            if perm_counts[p] == 0:
                del perm_counts[p]
        
        word.pop(idx)

    # 3. Greedy Deletion Loop
    # We do multiple passes. In each pass, we try to delete indices in a randomized order.
    # If a deletion happens, the list shrinks, so indices change.
    # To handle this, we can iterate from 0 to end, and if we delete at i, we stay at i.
    # Randomization: We can shuffle the *initial* order? No, that breaks the "stay at i" logic.
    # Instead, we can do forward passes and backward passes.
    # Or, we can maintain a list of indices to check.
    
    # A simple robust strategy:
    # Repeat until no progress or time out:
    #   progress = False
    #   for i in range(len(word)):
    #       if can_delete(i):
    #           do_delete(i)
    #           progress = True
    #           # do not increment i
    #       else:
    #           i += 1
    #   if not progress:
    #       # Try reverse
    #       progress = False
    #       for i in range(len(word)-1, -1, -1):
    #           if can_delete(i):
    #               do_delete(i)
    #               progress = True
    #           else:
    #               i -= 1
    #       if not progress:
    #           break

    # To add more variety, we can also try "random restarts" of the deletion process
    # by re-randomizing the word? No, that's dangerous.
    # We can try to delete in a random order by creating a list of indices, shuffling it,
    # and then checking them. If an index is deleted, we remove it from the list and adjust
    # subsequent indices? That's complex.
    
    # Let's stick to Forward/Backward passes, but interleave them or do multiple rounds.
    
    rounds = 0
    while time.time() < deadline - 2:
        progress = False
        
        # Forward pass
        i = 0
        while i < len(word):
            if can_delete(i):
                do_delete(i)
                progress = True
                # i stays same
            else:
                i += 1
                
        if not progress:
            # Backward pass
            i = len(word) - 1
            while i >= 0:
                if can_delete(i):
                    do_delete(i)
                    progress = True
                    # i stays same? 
                    # If we delete i, the element at i+1 moves to i.
                    # We want to check the new element at i.
                    # So we don't decrement i.
                else:
                    i -= 1
                    
        if not progress:
            # Try a few more randomized forward/backward mixes?
            # Or just break.
            # For n=7, this usually converges quickly.
            break
            
    return "".join(map(str, word))
