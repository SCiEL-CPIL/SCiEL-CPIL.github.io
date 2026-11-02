import time
import random

def _standard_path(n):
    """Generates the standard recursive path of permutations."""
    perms = [tuple([1])]
    for m in range(2, n + 1):
        new_perms = []
        for p in perms:
            # In the standard construction, we don't just expand, we interleave.
            # Actually, the path is defined by the order of permutations in the superpermutation.
            # The standard superpermutation S_n contains all perms of 1..n.
            # The order is: for each perm P of 1..n-1 in order in S_{n-1}, 
            # we have P, n, P.
            # This creates a specific sequence of n-permutations.
            # Let's generate the word first, then extract the path.
            pass
        
    # Easier: Generate the word, then extract the unique n-permutations in order of appearance.
    word = [1]
    for m in range(2, n + 1):
        # Extract perms of size m-1 from current word
        prev_perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            w = tuple(word[i:i + m - 1])
            if w not in seen:
                seen.add(w)
                prev_perms.append(w)
        
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            # Overlap
            overlap = 0
            for o in range(min(len(new_word), len(block)), -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
        
    # Extract path of n-permutations
    path = []
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if w not in seen:
            seen.add(w)
            path.append(w)
            
    return path

def _overlap(p1, p2):
    """Calculate maximum overlap between end of p1 and start of p2."""
    max_k = min(len(p1), len(p2))
    for k in range(max_k, -1, -1):
        if k == 0:
            return 0
        if p1[-k:] == p2[:k]:
            return k
    return 0

def _build_word_from_path(path, n):
    """Construct the superpermutation word from a given path of permutations."""
    if not path:
        return []
    word = list(path[0])
    for i in range(1, len(path)):
        p_prev = path[i-1]
        p_curr = path[i]
        k = _overlap(p_prev, p_curr)
        word += list(p_curr[k:])
    return word

def _is_valid_superperm(word, n):
    """Check if word contains all permutations of 1..n."""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i+n]))
    return len(seen) == len(seen) and len(seen) == 24 if n==4 else (len(seen) == 5040 if n==7 else True)
    # More general check
    import math
    return len(seen) == math.factorial(n)

def construct(n):
    if n <= 6:
        # Standard construction is optimal for n <= 6
        path = _standard_path(n)
        word = _build_word_from_path(path, n)
        return "".join(map(str, word))
    
    if n == 7:
        start_time = time.time()
        deadline = start_time + 110.0 # Safety margin
        
        # 1. Generate standard path
        path = _standard_path(n)
        N = len(path)
        
        # 2. Local Search to improve path
        # Objective: Maximize sum of overlaps
        # Current overlap sum
        curr_overlap_sum = 0
        for i in range(N - 1):
            curr_overlap_sum += _overlap(path[i], path[i+1])
            
        improved = False
        # Simple hill climbing with random restarts if stuck
        # Since N=5040, O(N) per pass is fine.
        # We try to swap adjacent elements if it increases overlap.
        
        # Precompute overlaps? No, N^2 is too big (25M).
        # Just calculate on the fly for neighbors.
        
        while time.time() < deadline:
            improved_in_pass = False
            # Forward pass
            for i in range(N - 1):
                if time.time() > deadline:
                    break
                
                p1 = path[i]
                p2 = path[i+1]
                
                # Current contributions
                # Overlap with prev (if i > 0)
                ov_prev = _overlap(path[i-1], p1) if i > 0 else 0
                ov_curr = _overlap(p1, p2)
                # Overlap with next (if i+1 < N-1)
                ov_next = _overlap(p2, path[i+2]) if i + 1 < N - 1 else 0
                
                # Total local overlap
                total_old = ov_prev + ov_curr + ov_next
                
                # Try swap
                # New order: p2, p1
                # ov_prev_new = overlap(path[i-1], p2)
                # ov_swap_new = overlap(p2, p1)
                # ov_next_new = overlap(p1, path[i+2])
                
                ov_prev_new = _overlap(path[i-1], p2) if i > 0 else 0
                ov_swap_new = _overlap(p2, p1)
                ov_next_new = _overlap(p1, path[i+2]) if i + 1 < N - 1 else 0
                
                total_new = ov_prev_new + ov_swap_new + ov_next_new
                
                if total_new > total_old:
                    path[i], path[i+1] = path[i+1], path[i]
                    curr_overlap_sum += (total_new - total_old)
                    improved_in_pass = True
                    # After swap, the element at i is now p2.
                    # We should check i-1 again? 
                    # To be safe, we can just continue, or go back.
                    # Going back is safer for hill climbing.
                    if i > 0:
                        i -= 1
                        
            if not improved_in_pass:
                # Try a few random swaps to escape local minima?
                # Or just break if no improvement in a full pass.
                # Given the small gap, maybe simple hill climbing is enough.
                # Let's try a few random perturbations if stuck.
                if random.random() < 0.1:
                    # Random swap
                    i = random.randint(0, N - 2)
                    j = i + 1
                    p1 = path[i]
                    p2 = path[j]
                    ov_prev = _overlap(path[i-1], p1) if i > 0 else 0
                    ov_curr = _overlap(p1, p2)
                    ov_next = _overlap(p2, path[i+2]) if i + 1 < N - 1 else 0
                    total_old = ov_prev + ov_curr + ov_next
                    
                    ov_prev_new = _overlap(path[i-1], p2) if i > 0 else 0
                    ov_swap_new = _overlap(p2, p1)
                    ov_next_new = _overlap(p1, path[i+2]) if i + 1 < N - 1 else 0
                    total_new = ov_prev_new + ov_swap_new + ov_next_new
                    
                    # Accept if not worse? Or just accept to move?
                    # Let's accept if better or equal to explore.
                    if total_new >= total_old:
                        path[i], path[j] = path[j], path[i]
                        curr_overlap_sum += (total_new - total_old)
                        improved_in_pass = True
                else:
                    break
        
        # 3. Build word
        word = _build_word_from_path(path, n)
        
        # 4. Verify
        if _is_valid_superperm(word, n):
            return "".join(map(str, word))
        else:
            # Fallback to standard
            path_std = _standard_path(n)
            word_std = _build_word_from_path(path_std, n)
            return "".join(map(str, word_std))

    else:
        # Fallback for other n
        path = _standard_path(n)
        word = _build_word_from_path(path, n)
        return "".join(map(str, word))
