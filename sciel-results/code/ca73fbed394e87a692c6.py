import time
import itertools
import random

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    """
    start_time = time.time()
    deadline = 115.0  # 5 seconds margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            target_overlap = m - 2
            overlap = 0
            if len(new_word) >= target_overlap and len(block) >= target_overlap:
                if new_word[-target_overlap:] == block[:target_overlap]:
                    overlap = target_overlap
                else:
                    max_overlap = min(len(new_word), len(block))
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            break
                        if new_word[len(new_word) - o:] == block[:o]:
                            overlap = o
                            break
            else:
                max_overlap = min(len(new_word), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
            
            new_word.extend(block[overlap:])
        
        word = new_word
        
    current_word = word[:]
    
    # Pre-compute all permutations and map to integer codes for speed
    all_perms = list(itertools.permutations(range(1, n+1)))
    num_perms = len(all_perms)
    
    # Use base-8 encoding for windows
    pow8 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow8[i] = pow8[i-1] * 8
        
    code_to_pidx = {}
    for idx, p in enumerate(all_perms):
        c = 0
        for i in range(n):
            c += p[i] * pow8[n - 1 - i]
        code_to_pidx[c] = idx

    def window_code(w, start):
        c = 0
        for i in range(n):
            c += w[start + i] * pow8[n - 1 - i]
        return c

    L = len(current_word)
    num_windows = L - n + 1
    wcodes = [0] * num_windows
    for i in range(num_windows):
        wcodes[i] = window_code(current_word, i)
    
    counts = [0] * num_perms
    for wc in wcodes:
        pidx = code_to_pidx.get(wc, -1)
        if pidx >= 0:
            counts[pidx] += 1

    def rebuild():
        nonlocal wcodes, counts, L, num_windows
        L = len(current_word)
        num_windows = L - n + 1
        wcodes = [0] * num_windows
        for i in range(num_windows):
            wcodes[i] = window_code(current_word, i)
        counts = [0] * num_perms
        for wc in wcodes:
            pidx = code_to_pidx.get(wc, -1)
            if pidx >= 0:
                counts[pidx] += 1

    def try_delete(pos):
        nonlocal wcodes, counts, L, num_windows
        L = len(current_word)
        num_windows = L - n + 1
        start_j = max(0, pos - n + 1)
        end_j = min(pos, num_windows - 1)
        
        valid = True
        for j in range(start_j, end_j + 1):
            pidx = code_to_pidx.get(wcodes[j], -1)
            if pidx >= 0 and counts[pidx] == 1:
                valid = False
                break
        
        if valid:
            current_word.pop(pos)
            rebuild()
            return True
        return False

    def try_swap(pos1, pos2):
        """Swap pos1 and pos2 if valid."""
        nonlocal wcodes, counts, L, num_windows
        L = len(current_word)
        if pos1 == pos2:
            return False
        if pos1 > pos2:
            pos1, pos2 = pos2, pos1
            
        # Affected windows: those containing pos1 or pos2
        # Window j contains indices j..j+n-1
        # Contains pos1 if pos1-n+1 <= j <= pos1
        # Contains pos2 if pos2-n+1 <= j <= pos2
        # Union: max(0, pos1-n+1) to min(pos2, L-n)
        
        start_j1 = max(0, pos1 - n + 1)
        end_j1 = min(pos1, L - n)
        start_j2 = max(0, pos2 - n + 1)
        end_j2 = min(pos2, L - n)
        
        start_j = min(start_j1, start_j2)
        end_j = max(end_j1, end_j2)
        
        # Check validity
        old_wcodes = {}
        for j in range(start_j, end_j + 1):
            old_wcodes[j] = wcodes[j]
            
        # Simulate swap
        current_word[pos1], current_word[pos2] = current_word[pos2], current_word[pos1]
        
        valid = True
        for j in range(start_j, end_j + 1):
            new_code = window_code(current_word, j)
            old_code = old_wcodes[j]
            
            old_pidx = code_to_pidx.get(old_code, -1)
            new_pidx = code_to_pidx.get(new_code, -1)
            
            # If old was unique and new is different, count drops to 0 -> Invalid
            if old_pidx >= 0 and new_pidx != old_pidx:
                if counts[old_pidx] == 1:
                    valid = False
                    break
            
        # Revert swap if invalid
        if not valid:
            current_word[pos1], current_word[pos2] = current_word[pos2], current_word[pos1]
            return False
            
        # Apply swap
        # Update wcodes and counts
        for j in range(start_j, end_j + 1):
            old_code = old_wcodes[j]
            new_code = window_code(current_word, j)
            
            old_pidx = code_to_pidx.get(old_code, -1)
            new_pidx = code_to_pidx.get(new_code, -1)
            
            if old_pidx >= 0:
                counts[old_pidx] -= 1
            if new_pidx >= 0:
                counts[new_pidx] += 1
            wcodes[j] = new_code
            
        return True

    # Main Loop
    no_improve_count = 0
    best_length = len(current_word)
    best_word = current_word[:]
    
    while time.time() < deadline:
        improved = False
        
        # 1. Backward Greedy Delete
        L = len(current_word)
        for i in range(L - 1, -1, -1):
            if try_delete(i):
                improved = True
                if len(current_word) < best_length:
                    best_length = len(current_word)
                    best_word = current_word[:]
                break
        if improved:
            no_improve_count = 0
            continue
        
        # 2. Forward Greedy Delete
        L = len(current_word)
        for i in range(L):
            if try_delete(i):
                improved = True
                if len(current_word) < best_length:
                    best_length = len(current_word)
                    best_word = current_word[:]
                break
        if improved:
            no_improve_count = 0
            continue
            
        # 3. Random Delete
        L = len(current_word)
        positions = list(range(L))
        random.shuffle(positions)
        for i in positions:
            if try_delete(i):
                improved = True
                if len(current_word) < best_length:
                    best_length = len(current_word)
                    best_word = current_word[:]
                break
        if improved:
            no_improve_count = 0
            continue
            
        # 4. Adjacent Swap
        L = len(current_word)
        if L > 1:
            pos = random.randint(0, L - 2)
            if try_swap(pos, pos + 1):
                improved = True
                no_improve_count = 0
                continue
        
        # 5. Distant Swap
        L = len(current_word)
        if L > 2:
            pos1 = random.randint(0, L - 1)
            pos2 = random.randint(0, L - 1)
            if pos1 != pos2:
                if try_swap(pos1, pos2):
                    improved = True
                    no_improve_count = 0
                    continue

        # 6. Stuck, try a few more random swaps
        for _ in range(20):
            if time.time() > deadline: break
            L = len(current_word)
            if L > 1:
                pos1 = random.randint(0, L - 1)
                pos2 = random.randint(0, L - 1)
                if pos1 != pos2:
                    if try_swap(pos1, pos2):
                        improved = True
                        no_improve_count = 0
                        break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 200:
                # Restart with perturbation
                # Restore best word and perturb
                current_word = best_word[:]
                rebuild()
                # Apply a few random valid swaps to perturb
                for _ in range(5):
                    L = len(current_word)
                    if L > 1:
                        pos1 = random.randint(0, L - 1)
                        pos2 = random.randint(0, L - 1)
                        if pos1 != pos2:
                            try_swap(pos1, pos2)
                no_improve_count = 0

    return "".join(map(str, best_word))
