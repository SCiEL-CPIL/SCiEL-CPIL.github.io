import time
import itertools
import random

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    Uses greedy nearest-neighbor construction followed by greedy deletion with count-based validity check.
    """
    start_time = time.time()
    deadline = 140.0  # 10 seconds margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic recursive superpermutation with greedy block ordering
    word = [1]
    for m in range(2, n + 1):
        # Collect all unique (m-1)-permutations from the current word
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        # Create blocks: P + [m] + P
        blocks = []
        for p in perms:
            block = list(p) + [m] + list(p)
            blocks.append(block)
        
        # Greedy Nearest Neighbor Ordering
        # We want to order blocks to maximize overlap between consecutive blocks.
        # Overlap of A and B is the largest k such that A[-k:] == B[:k].
        
        num_blocks = len(blocks)
        if num_blocks == 0:
            continue
            
        # Pre-compute overlaps between all pairs? 
        # For m=7, num_blocks=720. 720*720 = 518,400 pairs.
        # Each overlap check is O(m). 518,400 * 7 ~ 3.6M ops. Fast enough.
        
        # To speed up, we can just compute on the fly or precompute a matrix.
        # Let's precompute for simplicity and speed in the loop.
        # overlap[i][j] = max overlap of blocks[i] and blocks[j]
        
        # Actually, we can just do a greedy selection.
        # Start with block 0.
        current_block_idx = 0
        new_word = blocks[current_block_idx][:]
        visited = [False] * num_blocks
        visited[current_block_idx] = True
        
        for _ in range(num_blocks - 1):
            last_suffix = new_word
            best_next_idx = -1
            best_overlap = -1
            
            # Find next block with max overlap
            # Optimization: only check unvisited
            for j in range(num_blocks):
                if visited[j]:
                    continue
                b = blocks[j]
                # Calculate overlap
                max_possible = min(len(last_suffix), len(b))
                # We can limit overlap to m-2 or so? No, could be larger if structure allows, 
                # but typically m-2 is the max useful overlap for this construction.
                # Actually, max overlap is at most m-2 because the middle element is m.
                # Wait, block is P + [m] + P.
                # Suffix of A: ... P_A
                # Prefix of B: P_B ...
                # Overlap is determined by suffix of P_A and prefix of P_B.
                # Max overlap is m-2.
                
                overlap = 0
                # Check from max possible down to 0
                # We can start from m-2
                for k in range(min(m-2, max_possible), -1, -1):
                    if k == 0:
                        break
                    if last_suffix[-k:] == b[:k]:
                        overlap = k
                        break
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_next_idx = j
            
            if best_next_idx == -1:
                break
                
            visited[best_next_idx] = True
            new_word.extend(blocks[best_next_idx][best_overlap:])
            current_block_idx = best_next_idx
        
        word = new_word
        
    current_word = word[:]
    
    # Pre-compute all permutations and map to integer codes for speed
    all_perms = list(itertools.permutations(range(1, n+1)))
    num_perms = len(all_perms)
    
    # Use base-8 encoding for windows (digits 1-7 fit in base 8)
    pow8 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow8[i] = pow8[i-1] * 8
    
    # Map window code to permutation index
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
    
    # Initialize counts
    counts = [0] * num_perms
    
    # Count initial windows
    L = len(current_word)
    num_windows = L - n + 1
    for i in range(num_windows):
        wc = window_code(current_word, i)
        pidx = code_to_pidx.get(wc, -1)
        if pidx >= 0:
            counts[pidx] += 1
            
    # Check if valid initially (it should be)
    # If not, we have a bug in construction.
    
    # Greedy deletion with multiple strategies
    # Strategy 1: Backward pass, delete as many as possible per scan
    for pass_num in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        L = len(current_word)
        num_windows = L - n + 1
        
        # Rebuild window codes for this pass
        wcodes = [0] * num_windows
        for i in range(num_windows):
            wcodes[i] = window_code(current_word, i)
        
        # Iterate backwards
        for i in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
            
            start_j = max(0, i - n + 1)
            end_j = min(i, num_windows - 1)
            
            if start_j > end_j:
                continue
            
            # Check validity: no affected window is the last of its perm
            valid = True
            affected = []
            for j in range(start_j, end_j + 1):
                pidx = code_to_pidx.get(wcodes[j], -1)
                if pidx >= 0:
                    if counts[pidx] == 1:
                        valid = False
                        break
                    affected.append(pidx)
            
            if valid:
                for pidx in affected:
                    counts[pidx] -= 1
                current_word.pop(i)
                improved = True
                break  # Need to rebuild wcodes
        
        if not improved:
            break
    
    # Strategy 2: Forward pass
    for pass_num in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        L = len(current_word)
        num_windows = L - n + 1
        
        wcodes = [0] * num_windows
        for i in range(num_windows):
            wcodes[i] = window_code(current_word, i)
        
        for i in range(L):
            if time.time() > deadline:
                break
            
            start_j = max(0, i - n + 1)
            end_j = min(i, num_windows - 1)
            
            if start_j > end_j:
                continue
            
            valid = True
            affected = []
            for j in range(start_j, end_j + 1):
                pidx = code_to_pidx.get(wcodes[j], -1)
                if pidx >= 0:
                    if counts[pidx] == 1:
                        valid = False
                        break
                    affected.append(pidx)
            
            if valid:
                for pidx in affected:
                    counts[pidx] -= 1
                current_word.pop(i)
                improved = True
                break
        
        if not improved:
            break
    
    # Strategy 3: Random sampling for additional deletions
    for pass_num in range(10):
        if time.time() > deadline:
            break
        
        improved = False
        L = len(current_word)
        
        # Rebuild counts
        counts = [0] * num_perms
        num_windows = L - n + 1
        for i in range(num_windows):
            wc = window_code(current_word, i)
            pidx = code_to_pidx.get(wc, -1)
            if pidx >= 0:
                counts[pidx] += 1
        
        positions = list(range(L))
        random.shuffle(positions)
        
        for i in positions:
            if time.time() > deadline:
                break
            if i >= len(current_word):
                continue
            
            L = len(current_word)
            num_windows = L - n + 1
            
            start_j = max(0, i - n + 1)
            end_j = min(i, num_windows - 1)
            
            if start_j > end_j:
                continue
            
            valid = True
            affected = []
            for j in range(start_j, end_j + 1):
                wc = window_code(current_word, j)
                pidx = code_to_pidx.get(wc, -1)
                if pidx >= 0:
                    if counts[pidx] == 1:
                        valid = False
                        break
                    affected.append(pidx)
            
            if valid:
                for pidx in affected:
                    counts[pidx] -= 1
                current_word.pop(i)
                improved = True
        
        if not improved:
            break
    
    return "".join(map(str, current_word))
