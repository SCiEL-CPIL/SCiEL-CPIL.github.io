import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Step 1: Generate S_6 using standard construction
    # This gives us the set of permutations of 1..6
    # We need the list of all 720 permutations
    # We can just generate all permutations of range(1, 7)
    
    if n <= 6:
        # Use standard construction for small n
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find overlap
                max_overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_overlap = o
                        break
                new += block[max_overlap:]
            word = new
        return "".join(map(str, word))
        
    # For n >= 7, use the optimized final step
    # Generate S_6
    word = [1]
    for m in range(2, 7):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
        
    # S_6 is in 'word'
    # Extract the 720 permutations of 1..6 in order of appearance
    # Actually, we just need the set of all permutations.
    # The set is fixed: all permutations of 1..6.
    perms_6 = list(permutations(range(1, 7)))
    num_perms = len(perms_6)
    
    # Precompute overlap matrix
    # M[i][j] = overlap(perms_6[i], perms_6[j])
    # Overlap is longest suffix of i that is prefix of j
    M = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p_i = perms_6[i]
        for j in range(num_perms):
            p_j = perms_6[j]
            # Check overlaps from 5 down to 0
            max_ov = 0
            for k in range(5, -1, -1):
                if p_i[6-k:] == p_j[:k]:
                    max_ov = k
                    break
            M[i][j] = max_ov
            
    # Greedy search with restarts
    best_order = None
    best_score = -1
    
    # Standard order from S_6
    # We can use this as one of the candidates
    # But let's just run random restarts
    
    max_restarts = 5000
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        visited = [False] * num_perms
        order = []
        
        # Start with a random permutation
        start_idx = random.randint(0, num_perms - 1)
        current = start_idx
        visited[current] = True
        order.append(current)
        
        for _ in range(num_perms - 1):
            max_ov = -1
            next_idx = -1
            # Find best next
            for j in range(num_perms):
                if not visited[j]:
                    ov = M[current][j]
                    if ov > max_ov:
                        max_ov = ov
                        next_idx = j
                        # Small random tie-breaker
                        # If ov == max_ov, we could pick randomly among ties
                        # But for speed, we just pick the first max
            if next_idx == -1:
                # Should not happen
                break
            visited[next_idx] = True
            order.append(next_idx)
            current = next_idx
            
        # Calculate score (total overlap)
        score = 0
        for i in range(len(order) - 1):
            score += M[order[i]][order[i+1]]
            
        if score > best_score:
            best_score = score
            best_order = order
            # Optional: if we find a very good score, we can stop early?
            # Max possible overlap is 5 * 719 = 3595
            # Standard construction likely has high overlaps.
            # Let's keep searching.
            
    # Construct S_7 from best_order
    if best_order is None:
        # Fallback to standard construction
        return "".join(map(str, word)) # This is S_6, invalid for n=7
        
    new_word = []
    # Add the first block
    p_first = perms_6[best_order[0]]
    block = list(p_first) + [7] + list(p_first)
    new_word = block
    
    for idx in best_order[1:]:
        p_next = perms_6[idx]
        block = list(p_next) + [7] + list(p_next)
        
        # Find overlap
        max_ov = 0
        # Max possible overlap is 6 (suffix of new_word, prefix of block)
        # But block starts with p_next (len 6)
        # So max overlap is 6
        for o in range(min(len(new_word), 6), -1, -1):
            if new_word[len(new_word) - o:] == block[:o]:
                max_ov = o
                break
        new_word += block[max_ov:]
        
    return "".join(map(str, new_word))
