import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Standard construction
        word = [1]
        for m in range(2, n + 1):
            if time.time() > deadline:
                break
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
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
        return "".join(map(str, word))

    # For n >= 5, use the optimization strategy
    # 1. Build W_{n-1} standard
    # 2. Extract perms of n-1
    # 3. Optimize order of perms for inserting n
    
    # Build W_{n-1}
    word = [1]
    for m in range(2, n):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
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
        
    # Extract perms of n-1 from word
    perms, seen = [], set()
    m_prev = n - 1
    for i in range(len(word) - m_prev + 1):
        window = tuple(word[i:i + m_prev])
        if len(set(window)) == m_prev and window not in seen:
            seen.add(window)
            perms.append(window)
            
    num_perms = len(perms)
    
    # Compute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    max_k = m_prev - 1 # Max possible overlap is n-2
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            # Check overlaps
            # pi suffix vs pj prefix
            # Max possible overlap is m_prev - 1
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap_matrix[i][j] = 0
                    break
                if pi[-k:] == pj[:k]:
                    overlap_matrix[i][j] = k
                    break
            if time.time() > deadline:
                # Fallback to natural order if too slow
                # This is unlikely for n=7 (720 perms)
                break
        if time.time() > deadline:
            break

    # Initial order: natural order
    order = list(range(num_perms))
    
    # Calculate current total overlap
    def calc_total_overlap(ord_list):
        total = 0
        for i in range(len(ord_list)):
            u = ord_list[i]
            v = ord_list[(i + 1) % len(ord_list)]
            total += overlap_matrix[u][v]
        return total

    best_overlap = calc_total_overlap(order)
    best_order = order[:]
    
    # Local Search: 2-opt and Or-opt
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 2-opt
        for i in range(num_perms - 1):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, num_perms):
                # Reverse segment i..j
                # Calculate delta
                # Current: ... A - B - C ... (where A is before i, B is segment, C is after j)
                # Actually, it's a cycle.
                # Let's just recalculate or use delta.
                # Delta is complex for cycle. Recalculating is O(N).
                # 720 * 720 / 2 * 720 is too slow.
                # We need O(1) delta.
                
                # Let u = order[i-1], v = order[i], w = order[j], x = order[j+1]
                # Current edges: (u,v) and (w,x)
                # New edges: (u,w) and (v,x)
                # Delta = overlap_matrix[u][w] + overlap_matrix[v][x] - overlap_matrix[u][v] - overlap_matrix[w][x]
                
                u_idx = (i - 1) % num_perms
                v_idx = i
                w_idx = j
                x_idx = (j + 1) % num_perms
                
                u = best_order[u_idx]
                v = best_order[v_idx]
                w = best_order[w_idx]
                x = best_order[x_idx]
                
                delta = (overlap_matrix[u][w] + overlap_matrix[v][x]) - (overlap_matrix[u][v] + overlap_matrix[w][x])
                
                if delta > 0:
                    # Reverse segment i..j
                    best_order[i:j+1] = reversed(best_order[i:j+1])
                    best_overlap += delta
                    improved = True
                    # Don't break, keep searching? Or restart?
                    # Usually restart or continue. Continuing is faster.
                    
        # Or-opt (move a segment of length 1, 2, or 3)
        # Move segment [i..i+L-1] to position j
        # This is more complex to calculate delta.
        # Given time constraints, 2-opt might be sufficient or we can add simple 1-opt (swap)
        
        # 1-opt (swap adjacent or any two)
        # Swap is a special case of 2-opt? No.
        # Let's add a pass for swapping any two elements.
        for i in range(num_perms):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, num_perms):
                # Swap i and j
                # If i and j are adjacent, it's covered by 2-opt?
                # 2-opt reverses a segment. Reversing a segment of length 1 is a no-op.
                # Reversing a segment of length 2 is a swap.
                # So 2-opt covers swaps.
                pass

    # Construct the final word
    new_word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(new_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0:
                overlap = 0
                break
            if new_word[len(new_word) - o:] == block[:o]:
                overlap = o
                break
        new_word += block[overlap:]
        
    return "".join(map(str, new_word))
