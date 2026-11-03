import random
import time

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        total_words = q ** n
        neighbors = [[] for _ in range(total_words)]
        for w in range(total_words):
            nbrs = [w]
            for pos in range(n):
                div = q ** (n - 1 - pos)
                curr_digit = (w // div) % q
                for new_digit in range(q):
                    if new_digit != curr_digit:
                        new_w = w + (new_digit - curr_digit) * div
                        nbrs.append(new_w)
            neighbors[w] = nbrs
        
        covered = [False] * total_words
        code = []
        while not all(covered):
            best_count = -1
            best_w = -1
            for w in range(total_words):
                if w in code: continue
                count = sum(1 for nbr in neighbors[w] if not covered[nbr])
                if count > best_count:
                    best_count = count
                    best_w = w
            if best_w == -1: break
            code.append(best_w)
            for nbr in neighbors[best_w]:
                covered[nbr] = True
        return [list(map(int, [w // (q ** (n - 1 - i)) % q for i in range(n)])) for w in code]

    # Optimized path for q=3, n=6, radius=1
    total_words = 729
    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        nbrs = [w]
        for pos in range(n):
            div = 3 ** (n - 1 - pos)
            curr_digit = (w // div) % q
            for new_digit in range(q):
                if new_digit != curr_digit:
                    new_w = w + (new_digit - curr_digit) * div
                    nbrs.append(new_w)
        neighbors[w] = nbrs

    all_words = list(range(total_words))
    rng = random.Random(123)
    
    # 1. Greedy Initialization
    cover_count = [0] * total_words
    covered = [False] * total_words
    code_set = set()
    
    while not all(covered):
        best_count = -1
        best_candidates = []
        for w in all_words:
            if w in code_set:
                continue
            count = 0
            for nbr in neighbors[w]:
                if not covered[nbr]:
                    count += 1
            if count > best_count:
                best_count = count
                best_candidates = [w]
            elif count == best_count and count > 0:
                best_candidates.append(w)
        
        if best_count == 0:
            break
            
        selected = rng.choice(best_candidates)
        code_set.add(selected)
        for nbr in neighbors[selected]:
            cover_count[nbr] += 1
            if cover_count[nbr] == 1:
                covered[nbr] = True
                
    # 2. Local Search
    best_code = list(code_set)
    best_len = len(best_code)
    start_time = time.time()
    max_time = 100
    stuck_count = 0
    
    while time.time() - start_time < max_time:
        # Convert set to list for random access
        current_list = list(code_set)
        if not current_list:
            break
            
        idx = rng.randrange(len(current_list))
        c = current_list[idx]
        
        # Check if c is critical
        is_critical = False
        for nbr in neighbors[c]:
            if cover_count[nbr] == 1:
                is_critical = True
                break
                
        if not is_critical:
            # Remove c
            code_set.remove(c)
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
            stuck_count = 0
            
            if len(code_set) < best_len:
                best_len = len(code_set)
                best_code = list(code_set)
        else:
            stuck_count += 1
            if stuck_count > 50:
                # Random Swap to escape
                # Pick random d not in code_set
                d = rng.choice(all_words)
                while d in code_set:
                    d = rng.choice(all_words)
                
                # Remove c
                code_set.remove(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                    if cover_count[nbr] == 0:
                        covered[nbr] = False
                        
                # Add d
                code_set.add(d)
                for nbr in neighbors[d]:
                    cover_count[nbr] += 1
                    if cover_count[nbr] == 1:
                        covered[nbr] = True
                        
                stuck_count = 0

    # Convert best_code to final format
    final_code = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n-1, -1, -1):
            word[i] = temp % q
            temp //= q
        final_code.append(word)
        
    return final_code
