import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    full_mask = (1 << total_words) - 1
    
    covers = [0] * total_words
    
    def idx_to_list(idx):
        res = []
        for _ in range(n):
            res.append(idx % q)
            idx //= q
        return res
    
    def list_to_idx(lst):
        res = 0
        for i in range(n):
            res = res * q + lst[i]
        return res

    for c_idx in range(total_words):
        c = idx_to_list(c_idx)
        mask = 1 << c_idx
        for i in range(n):
            orig = c[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                c[i] = new_val
                w_idx = list_to_idx(c)
                mask |= (1 << w_idx)
            c[i] = orig
        covers[c_idx] = mask

    code = []
    covered_mask = 0
    
    while covered_mask != full_mask and time.time() < deadline:
        best_c = -1
        best_gain = 0
        for c in range(total_words):
            gain = (covers[c] & ~covered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_c = c
                if gain == 0:
                    break
        if best_c == -1:
            break
        code.append(best_c)
        covered_mask |= covers[best_c]

    if time.time() > deadline:
        return [idx_to_list(c) for c in code]

    def covers_all(code_list):
        u = 0
        for c in code_list:
            u |= covers[c]
        return u == full_mask

    def is_redundant(c, code_list):
        u = 0
        for c2 in code_list:
            if c2 != c:
                u |= covers[c2]
        return u == full_mask

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code)):
            if is_redundant(code[i], code):
                code.pop(i)
                changed = True
                break

    if time.time() > deadline:
        return [idx_to_list(c) for c in code]

    all_indices = list(range(total_words))
    random.seed(42)
    
    while time.time() < deadline:
        red_found = False
        for i in range(len(code)):
            if is_redundant(code[i], code):
                code.pop(i)
                red_found = True
                break
        
        if red_found:
            continue
            
        if not code:
            break
            
        c_out_idx = random.randint(0, len(code) - 1)
        c_out = code[c_out_idx]
        
        c_in = random.choice(all_indices)
        
        if c_in == c_out or c_in in code:
            continue
            
        code[c_out_idx] = c_in
        
        if covers_all(code):
            continue
        else:
            code[c_out_idx] = c_out

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code)):
            if is_redundant(code[i], code):
                code.pop(i)
                changed = True
                break

    return [idx_to_list(c) for c in code]
