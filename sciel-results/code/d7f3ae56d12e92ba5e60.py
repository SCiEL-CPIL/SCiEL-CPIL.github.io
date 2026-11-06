import time

def construct():
    start_time = time.time()
    best_length = 0
    best_seq = []
    
    def dfs(idx, current_len, suffix_prods, prods_set, current_seq):
        nonlocal best_length, best_seq
        
        if time.time() - start_time > 85:
            return
            
        remaining = 128 - idx + 1
        if current_len + remaining <= best_length:
            return
            
        if idx > 128:
            if current_len > best_length:
                best_length = current_len
                best_seq = list(current_seq)
            return

        x = idx
        
        # 1. Try Include
        if x not in prods_set:
            new_suffixes = [x]
            valid = True
            for s in suffix_prods:
                p = s * x
                new_suffixes.append(p)
                if p in prods_set:
                    valid = False
                    break
            
            if valid:
                prods_set.update(new_suffixes)
                current_seq.append(x)
                
                dfs(idx + 1, current_len + 1, new_suffixes, prods_set, current_seq)
                
                current_seq.pop()
                for p in new_suffixes:
                    prods_set.remove(p)
        
        # 2. Try Exclude
        dfs(idx + 1, current_len, suffix_prods, prods_set, current_seq)

    dfs(2, 0, [], set(), [])
    
    return best_seq
