import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_len = 0
    best_seq = []
    current_path = []
    
    # Greedy initial solution
    def greedy():
        seq = []
        suffix_products = set()
        all_products = set()
        for x in range(2, 129):
            new_suffixes = {p * x for p in suffix_products}
            new_suffixes.add(x)
            if not (new_suffixes & all_products):
                seq.append(x)
                suffix_products = new_suffixes
                all_products |= new_suffixes
        return seq

    init_seq = greedy()
    best_len = len(init_seq)
    best_seq = init_seq
    
    def dfs(index, current_len, suffix_products, all_products):
        nonlocal best_len, best_seq, current_path
        
        if time.time() > deadline:
            return
            
        if current_len + (128 - index + 1) <= best_len:
            return
            
        if index > 128:
            if current_len > best_len:
                best_len = current_len
                best_seq = list(current_path)
            return

        # Option 1: Include index
        x = index
        new_suffixes = set()
        for p in suffix_products:
            new_suffixes.add(p * x)
        new_suffixes.add(x)
        
        if not (new_suffixes & all_products):
            current_path.append(x)
            new_all = all_products | new_suffixes
            dfs(index + 1, current_len + 1, new_suffixes, new_all)
            current_path.pop()
            
        # Option 2: Exclude index
        dfs(index + 1, current_len, suffix_products, all_products)

    dfs(2, 0, set(), set())
    
    return best_seq
