import time

def construct():
    """
    Finds a maximum size Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses a bitset-based backtracking search for speed.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    required = {1, 2, 5, 11}
    max_val = 284
    target_size = 20
    
    # No hardcoded ruler to avoid invalidity risk. Rely on fast search.
    
    current_set = set(required)
    used_sums = 0
    req_list = sorted(required)
    for i in range(len(req_list)):
        for j in range(i, len(req_list)):
            s = req_list[i] + req_list[j]
            used_sums |= (1 << s)
            
    best_solution = list(required)
    best_size = len(best_solution)
    
    pool = [x for x in range(1, max_val + 1) if x not in required]
    
    def can_add(x, current_set, used_sums):
        # Check 2*x
        if (used_sums >> (2 * x)) & 1:
            return False
        for y in current_set:
            if y == x: continue
            s = x + y
            if (used_sums >> s) & 1:
                return False
        return True

    def get_new_sums(x, current_set):
        bits = 1 << (2 * x)
        for y in current_set:
            if y != x:
                bits |= (1 << (x + y))
        return bits

    def backtrack(current_set, used_sums, pool_index, depth):
        nonlocal best_solution, best_size
        
        current_size = len(current_set)
        if current_size > best_size:
            best_solution = list(current_set)
            best_size = current_size
            
        if current_size == target_size:
            return True
            
        if time.time() > deadline:
            return False
            
        if pool_index >= len(pool):
            return False
            
        need = target_size - current_size
        available = len(pool) - pool_index
        if available < need:
            return False
            
        for i in range(pool_index, len(pool)):
            x = pool[i]
            if not can_add(x, current_set, used_sums):
                continue
                
            current_set.add(x)
            new_bits = get_new_sums(x, current_set)
            used_sums |= new_bits
            
            if backtrack(current_set, used_sums, i + 1, depth + 1):
                return True
                
            current_set.remove(x)
            used_sums &= ~new_bits
                
        return False

    backtrack(current_set, used_sums, 0, 0)
    
    return sorted(best_solution)
