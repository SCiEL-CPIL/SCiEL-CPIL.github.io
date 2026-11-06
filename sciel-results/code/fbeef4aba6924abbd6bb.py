import random
import time

def construct():
    n = 4096
    
    def can_add(x, current_set, existing_sums):
        new_sums = []
        s = 3 * x
        if s in existing_sums:
            return False, []
        new_sums.append(s)
        
        for y in current_set:
            s = 2 * x + y
            if s in existing_sums:
                return False, []
            new_sums.append(s)
            
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                y = current_set[i]
                z = current_set[j]
                s = x + y + z
                if s in existing_sums:
                    return False, []
                new_sums.append(s)
        return True, new_sums

    def greedy_backtrack(target_size, time_limit=55):
        start_time = time.time()
        current_set = []
        existing_sums = set()
        used = set()
        
        def dfs(depth):
            if time.time() - start_time > time_limit:
                return None
            
            if depth == target_size:
                return current_set.copy()
            
            # Heuristic: Try larger numbers first to space them out
            # We iterate from n down to 1
            # To avoid O(n) check for every node, we can just iterate.
            # n=4096. Depth 80.
            # 4096 * 80 * 80^2 (can_add cost) = 4096 * 80 * 6400 = 2e9. Too slow.
            
            # We need to prune better.
            # Let's just use a random subset of candidates.
            
            candidates = [x for x in range(n, 0, -1) if x not in used]
            random.shuffle(candidates)
            
            # Limit candidates to speed up
            candidates = candidates[:100]
            
            for x in candidates:
                ok, new_sums = can_add(x, current_set, existing_sums)
                if ok:
                    current_set.append(x)
                    existing_sums.update(new_sums)
                    used.add(x)
                    
                    res = dfs(depth + 1)
                    if res:
                        return res
                    
                    current_set.pop()
                    existing_sums.difference_update(new_sums)
                    used.remove(x)
            
            return None
        
        return dfs(0)

    # Try to find a set of size 80
    res = greedy_backtrack(80)
    if res:
        return res
        
    # Fallback to original greedy
    best_set = []
    best_size = 0
    
    def run_greedy(order):
        current_set = []
        existing_sums = set()
        for x in order:
            ok, new_sums = can_add(x, current_set, existing_sums)
            if ok:
                current_set.append(x)
                existing_sums.update(new_sums)
        return current_set

    for _ in range(50):
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        res = run_greedy(candidates)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            
    return best_set
