import sys
import time

sys.setrecursionlimit(10000)

def construct():
    # 1. Generate a baseline using the greedy algorithm
    # Using bitset for sums for speed
    values = []
    p_sums = [0]
    diff_mask = 0
    
    for x in range(1, 257):
        valid = True
        new_diffs = []
        last_p = p_sums[-1]
        for i in range(len(p_sums)):
            d = last_p - p_sums[i]
            nd = x + d
            if (diff_mask >> nd) & 1:
                valid = False
                break
            new_diffs.append(nd)
        
        if valid:
            values.append(x)
            p_sums.append(last_p + x)
            for nd in new_diffs:
                diff_mask |= (1 << nd)
                
    best_seq = values[:]
    best_len = len(values)
    
    # 2. DFS to find longer sequences
    start_time = time.time()
    time_limit = 55.0
    nodes = 0
    node_limit = 1000000
    
    def dfs(seq, p_sums, diff_mask, next_min_val):
        nonlocal nodes, best_seq, best_len
        
        if time.time() - start_time > time_limit:
            return
        if nodes > node_limit:
            return
        nodes += 1
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        if next_min_val > 256:
            return
            
        last_p = p_sums[-1]
        
        for x in range(next_min_val, 257):
            valid = True
            new_diffs = []
            for i in range(len(p_sums)):
                d = last_p - p_sums[i]
                nd = x + d
                if (diff_mask >> nd) & 1:
                    valid = False
                    break
                new_diffs.append(nd)
                
            if not valid:
                continue
                
            new_p_sum = last_p + x
            new_p_sums = p_sums + [new_p_sum]
            new_mask = diff_mask
            for nd in new_diffs:
                new_mask |= (1 << nd)
                
            dfs(seq + [x], new_p_sums, new_mask, x + 1)
            
            if time.time() - start_time > time_limit:
                return

    dfs([], [0], 0, 1)
    
    return best_seq
