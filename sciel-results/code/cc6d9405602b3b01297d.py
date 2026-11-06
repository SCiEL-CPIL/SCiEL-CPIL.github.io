import sys
import time

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    best_seq = []
    MAX_DIFF = 5000
    
    def dfs(last_a, last_P, diffs_arr, seq, prefix_sums):
        nonlocal best_seq, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            
        # Pruning: Max possible length
        # We can add at most (256 - last_a) elements
        if len(seq) + (256 - last_a) <= len(best_seq):
            return
            
        for a_next in range(last_a + 1, 257):
            P_next = last_P + a_next
            
            if P_next > MAX_DIFF:
                break
                
            valid = True
            for i in range(len(prefix_sums)):
                d = P_next - prefix_sums[i]
                if d > MAX_DIFF:
                    continue
                if diffs_arr[d]:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            new_diffs = []
            for i in range(len(prefix_sums)):
                d = P_next - prefix_sums[i]
                new_diffs.append(d)
                diffs_arr[d] = True
                
            seq.append(a_next)
            prefix_sums.append(P_next)
            
            dfs(a_next, P_next, diffs_arr, seq, prefix_sums)
            
            seq.pop()
            prefix_sums.pop()
            for d in new_diffs:
                diffs_arr[d] = False
                
            if time.time() - start_time > time_limit:
                return

    seq = [1]
    prefix_sums = [0, 1]
    diffs_arr = [False] * (MAX_DIFF + 1)
    diffs_arr[1] = True
    
    dfs(1, 1, diffs_arr, seq, prefix_sums)
    
    return best_seq
