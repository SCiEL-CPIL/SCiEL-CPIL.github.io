import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def is_time_up():
        return time.time() > deadline

    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails = []
            total = 0
            for a in reversed(values + [x]):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums

    base_seq, base_sums = greedy()
    best_seq = base_seq
    base_len = len(base_seq)
    
    global_timeout_flag = False
    current_seq = []
    current_sums = set()

    def dfs(last, target_len):
        nonlocal global_timeout_flag, current_seq, current_sums
        
        if global_timeout_flag:
            return False
        
        if is_time_up():
            global_timeout_flag = True
            return False
            
        if len(current_seq) == target_len:
            return True
        
        remaining_slots = target_len - len(current_seq)
        available_numbers = 256 - last
        if available_numbers < remaining_slots:
            return False
            
        for x in range(last + 1, 257):
            if x in current_sums:
                continue

            new_sums = []
            total = x
            new_sums.append(total)
            
            valid = True
            for a in reversed(current_seq):
                total += a
                new_sums.append(total)
                if total in current_sums:
                    valid = False
                    break
            
            if not valid:
                continue
                
            current_seq.append(x)
            for s in new_sums:
                current_sums.add(s)
                
            if dfs(x, target_len):
                return True
                
            current_seq.pop()
            for s in new_sums:
                current_sums.remove(s)
                
        return False

    for target_len in range(base_len + 1, base_len + 20):
        if global_timeout_flag:
            break
            
        current_seq = []
        current_sums = set()
        
        if dfs(0, target_len):
            best_seq = current_seq[:]
        else:
            if global_timeout_flag:
                break
            else:
                break

    return best_seq
