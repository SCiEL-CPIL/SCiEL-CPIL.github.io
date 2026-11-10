import time
import sys

def construct():
    start_time = time.time()
    time_limit = 52.0
    
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    def is_valid(seq):
        n = len(seq)
        if n <= 1:
            return True
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
        sums = set()
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                s = prefix[j] - prefix[i]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    if not is_valid(best_seq):
        best_seq = [1]
    
    best_len = len(best_seq)
    
    def search_ruler(prefix, diffs, last_d):
        nonlocal best_seq, best_len, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        n = len(prefix)
        if n - 1 > best_len:
            seq = [prefix[i+1] - prefix[i] for i in range(n - 1)]
            best_seq = seq
            best_len = n - 1
        
        if n >= 11:
            return
        
        if last_d >= 256:
            return
        
        for d in range(last_d + 1, 257):
            if time.time() - start_time > time_limit:
                return
            
            new_mark = prefix[-1] + d
            
            ok = True
            for i in range(n):
                if (new_mark - prefix[i]) in diffs:
                    ok = False
                    break
            
            if not ok:
                continue
            
            prefix.append(new_mark)
            added = []
            for i in range(n):
                diff = new_mark - prefix[i]
                diffs.add(diff)
                added.append(diff)
            
            search_ruler(prefix, diffs, d)
            
            prefix.pop()
            for diff in added:
                diffs.discard(diff)
    
    prefix = [0]
    diffs = set()
    search_ruler(prefix, diffs, 0)
    
    return best_seq
