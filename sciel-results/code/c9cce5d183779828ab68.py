import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    best_seq = []
    best_len = 0
    
    # To speed up, we can pre-compute nothing, but use local vars
    # sys.setrecursionlimit(10000)
    
    def dfs(seq, sums, last_val, depth):
        nonlocal best_seq, best_len, deadline
        
        # Check time
        if time.time() > deadline:
            return
            
        if depth > best_len:
            best_len = depth
            best_seq = seq[:]
            
        # If we can't add any more, return
        # Max possible length is limited by range and sum constraints
        # Heuristic: if last_val is close to 256, we are near the end
        
        # Iterate possible next values
        # Optimization: We can limit the search space.
        # If we want to beat current best, we need to find a sequence of length > best_len.
        # But we don't know the target yet.
        
        for x in range(last_val + 1, 257):
            # Calculate new sums ending at x
            # sums is a bitset of all existing consecutive sums
            # New sums are: x, x+a_last, x+a_last+a_second_last, ...
            
            s = x
            # Check sum x
            if sums & (1 << s):
                continue
                
            new_sum_bits = 1 << s
            conflict = False
            
            # Add previous elements in reverse
            # seq is [a1, ..., a_k]
            # We need sums: x, x+a_k, x+a_k+a_{k-1}, ...
            
            # To avoid iterating reversed(seq) every time, we could maintain suffix sums?
            # But seq changes.
            # Given small depth, iterating is fine.
            
            for a in reversed(seq):
                s += a
                # Max sum is roughly 256 * 15 = 3840.
                # Bit shift is cheap.
                if sums & (1 << s):
                    conflict = True
                    break
                new_sum_bits |= (1 << s)
                
            if conflict:
                continue
                
            # Recurse
            # Pass a copy of seq?
            # seq.append(x); dfs(...); seq.pop() is faster than seq + [x]
            seq.append(x)
            dfs(seq, sums | new_sum_bits, x, depth + 1)
            seq.pop()
            
            if time.time() > deadline:
                return

    dfs([], 0, 0, 0)
    return best_seq
