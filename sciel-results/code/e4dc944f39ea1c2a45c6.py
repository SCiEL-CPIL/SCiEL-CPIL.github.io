import sys
import time

sys.setrecursionlimit(10000)

# Global variables for backtracking
best_sequence = []
best_length = 0
start_time = 0.0
TIME_LIMIT = 55.0  # Leave some buffer for overhead

def construct():
    global best_sequence, best_length, start_time
    start_time = time.time()
    best_sequence = []
    best_length = 0
    
    # First, run the greedy algorithm to get a baseline
    greedy_seq = greedy_construct()
    best_sequence = greedy_seq
    best_length = len(greedy_seq)
    
    # Now, try to find a longer sequence using backtracking
    def backtrack_optimized(seq, prefix_sums, diffs):
        global best_sequence, best_length
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        current_len = len(seq)
        
        if current_len > best_length:
            best_sequence = seq.copy()
            best_length = current_len
            
        last_val = seq[-1] if seq else 0
        last_prefix = prefix_sums[-1]
        
        # Pruning: Max possible length
        # We can add at most 256 - last_val more elements
        if current_len + (256 - last_val) <= best_length:
            return
            
        for a_next in range(last_val + 1, 257):
            if time.time() - start_time > TIME_LIMIT:
                return
                
            new_prefix = last_prefix + a_next
            valid = True
            new_diffs = []
            
            for i in range(len(prefix_sums)):
                diff = new_prefix - prefix_sums[i]
                if diff in diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
                
            # Convert to set for union
            new_diffs_set = set(new_diffs)
            all_diffs = diffs.union(new_diffs_set)
            
            backtrack_optimized(seq + [a_next], prefix_sums + [new_prefix], all_diffs)
            
    # Start backtracking
    for a1 in range(1, 257):
        # Prune if even the max possible length starting with a1 is not better
        if best_length >= 256 - a1 + 1:
            break
            
        if time.time() - start_time > TIME_LIMIT:
            break
            
        backtrack_optimized([a1], [a1], {a1})
        
    return best_sequence

def greedy_construct():
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values
