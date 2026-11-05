import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Helper: Check if a sequence is valid (all consecutive block sums distinct)
    def is_valid(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Helper: Fast check for appending x to the end of seq
    # Returns (is_valid, new_sums_list)
    def check_append(seq, x):
        # New sums are suffixes of seq + x
        # i.e., x, seq[-1]+x, seq[-2]+seq[-1]+x, ...
        new_sums = []
        total = x
        new_sums.append(total)
        for a in reversed(seq):
            total += a
            new_sums.append(total)
            
        # We don't have the full set of sums here for fast check unless we pass it
        # For the greedy loop, we maintain the set.
        # This function is just for clarity, the actual loop will do the check inline.
        pass

    best_seq = []
    best_len = 0
    
    # 1. Optimized Greedy Forward
    seq = []
    sums = set()
    for x in range(1, 257):
        # Check if x can be appended
        tails, total = [x], x
        for a in reversed(seq):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            seq.append(x)
            sums.update(tails)
    if len(seq) > best_len:
        best_seq = seq
        best_len = len(seq)

    # 2. Optimized Greedy Reverse
    seq = []
    sums = set()
    for x in range(256, 0, -1):
        # x is new first element
        # New sums are prefixes of x + seq
        # i.e., x, x+seq[0], x+seq[0]+seq[1], ...
        heads, total = [x], x
        for a in seq:
            total += a
            heads.append(total)
        if not sums.intersection(heads):
            seq.insert(0, x)
            sums.update(heads)
    if len(seq) > best_len:
        best_seq = seq
        best_len = len(seq)

    # 3. Randomized Restarts with Greedy Append
    # This explores different orderings of appending
    rng = random.Random(42)
    
    # We will run this loop until deadline
    # Each iteration is fast: O(256 * N)
    while time.time() < deadline - 10.0: # Leave 10s for local search
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        seq = []
        sums = set()
        for x in candidates:
            # Try to append x
            tails, total = [x], x
            for a in reversed(seq):
                total += a
                tails.append(total)
            
            if not sums.intersection(tails):
                seq.append(x)
                sums.update(tails)
        
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)

    # 4. Local Search / Polish
    # Try to improve the best_seq by inserting remaining numbers
    # This is slower, so we do it in the remaining time
    if best_seq:
        current_seq = best_seq[:]
        available = set(range(1, 257)) - set(current_seq)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to insert each available number
            # Sort available to try smaller ones first? Or random?
            # Random to avoid bias
            avail_list = list(available)
            rng.shuffle(avail_list)
            
            for x in avail_list:
                if time.time() > deadline:
                    break
                    
                # Try to insert x at every position
                n = len(current_seq)
                # Positions 0 to n
                for pos in range(n + 1):
                    # Create new sequence
                    new_seq = current_seq[:pos] + [x] + current_seq[pos:]
                    
                    # Check validity
                    if is_valid(new_seq):
                        current_seq = new_seq
                        available.remove(x)
                        improved = True
                        break # Restart loop with new available set
                    
        if len(current_seq) > best_len:
            best_seq = current_seq
            best_len = len(current_seq)

    return best_seq

# To ensure the function is accessible
if __name__ == "__main__":
    res = construct()
    print(len(res))
