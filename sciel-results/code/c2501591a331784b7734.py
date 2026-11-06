import time
import random
import sys

def construct():
    start_time = time.time()
    TIMEOUT = 58.0  # Safety margin
    
    # Heuristic to get a good starting point
    def greedy_heuristic():
        best = []
        best_sums = set()
        best_tails = []
        
        # Try multiple random permutations or just a simple greedy with backtracking limit
        # Simple greedy: pick smallest valid x
        seq = []
        sums = set()
        tails = []
        
        for x in range(1, 257):
            if not seq:
                new_tails = [x]
            else:
                new_tails = [x + t for t in [0] + tails] # Wait, tails are suffixes.
                # If seq = [a, b], tails = [b, a+b].
                # Append x.
                # New suffixes: x, x+b, x+b+a.
                # Formula: x + 0, x + b, x + (a+b).
                # So new_tails = [x + t for t in [0] + tails] is correct if tails is ordered [last, last+prev, ...]
                # My previous code: tails = [b, a+b].
                # [0] + tails = [0, b, a+b].
                # x + [0, b, a+b] = [x, x+b, x+a+b]. Correct.
            
            if not sums.intersection(new_tails):
                seq.append(x)
                sums.update(new_tails)
                tails = new_tails
        
        return seq

    # Better Heuristic: Randomized Greedy
    def randomized_greedy():
        best_seq = []
        for _ in range(100): # 100 restarts
            seq = []
            sums = set()
            tails = []
            available = list(range(1, 257))
            random.shuffle(available)
            
            # We must maintain increasing order. So we can't just shuffle.
            # We pick a start, then pick next from remaining > last.
            # Let's pick a random start, then greedily pick next valid.
            start_idx = random.randint(0, 250)
            x = available[start_idx]
            seq = [x]
            sums = {x}
            tails = [x]
            
            for y in available:
                if y <= seq[-1]:
                    continue
                new_tails = [y + t for t in [0] + tails]
                if not sums.intersection(new_tails):
                    seq.append(y)
                    sums.update(new_tails)
                    tails = new_tails
            
            if len(seq) > len(best_seq):
                best_seq = seq
        
        return best_seq

    # Initial best
    best_seq = randomized_greedy()
    best_len = len(best_seq)
    
    # DFS to improve
    # State: seq, sums, tails
    # We want to find a sequence longer than best_len
    
    def dfs(seq, sums, tails, depth):
        nonlocal best_seq, best_len, start_time, TIMEOUT
        
        if time.time() - start_time > TIMEOUT:
            return False
        
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)
            
        # Pruning
        if seq:
            # Max possible extension
            # We can add at most 256 - seq[-1] numbers.
            if len(seq) + (256 - seq[-1]) <= best_len:
                return False
            last_val = seq[-1]
        else:
            last_val = 0
            if 256 <= best_len:
                return False

        # Candidate generation
        # Try values from last_val + 1 to 256
        # Order: We can try to prioritize values that are "far" from existing sums?
        # Or just iterate.
        
        for x in range(last_val + 1, 257):
            # Calculate new tails
            new_tails = [x + t for t in [0] + tails]
            
            # Check collision
            # Optimization: Check intersection
            if any(s in sums for s in new_tails):
                continue
                
            # Valid move
            # Create new sums set
            # To save time, we can pass the set by reference and undo?
            # But sets are mutable. 
            # new_sums = sums | set(new_tails)
            
            seq.append(x)
            # We need to pass the updated sums.
            # Creating a new set is O(N). N is small.
            new_sums = sums.union(new_tails)
            
            dfs(seq, new_sums, new_tails, depth + 1)
            seq.pop()
            
            if time.time() - start_time > TIMEOUT:
                return True

        return False

    # Start DFS from the best sequence found by heuristic?
    # Or start from scratch?
    # Starting from the heuristic result might be good to extend it.
    # But the heuristic result might not be extendable easily.
    # Let's start DFS from empty sequence, but use the best_len to prune.
    
    # To make DFS more effective, we can try to extend the current best sequence first?
    # No, the best sequence might be a dead end.
    # Let's run DFS from empty.
    
    # Optimization: If best_len is already high, we can restrict the search space.
    # Also, we can use a "target" length.
    # Let's just run the DFS.
    
    # To avoid re-exploring the same states, we could use memoization, but the state is (last_val, sums, tails) which is huge.
    # So simple DFS is best.
    
    # One improvement: Try to extend the *heuristic* solution first.
    # If we can extend it, we update best_len, which prunes more of the DFS from scratch.
    
    # Let's try extending the best_seq from randomized_greedy
    # We need the sums and tails for best_seq.
    
    def get_sums_tails(seq):
        sums = set()
        tails = []
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                sums.add(s)
        # Calculate tails
        t = 0
        for x in reversed(seq):
            t += x
            tails.append(t)
        return sums, tails

    if best_seq:
        best_sums, best_tails = get_sums_tails(best_seq)
        # Try to extend this specific sequence
        # We can't easily "pop" from the original list if we mutated it? 
        # We passed a copy? No, randomized_greedy returns a list.
        # Let's create a copy for DFS extension.
        seq_copy = list(best_seq)
        dfs(seq_copy, best_sums, best_tails, len(best_seq))
        
    # If time remains, do a general DFS from start to see if we missed something
    if time.time() - start_time < TIMEOUT - 5:
        dfs([], set(), [], 0)

    return best_seq
