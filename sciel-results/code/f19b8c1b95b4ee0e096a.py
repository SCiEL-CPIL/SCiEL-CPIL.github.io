import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin (evaluator kills at 90s)
    
    def is_valid(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True
    
    def check_insertion(seq, x, pos):
        """Check if inserting x at position pos keeps all block products distinct.
        Only needs to check blocks that include x, since other blocks are unchanged."""
        n = len(seq)
        # Collect all existing products
        existing_prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                existing_prods.add(p)
        
        # Now check all blocks that include the new element x at position pos
        # A block [a..b] in the new sequence includes x if a <= pos <= b
        # In terms of original indices: blocks starting at i (0..pos) and ending at j (pos..n-1)
        # where the new block is seq[i..pos-1] + [x] + seq[pos..j]
        for i in range(pos + 1):
            p = 1
            # Multiply elements before x
            for k in range(i, pos):
                p *= seq[k]
            p *= x
            # Multiply elements after x
            for k in range(pos, n):
                p *= seq[k]
                if p in existing_prods:
                    return False
                existing_prods.add(p)
        return True
    
    # 1. Greedy Initialization
    seq = []
    for x in range(2, 129):
        candidate = seq + [x]
        if is_valid(candidate):
            seq = candidate
    
    best_seq = list(seq)
    best_len = len(seq)
    
    # 2. Fixed-Point Insertion with fast check
    changed = True
    while changed and time.time() < DEADLINE - 5.0:
        changed = False
        excluded = [x for x in range(2, 129) if x not in seq]
        for x in excluded:
            if time.time() > DEADLINE - 5.0:
                break
            # Try all possible positions
            for pos in range(len(seq) + 1):
                # Check if position is valid (strictly increasing)
                if pos > 0 and seq[pos-1] >= x:
                    continue
                if pos < len(seq) and seq[pos] <= x:
                    continue
                
                if check_insertion(seq, x, pos):
                    seq = seq[:pos] + [x] + seq[pos:]
                    changed = True
                    if len(seq) > best_len:
                        best_len = len(seq)
                        best_seq = list(seq)
                    break  # Restart with new excluded set
    
    # 3. Try 1-out/1-in swaps to potentially unlock longer sequences
    if time.time() < DEADLINE - 10.0:
        improved = True
        while improved and time.time() < DEADLINE - 10.0:
            improved = False
            seq_set = set(seq)
            for i in range(len(seq)):
                if time.time() > DEADLINE - 10.0:
                    break
                for x in range(2, 129):
                    if x in seq_set:
                        continue
                    if time.time() > DEADLINE - 10.0:
                        break
                    # Try replacing seq[i] with x
                    if i > 0 and seq[i-1] >= x:
                        continue
                    if i < len(seq) - 1 and seq[i+1] <= x:
                        continue
                    
                    trial = list(seq)
                    trial[i] = x
                    if is_valid(trial):
                        # Check if this allows further insertions
                        seq = trial
                        seq_set = set(seq)
                        improved = True
                        if len(seq) > best_len:
                            best_len = len(seq)
                            best_seq = list(seq)
                        break
                if improved:
                    break
    
    # 4. Final fixed-point pass after local search
    if time.time() < DEADLINE - 3.0:
        changed = True
        while changed and time.time() < DEADLINE - 3.0:
            changed = False
            excluded = [x for x in range(2, 129) if x not in seq]
            for x in excluded:
                if time.time() > DEADLINE - 3.0:
                    break
                for pos in range(len(seq) + 1):
                    if pos > 0 and seq[pos-1] >= x:
                        continue
                    if pos < len(seq) and seq[pos] <= x:
                        continue
                    if check_insertion(seq, x, pos):
                        seq = seq[:pos] + [x] + seq[pos:]
                        changed = True
                        if len(seq) > best_len:
                            best_len = len(seq)
                            best_seq = list(seq)
                        break
    
    return best_seq
