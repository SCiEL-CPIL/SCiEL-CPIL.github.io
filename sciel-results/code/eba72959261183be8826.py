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
    
    def compute_all_products(seq):
        """Compute all consecutive block products for a sequence."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods
    
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
    
    def try_1out_2in(seq, deadline):
        """Try removing one element and inserting two new elements to escape local optima."""
        seq_list = list(seq)
        n = len(seq_list)
        seq_set = set(seq_list)
        
        for remove_idx in range(n):
            if time.time() > deadline:
                return None, False
            removed = seq_list[remove_idx]
            # Try inserting two new elements after removal
            reduced = seq_list[:remove_idx] + seq_list[remove_idx+1:]
            reduced_set = set(reduced)
            
            # First try to insert one new element
            for x1 in range(2, 129):
                if x1 in reduced_set:
                    continue
                if time.time() > deadline:
                    return None, False
                for pos1 in range(len(reduced) + 1):
                    if pos1 > 0 and reduced[pos1-1] >= x1:
                        continue
                    if pos1 < len(reduced) and reduced[pos1] <= x1:
                        continue
                    if check_insertion(reduced, x1, pos1):
                        trial1 = reduced[:pos1] + [x1] + reduced[pos1:]
                        trial1_set = set(trial1)
                        
                        # Now try to insert a second new element
                        for x2 in range(2, 129):
                            if x2 in trial1_set:
                                continue
                            if time.time() > deadline:
                                return None, False
                            for pos2 in range(len(trial1) + 1):
                                if pos2 > 0 and trial1[pos2-1] >= x2:
                                    continue
                                if pos2 < len(trial1) and trial1[pos2] <= x2:
                                    continue
                                if check_insertion(trial1, x2, pos2):
                                    trial2 = trial1[:pos2] + [x2] + trial1[pos2:]
                                    return trial2, True
                        
                        # If we could insert x1, accept even without x2
                        if len(trial1) > len(seq_list):
                            return trial1, True
                        return trial1, True
        return None, False
    
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
    while changed and time.time() < DEADLINE - 15.0:
        changed = False
        excluded = [x for x in range(2, 129) if x not in seq]
        for x in excluded:
            if time.time() > DEADLINE - 15.0:
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
    
    # 3. 1-out/2-in local search to escape local optima
    if time.time() < DEADLINE - 10.0:
        improved = True
        while improved and time.time() < DEADLINE - 10.0:
            improved = False
            result, success = try_1out_2in(seq, DEADLINE - 10.0)
            if success and result is not None:
                seq = result
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = list(seq)
                improved = True
    
    # 4. Try 1-out/1-in swaps to potentially unlock longer sequences
    if time.time() < DEADLINE - 5.0:
        improved = True
        while improved and time.time() < DEADLINE - 5.0:
            improved = False
            seq_set = set(seq)
            for i in range(len(seq)):
                if time.time() > DEADLINE - 5.0:
                    break
                for x in range(2, 129):
                    if x in seq_set:
                        continue
                    if time.time() > DEADLINE - 5.0:
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
    
    # 5. Final fixed-point pass after local search
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
