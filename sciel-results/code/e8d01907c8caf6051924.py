import time
import random

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 55  # 55s safety margin before 60s kill
    
    def check_valid(seq):
        """Check if all non-empty consecutive block sums are distinct."""
        if not seq:
            return True
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

    def suffix_sums(seq):
        """Return list of suffix sums: [seq[-1], seq[-2]+seq[-1], ..., sum(seq)]"""
        result = []
        s = 0
        for x in reversed(seq):
            s += x
            result.append(s)
        return result

    def greedy_reverse():
        """Build sequence backwards: pick largest feasible value first."""
        seq = []
        used_sums = set()
        # We'll build from high to low, but the final sequence must be increasing.
        # So we pick values in decreasing order, but insert them at the front.
        # Actually, let's think: if we build backwards, we start with the largest element.
        # Then we try to add a smaller element to the left.
        # The new consecutive sums when prepending x to seq [a1, a2, ..., ak] are:
        # x, x+a1, x+a1+a2, ..., x+a1+...+ak
        # Plus all existing sums from seq.
        
        # Let's try: start empty, pick largest possible, then next largest that fits to the left.
        for x in range(N, 0, -1):
            # Try to prepend x to current seq
            if not seq:
                seq = [x]
                used_sums.add(x)
                continue
            
            # Compute new sums if we prepend x
            new_sums = [x]
            s = x
            for a in seq:
                s += a
                new_sums.append(s)
            
            # Check if any new sum conflicts
            conflict = False
            for ns in new_sums:
                if ns in used_sums:
                    conflict = True
                    break
            
            if not conflict:
                seq = [x] + seq
                for ns in new_sums:
                    used_sums.add(ns)
        
        return seq

    # Try multiple random seeds for diversity
    best_seq = []
    best_len = 0
    
    # First, get a baseline from reverse greedy
    base_seq = greedy_reverse()
    if len(base_seq) > best_len:
        best_len = len(base_seq)
        best_seq = base_seq[:]
    
    # Local search: try to improve by swapping, inserting, or deleting
    def local_search(seq, time_limit=50):
        nonlocal best_seq, best_len
        random.shuffle(seq)  # This won't work, seq is increasing. Let's use a different approach.
        
        # Better: start from a valid sequence and try modifications
        # Modification types:
        # 1. Delete an element
        # 2. Swap two adjacent elements (but must maintain increasing order)
        # 3. Insert a new element at a valid position
        
        # Let's use a hill-climbing approach with random restarts
        def try_modify(seq):
            """Try random modifications and return improved seq or None."""
            if not seq:
                return None
            
            # Generate random modification
            mod_type = random.random()
            
            if mod_type < 0.3:
                # Delete a random element
                idx = random.randint(0, len(seq) - 1)
                new_seq = seq[:idx] + seq[idx+1:]
                if check_valid(new_seq):
                    return new_seq
                return None
                
            elif mod_type < 0.6:
                # Swap two elements (must maintain increasing order)
                if len(seq) < 2:
                    return None
                i, j = random.sample(range(len(seq)), 2)
                if i > j:
                    i, j = j, i
                # After swap, check if still increasing
                if i + 1 == j:
                    # Adjacent swap
                    if seq[i] < seq[j]:  # Already increasing, swap would break it
                        # Actually, we need seq[i] > seq[j] for swap to maintain increasing? No.
                        # If seq = [1, 3, 5], swap 1 and 3 -> [3, 1, 5] not increasing.
                        # So adjacent swap only works if we're going to re-sort? No.
                        # Let's skip adjacent swaps for now, they break monotonicity.
                        return None
                else:
                    # Non-adjacent swap
                    new_seq = seq[:]
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    # Check if still increasing
                    for k in range(len(new_seq) - 1):
                        if new_seq[k] >= new_seq[k+1]:
                            return None
                    if check_valid(new_seq):
                        return new_seq
                    return None
                return None
            else:
                # Insert a new element
                # Try to insert a random value in a random position
                pos = random.randint(0, len(seq))
                # Find a value that fits at pos
                if pos == 0:
                    lo, hi = 1, seq[0] - 1 if seq else N
                elif pos == len(seq):
                    lo, hi = seq[-1] + 1, N
                else:
                    lo, hi = seq[pos-1] + 1, seq[pos] - 1
                
                if lo > hi:
                    return None
                
                x = random.randint(lo, hi)
                new_seq = seq[:pos] + [x] + seq[pos:]
                if check_valid(new_seq):
                    return new_seq
                return None
        
        # Hill climbing
        current = seq[:]
        no_improve = 0
        while time.time() < time_limit:
            improved = False
            for _ in range(100):  # Try 100 modifications per iteration
                if time.time() > time_limit:
                    break
                new_seq = try_modify(current)
                if new_seq is not None:
                    if len(new_seq) > len(current):
                        current = new_seq
                        improved = True
                        no_improve = 0
                        if len(current) > best_len:
                            best_len = len(current)
                            best_seq = current[:]
                        break
                    elif len(new_seq) == len(current) and random.random() < 0.1:
                        # Sometimes accept same-length for diversity
                        current = new_seq
                        no_improve = 0
                        break
            if not improved:
                no_improve += 1
                if no_improve > 20:
                    break
                # Random restart: perturb current
                if current:
                    idx = random.randint(0, len(current) - 1)
                    current = current[:idx] + current[idx+1:]
                    if not check_valid(current):
                        # Revert
                        idx2 = random.randint(0, len(current))
                        # Just break and try again
                        break
        
        return current

    # Run local search from the greedy result
    if best_seq:
        improved = local_search(best_seq[:], time_limit=deadline - 5)
        if improved and len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved[:]
    
    # Also try random initial sequences
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        
        # Create a random increasing sequence
        length = random.randint(10, 30)
        seq = sorted(random.sample(range(1, N + 1), length))
        
        # Validate and improve
        if check_valid(seq):
            improved = local_search(seq, time_limit=deadline - 10)
            if improved and len(improved) > best_len:
                best_len = len(improved)
                best_seq = improved[:]
    
    return best_seq
