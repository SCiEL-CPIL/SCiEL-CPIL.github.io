import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s kill
    
    best_seq = []
    best_len = 0
    
    def update_state(seq):
        """
        Rebuilds prefix_sums and forbidden_sums from a sequence.
        """
        prefix_sums = [0]
        forbidden_sums = set()
        for x in seq:
            new_prefix = prefix_sums[-1] + x
            for p in prefix_sums:
                forbidden_sums.add(new_prefix - p)
            prefix_sums.append(new_prefix)
        return prefix_sums, forbidden_sums

    def is_valid_candidate(x, prefix_sums, forbidden_sums):
        """
        Checks if adding x to the current sequence (ending at prefix_sums[-1])
        results in a valid sequence.
        """
        new_prefix = prefix_sums[-1] + x
        for p in prefix_sums:
            d = new_prefix - p
            if d in forbidden_sums:
                return False
        return True

    def greedy_construct(seed=None, prob_smallest=0.8):
        if seed is not None:
            random.seed(seed)
        
        seq = []
        prefix_sums = [0]
        forbidden_sums = set()
        last = 0
        
        for _ in range(256):
            valid_candidates = []
            # Optimization: We can break early if x is too large? No, we need to check all.
            # But we can limit the range if we know we can't fit more?
            # For now, iterate all.
            for x in range(last + 1, 257):
                if is_valid_candidate(x, prefix_sums, forbidden_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Selection Strategy
            if random.random() < prob_smallest:
                x = valid_candidates[0]
            else:
                x = random.choice(valid_candidates)
            
            # Commit x
            seq.append(x)
            last = x
            new_prefix = prefix_sums[-1] + x
            prefix_sums.append(new_prefix)
            
            # Update forbidden_sums
            for p in prefix_sums[:-1]:
                forbidden_sums.add(new_prefix - p)
            
        return seq

    def is_valid_sequence(seq):
        if not seq:
            return True
        if any(seq[i] >= seq[i+1] for i in range(len(seq)-1)):
            return False
        if any(x < 1 or x > 256 for x in seq):
            return False
        p = [0]
        for x in seq:
            p.append(p[-1] + x)
        sums = set()
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                s = p[j] - p[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def local_search(seq, time_limit):
        """
        Attempts to improve the sequence by removing elements and re-adding them.
        Or swapping? Swapping is hard with strict increase.
        Removing and re-adding allows "sliding" elements.
        """
        current_seq = seq[:]
        best_local_len = len(current_seq)
        best_local_seq = current_seq[:]
        
        # We will try to remove one element and see if we can add more
        # Or just try to "shift" elements?
        # Simpler: Try removing an element at position i, and then greedily extend.
        
        # To keep it fast, we only do this for a few iterations or specific positions.
        # Let's try removing each element and re-greedy from there.
        
        for i in range(len(current_seq)):
            if time.time() > time_limit:
                break
            
            # Remove element at i
            temp_seq = current_seq[:i] + current_seq[i+1:]
            
            # Is temp_seq valid?
            # We can check it, but if it's invalid, we can't use it.
            # Let's assume it's valid (removing an element usually doesn't break validity of remaining blocks? 
            # Wait, if we remove a_i, the block a_{i-1}+a_{i+1} is new. It might collide.
            # So we must check validity of temp_seq.
            
            if not is_valid_sequence(temp_seq):
                continue
                
            # Now, can we add more elements to temp_seq?
            # We extend from the end.
            extended_seq = temp_seq[:]
            ext_prefix, ext_forbidden = update_state(extended_seq)
            
            # Greedy extend
            last_val = extended_seq[-1] if extended_seq else 0
            for _ in range(256 - len(extended_seq)):
                valid_candidates = []
                for x in range(last_val + 1, 257):
                    if is_valid_candidate(x, ext_prefix, ext_forbidden):
                        valid_candidates.append(x)
                if not valid_candidates:
                    break
                # Always pick smallest for local search extension? 
                # Or random? Let's pick smallest to be deterministic and fast.
                x = valid_candidates[0]
                extended_seq.append(x)
                last_val = x
                new_p = ext_prefix[-1] + x
                for p in ext_prefix:
                    ext_forbidden.add(new_p - p)
                ext_prefix.append(new_p)
                
            if len(extended_seq) > best_local_len:
                best_local_len = len(extended_seq)
                best_local_seq = extended_seq[:]
                
        return best_local_seq

    # Phase 1: Randomized Greedy
    attempt_count = 0
    # Run a large number of greedy attempts
    # We reserve 10 seconds for local search
    greedy_deadline = start_time + time_limit - 10
    while time.time() < greedy_deadline:
        seq = greedy_construct(seed=attempt_count, prob_smallest=0.8)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        
        # Also try a variant with lower bias to smallest
        if attempt_count % 5 == 0:
            seq2 = greedy_construct(seed=attempt_count + 1000, prob_smallest=0.5)
            if len(seq2) > best_len:
                best_len = len(seq2)
                best_seq = seq2
                
        attempt_count += 1
        
    # Phase 2: Local Search on Best Sequence
    if best_seq and time.time() < start_time + time_limit - 2:
        # Run local search
        improved_seq = local_search(best_seq, start_time + time_limit - 2)
        if len(improved_seq) > best_len:
            best_len = len(improved_seq)
            best_seq = improved_seq
            
    return best_seq
