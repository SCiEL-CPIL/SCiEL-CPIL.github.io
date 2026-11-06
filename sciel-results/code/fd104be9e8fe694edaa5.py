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

    def beam_search(width, seed=None):
        """
        Beam search with a fixed width.
        """
        if seed is not None:
            random.seed(seed)
            
        # Each beam is (seq, prefix_sums, forbidden_sums, last_val)
        beams = [([], [0], set(), 0)]
        
        max_depth = 25 
        for depth in range(max_depth):
            if time.time() > time_limit - 5:
                break
                
            next_beams = []
            for beam in beams:
                seq, prefix_sums, forbidden_sums, last = beam
                
                valid_candidates = []
                for x in range(last + 1, 257):
                    if is_valid_candidate(x, prefix_sums, forbidden_sums):
                        valid_candidates.append(x)
                
                if not valid_candidates:
                    continue
                
                # Select candidates to expand
                # We want to keep diversity.
                if len(valid_candidates) > width:
                    # Mix of smallest and random
                    k_smallest = width // 2
                    k_random = width - k_smallest
                    smallest = valid_candidates[:k_smallest]
                    # Ensure we don't sample from the smallest part if possible?
                    # Just sample from all, then union with smallest?
                    # Or sample from the rest?
                    # Let's sample from the whole list, but ensure smallest are included.
                    # If we sample randomly, we might miss the smallest.
                    # So: take smallest k_smallest. Take random k_random from the rest.
                    rest = valid_candidates[k_smallest:]
                    if len(rest) >= k_random:
                        others = random.sample(rest, k_random)
                    else:
                        others = rest
                    candidates_to_try = smallest + others
                else:
                    candidates_to_try = valid_candidates
                
                for x in candidates_to_try:
                    new_seq = seq + [x]
                    new_prefix = prefix_sums[-1] + x
                    new_prefix_sums = prefix_sums + [new_prefix]
                    
                    new_forbidden = forbidden_sums.copy()
                    for p in prefix_sums:
                        new_forbidden.add(new_prefix - p)
                    
                    next_beams.append((new_seq, new_prefix_sums, new_forbidden, x))
            
            if not next_beams:
                break
            
            # Prune beams: keep only the best 'width' ones
            # Score: prefer smaller last value to allow more room?
            # Or prefer beams that have more valid candidates for the next step?
            # Calculating valid candidates for next step is expensive.
            # Smaller last value is a good proxy for "more room".
            next_beams.sort(key=lambda b: b[3])
            beams = next_beams[:width]
            
        # Find the longest sequence in the beams
        if not beams:
            return []
        best_beam = max(beams, key=lambda b: len(b[0]))
        return best_beam[0]

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

    # Phase 1: Randomized Greedy
    attempt_count = 0
    while time.time() - start_time < time_limit - 15:  # Leave 15s for beam search
        # Vary the probability
        p_smallest = random.uniform(0.5, 1.0)
        seq = greedy_construct(seed=attempt_count, prob_smallest=p_smallest)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        
        # Also try a variant with lower bias to smallest
        if attempt_count % 3 == 0:
            seq2 = greedy_construct(seed=attempt_count + 1000, prob_smallest=0.2)
            if len(seq2) > best_len:
                best_len = len(seq2)
                best_seq = seq2
                
        attempt_count += 1
        
    # Phase 2: Beam Search
    # Run beam search with different widths
    for width in [10, 20, 50]:
        if time.time() - start_time > time_limit - 2:
            break
        improved_seq = beam_search(width, seed=attempt_count + width)
        if len(improved_seq) > best_len:
            best_len = len(improved_seq)
            best_seq = improved_seq

    return best_seq
