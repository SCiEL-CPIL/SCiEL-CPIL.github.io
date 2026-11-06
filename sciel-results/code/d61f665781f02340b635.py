import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin for 90s limit
    
    def check_collisions(seq):
        """
        Returns a dictionary: sum_value -> list of (start, end) indices.
        Also returns a set of all sums.
        """
        sums_dict = {}
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums_dict:
                    sums_dict[s].append((i, j))
                else:
                    sums_dict[s] = [(i, j)]
        return sums_dict

    def is_valid(seq):
        sums_dict = check_collisions(seq)
        for s, blocks in sums_dict.items():
            if len(blocks) > 1:
                return False
        return True

    def count_collisions_involving(seq, index):
        """
        Count how many sum collisions involve the element at 'index'.
        A collision is between two blocks. If 'index' is in either block, it's involved.
        """
        sums_dict = check_collisions(seq)
        count = 0
        for s, blocks in sums_dict.items():
            if len(blocks) > 1:
                # Check if any block contains 'index'
                # Actually, if two blocks have the same sum, and one contains index and the other doesn't,
                # removing index breaks the collision for that pair.
                # If both contain index, removing it might not break it immediately if the sums change?
                # No, removing an element changes the sequence length and indices.
                # Simpler: Just count pairs of blocks with same sum where at least one block contains 'index'.
                involved = False
                for (i1, j1) in blocks:
                    if i1 <= index <= j1:
                        involved = True
                        break
                if involved:
                    count += 1
        return count

    def greedy_insert(seq, candidates):
        """
        Try to insert each candidate from 'candidates' into 'seq' at the best position.
        Returns the new sequence if an insertion was made, else None.
        """
        best_new_seq = None
        best_score = -1
        for x in candidates:
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [x] + seq[pos:]
                if is_valid(new_seq):
                    # Score: Prefer inserting smaller numbers? Or numbers that allow more future growth?
                    # For now, just accept the first valid insertion found in this pass?
                    # To be more strategic, we can look for the insertion that minimizes the max sum?
                    # Let's just return the first valid one to keep it fast, 
                    # but we iterate candidates in a specific order.
                    return new_seq
        return None

    def local_search_restart(initial_seq):
        """
        1. Remove elements causing collisions.
        2. Grow the sequence by inserting unused elements.
        """
        seq = list(initial_seq)
        unused = set(range(1, 257)) - set(seq)
        
        # Phase 1: Clean up collisions
        while not is_valid(seq):
            # Find element with most collisions
            max_col = -1
            target_idx = -1
            for i in range(len(seq)):
                col = count_collisions_involving(seq, i)
                if col > max_col:
                    max_col = col
                    target_idx = i
            
            if target_idx == -1:
                # No collisions found by this metric? Should not happen if is_valid is False
                break
                
            # Remove the element
            removed_val = seq[target_idx]
            seq.pop(target_idx)
            unused.add(removed_val)
            
            if time.time() - start_time > deadline:
                break

        # Phase 2: Grow the sequence
        # We try to insert unused elements.
        # To maximize length, we should try to insert as many as possible.
        # Order of insertion: Smaller numbers first? They leave more room for larger sums?
        # Or larger numbers first?
        # Let's try sorting unused by value.
        sorted_unused = sorted(unused)
        
        progress = True
        while progress:
            progress = False
            # We need to re-sort or iterate carefully.
            # Since inserting one might block another, we do passes.
            for x in sorted_unused:
                if x in seq:
                    continue
                new_seq = greedy_insert(seq, [x])
                if new_seq is not None:
                    seq = new_seq
                    unused.remove(x)
                    progress = True
                    if time.time() - start_time > deadline:
                        return seq
            # If no progress in a full pass, we might be stuck.
            # Try a different order? Or break.
            # For simplicity, break.
        
        return seq

    best_seq = []
    
    # Strategy: Multiple random restarts with different initial sets
    # Initial sets: Random subsets of [1, 256] of varying sizes.
    
    # 1. Try a greedy "smallest first" as a baseline
    baseline = []
    for x in range(1, 257):
        if time.time() - start_time > deadline:
            break
        new_seq = greedy_insert(baseline, [x])
        if new_seq is not None:
            baseline = new_seq
    if len(baseline) > len(best_seq):
        best_seq = baseline

    # 2. Random restarts
    # We have ~80 seconds left.
    # Each restart involves cleanup (O(N^3)) and growth (O(K * N^3)).
    # N is small (~20). K is ~200.
    # 200 * 20^3 = 1.6 million ops per restart. Very fast.
    # We can do many restarts.
    
    restart_count = 0
    while time.time() - start_time < deadline - 5:
        # Generate a random initial sequence
        # Size: Between 10 and 20
        size = random.randint(10, 20)
        initial = random.sample(range(1, 257), size)
        # Shuffle to get different orderings
        random.shuffle(initial)
        
        seq = local_search_restart(initial)
        restart_count += 1
        
        if len(seq) > len(best_seq):
            best_seq = seq
            
        # If we found a very long sequence, we might want to try to refine it further
        # But for now, just keep the best.
        
    return best_seq
