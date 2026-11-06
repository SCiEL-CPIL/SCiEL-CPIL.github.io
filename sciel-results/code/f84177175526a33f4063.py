"""Improved program for increasing sequences with unique consecutive-block sums."""
import time
import random
import sys

sys.setrecursionlimit(10000)

# EVOLVE-BLOCK-START

def _validate(seq):
    """Check that seq is strictly increasing in [1,256] and all consecutive block sums are distinct."""
    if not seq:
        return False
    if any(x < 1 or x > 256 for x in seq):
        return False
    if any(seq[i] >= seq[i+1] for i in range(len(seq)-1)):
        return False
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _greedy_smallest_first():
    """Original greedy: try each x from 1 to 256, add if no sum collision."""
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


def _greedy_largest_first():
    """Greedy from largest to smallest, then reverse."""
    values, sums = [], set()
    for x in range(256, 0, -1):
        # Check if x can be prepended to values
        tails, total = [x], x
        for a in values:
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.insert(0, x)
            sums.update(tails)
    return values


def _greedy_with_lookahead(start_order=None, max_iter=2000):
    """
    Greedy with lookahead: at each step, among all candidates that don't immediately
    conflict, pick the one that leaves the most future flexibility.
    Flexibility is measured by how few future sums it blocks.
    """
    if start_order is None:
        start_order = list(range(1, 257))
    
    values = []
    sums = set()
    remaining = set(start_order)
    
    for _ in range(len(start_order)):
        if not remaining:
            break
        best_candidate = None
        best_score = -1
        
        for x in sorted(remaining):
            # Check if x can be appended
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if sums.intersection(tails):
                continue
            
            # Lookahead: count how many other remaining candidates would be blocked
            new_sums = sums | set(tails)
            blocked = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if y would be blocked by adding x
                y_tails, y_total = [y], y
                for a in reversed(values):
                    y_total += a
                    y_tails.append(y_total)
                # Also need to check if y after x
                y_after_x_tails, y_after_x_total = [x + y], x + y
                for a in reversed(values):
                    y_after_x_total += a
                    y_after_x_tails.append(y_after_x_total)
                if new_sums.intersection(y_tails) or new_sums.intersection(y_after_x_tails):
                    blocked += 1
            
            score = len(remaining) - blocked
            if score > best_score:
                best_score = score
                best_candidate = x
        
        if best_candidate is None:
            break
        
        values.append(best_candidate)
        remaining.discard(best_candidate)
        # Update sums
        tails, total = [best_candidate], best_candidate
        for a in reversed(values[:-1]):
            total += a
            tails.append(total)
        sums.update(tails)
    
    return values


def _dfs_backtrack(values, sums, remaining, depth, deadline, best_seq, best_len):
    """
    DFS with backtracking to find a longer sequence.
    """
    if depth >= best_len:
        if len(values) > len(best_seq[0]):
            best_seq[0] = list(values)
            best_len[0] = len(values)
        # Don't prune if we just matched best; we might find longer
        # But if we're at best_len and remaining is small, prune
        if depth == best_len[0] and len(remaining) < (best_len[0] - depth + 1):
            return
    
    if time.time() > deadline:
        return
    
    if not remaining:
        return
    
    # Try candidates in order of least blocking (greedy heuristic for ordering)
    candidates = list(remaining)
    # Sort by value to keep deterministic-ish but try diverse orders
    candidates.sort()
    
    for idx, x in enumerate(candidates):
        if time.time() > deadline:
            return
        
        # Check if x can be appended
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        
        if sums.intersection(tails):
            continue
        
        # Add x
        values.append(x)
        new_sums = sums | set(tails)
        remaining.discard(x)
        
        _dfs_backtrack(values, new_sums, remaining, depth + 1, deadline, best_seq, best_len)
        
        # Undo
        values.pop()
        remaining.add(x)
        
        # If we found a longer sequence, we can stop exploring this branch
        if best_len[0] > depth:
            # Continue to see if we can do even better
            pass


def _construct_with_mixed_strategy(deadline):
    """
    Combine multiple strategies:
    1. Greedy smallest-first (baseline)
    2. Greedy largest-first
    3. Greedy with lookahead
    4. DFS backtracking seeded from best greedy result
    5. Random restarts with different orderings
    """
    best_seq = _greedy_smallest_first()
    best_len = len(best_seq)
    
    # Try largest-first
    seq2 = _greedy_largest_first()
    if len(seq2) > best_len:
        best_seq = seq2
        best_len = len(seq2)
    
    # Try lookahead greedy
    if time.time() < deadline - 10:
        seq3 = _greedy_with_lookahead()
        if len(seq3) > best_len:
            best_seq = seq3
            best_len = len(seq3)
    
    # Try random orderings
    random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        seq = _greedy_with_lookahead(start_order=order)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # DFS backtracking from the best sequence found
    if time.time() < deadline - 5 and best_len < 256:
        # Start DFS from the best sequence, try to extend it
        # Use a modified DFS that starts from the current best
        values = list(best_seq)
        sums = set()
        n = len(values)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += values[j]
                sums.add(s)
        
        remaining = set(range(1, 257)) - set(values)
        best_seq_holder = [list(values)]
        best_len_holder = [len(values)]
        
        # Try to extend by trying to insert elements
        # Simple approach: try adding one element at a time from remaining
        # This is a shallow DFS
        
        def try_extend(vals, sm, rem, depth, max_depth=3):
            if time.time() > deadline - 2:
                return
            if depth >= max_depth:
                return
            if len(vals) > best_len_holder[0]:
                best_len_holder[0] = len(vals)
                best_seq_holder[0] = list(vals)
            
            for x in sorted(rem):
                if time.time() > deadline - 2:
                    return
                
                # Check if x can be appended
                tails, total = [x], x
                for a in reversed(vals):
                    total += a
                    tails.append(total)
                
                if sm.intersection(tails):
                    continue
                
                new_vals = vals + [x]
                new_sm = sm | set(tails)
                new_rem = rem - {x}
                
                try_extend(new_vals, new_sm, new_rem, depth + 1, max_depth)
        
        try_extend(values, sums, remaining, 0, max_depth=2)
        
        if best_len_holder[0] > best_len:
            best_seq = best_seq_holder[0]
            best_len = best_len_holder[0]
    
    return best_seq


def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline to be safe
    
    # Start with the baseline greedy
    best_seq = _greedy_smallest_first()
    best_len = len(best_seq)
    
    # Try other strategies
    seq2 = _greedy_largest_first()
    if len(seq2) > best_len:
        best_seq = seq2
        best_len = len(seq2)
    
    # Try lookahead greedy
    if time.time() < deadline - 15:
        seq3 = _greedy_with_lookahead()
        if len(seq3) > best_len:
            best_seq = seq3
            best_len = len(seq3)
    
    # Try random orderings
    random.seed(12345)
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        seq = _greedy_with_lookahead(start_order=order)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Try a more targeted DFS: start from best and try to insert missing elements
    if time.time() < deadline - 5 and best_len < 256:
        values = list(best_seq)
        sums = set()
        n = len(values)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += values[j]
                sums.add(s)
        
        remaining = set(range(1, 257)) - set(values)
        
        # Try to extend by appending
        extended = True
        while extended and time.time() < deadline - 3:
            extended = False
            for x in sorted(remaining):
                if time.time() > deadline - 3:
                    break
                tails, total = [x], x
                for a in reversed(values):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    values.append(x)
                    sums.update(tails)
                    remaining.discard(x)
                    extended = True
                    best_len = len(values)
                    best_seq = list(values)
                    break
    
    # Final validation
    if not _validate(best_seq):
        # Fallback to the known-valid greedy
        best_seq = _greedy_smallest_first()
    
    return best_seq

# EVOLVE-BLOCK-END
