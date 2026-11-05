"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import log, exp

def _check_validity(seq):
    """Check if all nonempty consecutive block products are distinct."""
    if not seq:
        return True
    products = set()
    for i in range(len(seq)):
        prod = 1
        for j in range(i, len(seq)):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _incremental_insert_check(seq, new_val, pos):
    """Check if inserting new_val at position pos in seq keeps all block products distinct.
    Only need to check blocks that include the new element.
    Returns (True, set_of_new_products) if valid, else (False, None)."""
    n = len(seq)
    # All blocks that include new_val at position pos:
    # They are seq[i..pos-1] + [new_val] + seq[pos+1..j] for various i, j
    # Actually, blocks containing new_val are:
    # For start s in [0, pos] (where pos is the 0-indexed position in the new sequence),
    # and end e in [pos, pos + (n - pos)]... 
    # Let me think more carefully.
    # In the new sequence, new_val is at index pos.
    # A block [a..b] contains new_val iff a <= pos <= b.
    # The product is prod(seq[a..pos-1]) * new_val * prod(seq[pos+1..b])
    # where seq[pos-1] means the element before pos in the original seq (which is seq[pos-1] if pos>0)
    # and seq[pos+1..b] means elements after pos in new seq, which are seq[pos..b-1] in original.
    
    # Let's just compute all blocks containing the new element and check against existing products.
    
    # First, get all existing products
    existing_products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            existing_products.add(prod)
    
    # Now check all blocks containing new_val at position pos in the new sequence
    # In the new sequence, positions 0..pos-1 are seq[0..pos-1], position pos is new_val,
    # positions pos+1..n are seq[pos..n-1]
    
    # Block from start s to end e where s <= pos <= e (in new sequence indices)
    # s ranges from 0 to pos, e ranges from pos to n
    # Product = (prod of seq[s..pos-1] if s < pos else 1) * new_val * (prod of seq[pos..e-1] if e > pos else 1)
    
    new_products = set()
    for s in range(pos + 1):  # s from 0 to pos
        left_prod = 1
        for k in range(s, pos):
            left_prod *= seq[k]
        for e in range(pos, n + 1):  # e from pos to n
            right_prod = 1
            for k in range(pos, e):
                right_prod *= seq[k]
            block_prod = left_prod * new_val * right_prod
            if block_prod in existing_products or block_prod in new_products:
                return False, None
            new_products.add(block_prod)
    
    return True, new_products

def _greedy_insertion(seq, candidates):
    """Try to insert candidates into seq to maximize length while maintaining validity."""
    result = list(seq)
    remaining = list(candidates)
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(remaining):
            # Try inserting c at each position
            best_pos = None
            for pos in range(len(result) + 1):
                if pos < len(result) and result[pos] <= c:
                    continue  # Need strictly increasing
                if pos > 0 and result[pos - 1] >= c:
                    continue
                valid, _ = _incremental_insert_check(result, c, pos)
                if valid:
                    best_pos = pos
                    break
            if best_pos is not None:
                result.insert(best_pos, c)
                remaining.pop(i)
                changed = True
                break
    return result

def construct():
    deadline = time.time() + 85  # 85 second deadline, 5 second safety margin
    best_seq = []
    best_len = 0
    
    # First, get a good baseline using the original greedy approach
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    best_seq = values
    best_len = len(values)
    
    # Now try to improve with multi-start greedy
    random.seed(42)
    all_candidates = list(range(2, 129))
    
    # Try different starting points and orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Create a shuffled candidate list
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        
        # Start with a small random subset
        start_size = min(3, len(shuffled))
        start_seq = sorted(shuffled[:start_size])
        
        # Verify start_seq is valid
        if not _check_validity(start_seq):
            continue
        
        # Try to insert remaining candidates
        remaining = [c for c in shuffled if c not in start_seq]
        result = _greedy_insertion(start_seq, remaining)
        
        if len(result) > best_len:
            best_len = len(result)
            best_seq = result
    
    # Try targeted improvement: take best_seq and try to insert more elements
    if time.time() < deadline - 5:
        used = set(best_seq)
        unused = [x for x in range(2, 129) if x not in used]
        
        # Try inserting unused elements one by one
        improved = list(best_seq)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for c in unused[:]:
                best_pos = None
                for pos in range(len(improved) + 1):
                    if pos < len(improved) and improved[pos] <= c:
                        continue
                    if pos > 0 and improved[pos - 1] >= c:
                        continue
                    valid, _ = _incremental_insert_check(improved, c, pos)
                    if valid:
                        best_pos = pos
                        break
                if best_pos is not None:
                    improved.insert(best_pos, c)
                    unused.remove(c)
                    changed = True
        
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved
    
    return best_seq
# EVOLVE-BLOCK-END
