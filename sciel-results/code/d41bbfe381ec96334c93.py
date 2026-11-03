import time
import random
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet q with covering radius radius.
    Returns a list of codewords (lists of ints).
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits
    
    # Precompute neighbor masks as Python integers (bitmasks)
    # For q=3, n=8: total_words=6561, each mask is a 6561-bit integer
    # This is efficient for set operations using bitwise AND/OR
    
    neighbor_masks = [0] * total_words
    
    for idx in range(total_words):
        mask = 1 << idx
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    new_idx = base + new_val * powers[i]
                    mask |= (1 << new_idx)
        neighbor_masks[idx] = mask
    
    # Criticality-based greedy: prioritize uncovered words with fewest covering candidates
    # coverage_count[w] = number of codewords that cover w
    # uncovered_mask: bitmask of uncovered words
    
    uncovered_mask = (1 << total_words) - 1
    code = []
    coverage_count = [0] * total_words
    
    # Precompute: for each word, which words cover it (reverse neighbors)
    # Since distance is symmetric, rev_neighbors[i] == neighbors[i]
    # So the candidates for covering word w are exactly neighbor_masks[w]
    
    def greedy_step():
        """Pick the hardest uncovered word, then best candidate to cover it."""
        # Find uncovered words
        # For efficiency, iterate through uncovered words and find the one with fewest candidates
        # that can still cover other uncovered words
        
        # Get list of uncovered indices
        uncov = []
        m = uncovered_mask
        while m:
            # Get lowest set bit
            lb = m & (-m)
            idx = lb.bit_length() - 1
            uncov.append(idx)
            m ^= lb
        
        if not uncov:
            return None
        
        # For each uncovered word, count how many of its neighbors are also uncovered
        # and how many candidates can cover it
        # We want to pick the word that is "hardest" to cover
        
        # Strategy: pick a random uncovered word, then pick the best candidate
        # But to be smarter, pick the uncovered word with the fewest "useful" candidates
        # (candidates that cover the most other uncovered words)
        
        # For speed, sample a few uncovered words
        sample_size = min(5, len(uncov))
        sampled = random.sample(uncov, sample_size)
        
        best_word = None
        best_cand = None
        best_cov = -1
        
        for w in sampled:
            # Candidates are neighbors of w
            # Find the candidate that covers the most uncovered words
            cand_mask = neighbor_masks[w]
            # Get candidate indices
            cands = []
            cm = cand_mask
            while cm:
                lb = cm & (-cm)
                ci = lb.bit_length() - 1
                cands.append(ci)
                cm ^= lb
            
            # For each candidate, count uncovered neighbors
            for c in cands:
                # Count how many uncovered words are covered by c
                # uncovered_mask & neighbor_masks[c] gives the covered uncovered words
                covered_uncovered = uncovered_mask & neighbor_masks[c]
                cov_count = covered_uncovered.bit_count()
                if cov_count > best_cov:
                    best_cov = cov_count
                    best_cand = c
                    best_word = w
        
        return best_cand
    
    # Main greedy loop
    while uncovered_mask != 0:
        if time.time() - start_time > deadline - 10:
            break
        
        cand = greedy_step()
        if cand is None:
            break
        
        code.append(cand)
        # Update uncovered mask and coverage count
        covered = neighbor_masks[cand]
        uncovered_mask &= ~covered
        
        # Update coverage count
        m = covered
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            coverage_count[idx] += 1
            m ^= lb
    
    # Local search: 2-opt style
    # Try to remove redundant codewords
    # A codeword c is redundant if all words it covers are also covered by other codewords
    
    def is_redundant(c):
        """Check if codeword c is redundant (all its neighbors are covered by others)."""
        # For each word w covered by c, check if coverage_count[w] > 1
        # (since c covers w, if coverage_count[w] > 1, some other codeword covers it too)
        m = neighbor_masks[c]
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            if coverage_count[idx] <= 1:
                return False
            m ^= lb
        return True
    
    def try_remove():
        """Try to remove a redundant codeword. Returns True if removed."""
        for i in range(len(code)):
            if is_redundant(code[i]):
                # Remove it
                c = code[i]
                m = neighbor_masks[c]
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    coverage_count[idx] -= 1
                    m ^= lb
                code.pop(i)
                return True
        return False
    
    # Phase 1: Simple removal of redundant codewords
    improved = True
    while improved:
        if time.time() - start_time > deadline - 5:
            break
        improved = try_remove()
    
    # Phase 2: 2-opt local search
    # Try replacing one codeword with another to enable further removals
    # This is more complex: we try to swap codeword A with codeword B
    # if B is not already in code, and after the swap we can remove more codewords
    
    def try_swap_remove():
        """Try to find a swap that enables removal of another codeword."""
        if len(code) < 2:
            return False
        
        # For each pair (i, j), try replacing code[i] with some candidate
        # that covers what code[i] covers and enables removal of code[j]
        
        # Simplified: for each codeword i, try replacing it with a neighbor
        # that is not already in code, and check if we can then remove another codeword
        
        for i in range(len(code)):
            if time.time() - start_time > deadline - 3:
                return False
            
            c_i = code[i]
            
            # Try candidates that are neighbors of c_i but not already in code
            # and not equal to c_i
            in_code_set = set(code)
            
            # Get neighbors of c_i
            m = neighbor_masks[c_i]
            candidates = []
            while m:
                lb = m & (-m)
                ci = lb.bit_length() - 1
                if ci != c_i and ci not in in_code_set:
                    candidates.append(ci)
                m ^= lb
            
            # Try a few random candidates
            random.shuffle(candidates)
            for new_c in candidates[:20]:
                # Tentatively replace c_i with new_c
                # Remove c_i's coverage
                m_remove = neighbor_masks[c_i]
                while m_remove:
                    lb = m_remove & (-m_remove)
                    idx = lb.bit_length() - 1
                    coverage_count[idx] -= 1
                    m_remove ^= lb
                
                # Add new_c's coverage
                m_add = neighbor_masks[new_c]
                while m_add:
                    lb = m_add & (-m_add)
                    idx = lb.bit_length() - 1
                    coverage_count[idx] += 1
                    m_add ^= lb
                
                # Check if we can remove another codeword
                removed_idx = -1
                for j in range(len(code)):
                    if j == i:
                        continue
                    if is_redundant(code[j]):
                        removed_idx = j
                        break
                
                if removed_idx != -1:
                    # Perform the swap and removal
                    code[i] = new_c
                    c_removed = code[removed_idx]
                    m_rm = neighbor_masks[c_removed]
                    while m_rm:
                        lb = m_rm & (-m_rm)
                        idx = lb.bit_length() - 1
                        coverage_count[idx] -= 1
                        m_rm ^= lb
                    code.pop(removed_idx)
                    return True
                else:
                    # Undo the swap
                    m_undo_remove = neighbor_masks[new_c]
                    while m_undo_remove:
                        lb = m_undo_remove & (-m_undo_remove)
                        idx = lb.bit_length() - 1
                        coverage_count[idx] -= 1
                        m_undo_remove ^= lb
                    
                    m_undo_add = neighbor_masks[c_i]
                    while m_undo_add:
                        lb = m_undo_add & (-m_undo_add)
                        idx = lb.bit_length() - 1
                        coverage_count[idx] += 1
                        m_undo_add ^= lb
        
        return False
    
    # Run 2-opt for a while
    improved = True
    while improved:
        if time.time() - start_time > deadline - 2:
            break
        improved = try_swap_remove()
        # After a swap, try simple removals again
        while try_remove():
            pass
    
    # Final cleanup: remove any remaining redundant codewords
    improved = True
    while improved:
        if time.time() - start_time > deadline - 1:
            break
        improved = try_remove()
    
    # Convert to list of lists
    result = [get_digits(c) for c in code]
    return result
