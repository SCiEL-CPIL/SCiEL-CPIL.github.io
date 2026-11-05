import itertools
import time
import sys
from collections import defaultdict

def _standard_superperm(n):
    """Generate standard superpermutation for n using recursive construction."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    if n == 3:
        return [1, 2, 3, 1, 2, 1, 3, 2, 1]
    
    prev = _standard_superperm(n - 1)
    
    seen = set()
    order = []
    for i in range(len(prev) - (n - 1) + 1):
        window = tuple(prev[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    
    return new

def _verify_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive subsequences."""
    all_perms = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in all_perms:
            found.add(p)
        if len(found) == len(all_perms):
            return True
    return len(found) == len(all_perms)

def _construct_from_order(order, n):
    """
    Construct superpermutation for n from the given order of (n-1)-permutations.
    Returns the word as a list of ints.
    """
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    return new

def _extract_order_from_word(word, n):
    """Extract the order of (n-1)-permutations from a superpermutation for n."""
    seen = set()
    order = []
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    return order

def _compute_length_from_order(order, n):
    """
    Compute the exact length of the superpermutation built from the given order.
    Uses incremental string building with overlap detection.
    """
    total = 0
    current = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(current), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or current[len(current) - ov:] == block[:ov]:
                overlap = ov
                break
        total += len(block) - overlap
        current.extend(block[overlap:])
        # Keep current bounded for speed - we only need the tail
        if len(current) > 2 * n + 10:
            current = current[-(2 * n + 10):]
    return total

def _greedy_set_cover_superperm(n, deadline):
    """
    Greedy set-cover heuristic: build the word one symbol at a time.
    At each step, choose the next symbol that maximizes the number of NEW
    permutations completed, with tie-breaking by minimum remaining options
    (constrained permutations first).
    
    A permutation is "completed" when it appears as a window of length n.
    We track which permutations have been found.
    
    Key insight: we want to maximize overlap between consecutive windows,
    which means preferring symbols that extend existing partial matches.
    """
    all_perms = set(itertools.permutations(range(1, n + 1)))
    found = set()
    word = []
    
    # Start with a reasonable seed - the first permutation
    word = list(range(1, n + 1))
    found.add(tuple(word))
    
    symbols = list(range(1, n + 1))
    
    while len(found) < len(all_perms) and time.time() < deadline:
        # For each candidate symbol, compute:
        # 1. How many new permutations would be completed
        # 2. Among those, prioritize completing the most "constrained" ones
        #    (permutations with fewest possible next extensions)
        
        best_sym = None
        best_score = -1
        best_new_count = 0
        
        # Get the last n-1 symbols as context
        if len(word) >= n - 1:
            context = tuple(word[-(n-1):])
        else:
            context = tuple(word)
        
        for sym in symbols:
            # What new window would be formed?
            new_window = context + (sym,)
            
            # How many new permutations does this complete?
            # Only the last window is new, so at most 1 new permutation per step
            # But we can look ahead: what permutations could be completed soon?
            
            # Strategy: score based on how many "partial" matches this extends
            # A partial match is a suffix of the current word that is a prefix
            # of some unfound permutation
            
            # Count how many unfound permutations have this new_window as a prefix
            # (i.e., the new window is the first n symbols of some unfound perm)
            # Actually, new_window IS a full permutation if it's a valid perm.
            
            new_count = 0
            if new_window in all_perms and new_window not in found:
                new_count = 1
            
            # Look-ahead: for each unfound permutation, check if extending
            # by sym brings us closer (longer common prefix with some unfound perm)
            # This is expensive, so use a heuristic:
            # Score = new_count * 1000 + (number of unfound perms that have new_window as prefix)
            
            # Faster heuristic: count how many unfound perms start with the
            # last min(n, len(context)+1) symbols
            prefix_len = min(n, len(context) + 1)
            if prefix_len >= n:
                # new_window is a full permutation
                if new_window in all_perms and new_window not in found:
                    score = 10000
                else:
                    score = 0
            else:
                # Count unfound perms that have this prefix
                prefix = new_window
                count = 0
                for p in all_perms:
                    if p in found:
                        continue
                    if p[:prefix_len] == prefix:
                        count += 1
                score = count
            
            # Tie-break: prefer symbols that appear less recently (diversity)
            # to avoid getting stuck in loops
            last_occurrence = -1
            for k in range(len(word) - 1, -1, -1):
                if word[k] == sym:
                    last_occurrence = len(word) - 1 - k
                    break
            diversity_bonus = min(last_occurrence, 10) * 0.1 if last_occurrence >= 0 else 1.0
            
            total_score = score + diversity_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_sym = sym
                best_new_count = new_count
        
        if best_sym is None:
            break
        
        word.append(best_sym)
        
        # Update found set
        if len(word) >= n:
            last_window = tuple(word[-n:])
            if last_window in all_perms:
                found.add(last_window)
        
        # Safety: if we're making no progress, break
        if len(word) > n * n * 2:  # Should never happen for valid construction
            break
    
    return word, len(found) == len(all_perms)

def _greedy_with_backtrack(n, deadline):
    """
    Improved greedy: at each step, consider all possible next symbols and
    pick the one that maximizes a composite score combining:
    1. Immediate new permutations completed
    2. Number of unfound permutations that remain reachable
    3. Avoidance of creating "dead-end" states where too many perms are blocked
    
    Uses a lookahead of 2-3 steps for better decisions.
    """
    all_perms = set(itertools.permutations(range(1, n + 1)))
    found = set()
    word = list(range(1, n + 1))
    found.add(tuple(word))
    
    symbols = list(range(1, n + 1))
    total_perms = len(all_perms)
    
    # Precompute: for each permutation, its suffixes of length 1..n-1
    # This helps with fast prefix matching
    
    # For efficiency, precompute prefix counts for unfound perms
    # prefix_count[k][prefix_tuple] = number of unfound perms with that prefix
    
    def get_prefix_counts():
        """Compute prefix counts for all unfound permutations."""
        counts = defaultdict(int)
        for p in all_perms:
            if p in found:
                continue
            for k in range(1, n):
                counts[p[:k]] += 1
        return counts
    
    prefix_counts = get_prefix_counts()
    
    while len(found) < total_perms and time.time() < deadline:
        if len(word) < n - 1:
            # Not enough context yet, just append next symbol
            next_sym = n + 1 - len(word) if len(word) < n else 1
            word.append(next_sym)
            if len(word) >= n:
                w = tuple(word[-n:])
                if w in all_perms:
                    found.add(w)
                    # Update prefix counts
                    for k in range(1, n):
                        if w[:k] in prefix_counts:
                            prefix_counts[w[:k]] -= 1
            continue
        
        context = tuple(word[-(n-1):])
        
        best_sym = None
        best_score = -1
        
        for sym in symbols:
            new_window = context + (sym,)
            
            # Score component 1: immediate completion
            immediate = 0
            if new_window in all_perms and new_window not in found:
                immediate = 1000
            
            # Score component 2: how many unfound perms does this prefix match?
            # Higher means more "potential" to complete soon
            potential = prefix_counts.get(new_window, 0) if len(new_window) < n else 0
            
            # Score component 3: penalty for creating a state where
            # the next step has very few options (dead-end risk)
            # If after appending sym, the new context has very few
            # matching unfound perms, that's risky
            new_context = new_window[-(n-1):] if len(new_window) >= n - 1 else new_window
            next_potential = 0
            for ns in symbols:
                nc = new_context + (ns,)
                if len(nc) < n:
                    next_potential += prefix_counts.get(nc, 0)
                elif nc in all_perms and nc not in found:
                    next_potential += 10
            
            # If next_potential is 0, we're in a dead end
            dead_end_penalty = -5000 if next_potential == 0 else 0
            
            # Score component 4: diversity - prefer symbols not recently used
            recent_count = sum(1 for x in word[-5:] if x == sym) if len(word) >= 5 else sum(1 for x in word if x == sym)
            diversity_penalty = recent_count * 10
            
            total_score = immediate + potential * 10 + dead_end_penalty - diversity_penalty
            
            if total_score > best_score:
                best_score = total_score
                best_sym = sym
        
        if best_sym is None:
            break
        
        word.append(best_sym)
        
        # Update found and prefix counts
        if len(word) >= n:
            w = tuple(word[-n:])
            if w in all_perms and w not in found:
                found.add(w)
                for k in range(1, n):
                    if w[:k] in prefix_counts:
                        prefix_counts[w[:k]] -= 1
        
        # Periodically refresh prefix counts (they get stale)
        # Actually, we update them incrementally above, so they should be fine
        
        # Safety check
        if len(word) > total_perms + n * 10:
            break
    
    return word, len(found) == total_perms

def _vns_improve_order(order, n, deadline):
    """
    Variable-neighborhood search on the block ordering.
    Neighborhoods: 2-opt (reverse segment), Or-opt (move segment).
    Accepts moves that strictly reduce total length.
    """
    best_order = list(order)
    best_len = _compute_length_from_order(best_order, n)
    
    def try_2opt(order, n):
        """Try 2-opt moves: reverse segment [i, j]."""
        best = list(order)
        best_len = _compute_length_from_order(best, n)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best) - 1):
                if time.time() > deadline:
                    break
                max_j = min(i + 50, len(best) - 1)
                for j in range(i + 1, max_j + 1):
                    if time.time() > deadline:
                        break
                    new_order = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_len = _compute_length_from_order(new_order, n)
                    if new_len < best_len:
                        best = new_order
                        best_len = new_len
                        improved = True
                        break
                if improved:
                    break
        return best, best_len
    
    def try_oropt(order, n):
        """Try Or-opt moves: move segment [i, i+k-1] to position j."""
        best = list(order)
        best_len = _compute_length_from_order(best, n)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for k in range(1, min(4, len(best) - i + 1)):
                    if time.time() > deadline:
                        break
                    segment = best[i:i+k]
                    rest = best[:i] + best[i+k:]
                    for j in range(len(rest) + 1):
                        if time.time() > deadline:
                            break
                        if j == i or j == i + k:
                            continue
                        new_order = rest[:j] + segment + rest[j:]
                        new_len = _compute_length_from_order(new_order, n)
                        if new_len < best_len:
                            best = new_order
                            best_len = new_len
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best, best_len
    
    best_order, best_len = try_2opt(best_order, n)
    if time.time() < deadline:
        best_order, best_len = try_oropt(best_order, n)
    if time.time() < deadline:
        best_order, best_len = try_2opt(best_order, n)
    
    return best_order, best_len

def _delete_pass(word, n, deadline):
    """
    Character-level deletion pass. Try to remove characters that are redundant.
    """
    S = list(word)
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
    
    best_len = len(S)
    
    while time.time() < deadline:
        made_removal = False
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            valid = True
            old_perms_to_dec = []
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid:
                continue
            
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 2 - n + 1)
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            S.pop(i)
            best_len = len(S)
            made_removal = True
        
        if not made_removal:
            break
    
    return S

def construct(n):
    start_time = time.time()
    deadline = start_time + 140
    
    if n <= 5:
        word = _standard_superperm(n)
        if time.time() < deadline:
            word = _delete_pass(word, n, deadline)
        return "".join(map(str, word))
    
    # For n >= 6: try multiple strategies, keep the best valid result
    
    best_word = None
    best_len = float('inf')
    
    # Strategy 1: Standard recursive construction + VNS + deletion
    base = _standard_superperm(n - 1)
    order = _extract_order_from_word(base, n - 1)
    
    if time.time() < deadline - 20:
        deadline_vns = deadline - 20
        order_opt, _ = _vns_improve_order(order, n, deadline_vns)
        word1 = _construct_from_order(order_opt, n)
        if time.time() < deadline - 5:
            word1 = _delete_pass(word1, n, deadline - 5)
        if _verify_superpermutation(word1, n):
            if len(word1) < best_len:
                best_word = word1
                best_len = len(word1)
    
    # Strategy 2: Greedy set-cover with lookahead
    if time.time() < deadline - 10:
        deadline_greedy = deadline - 10
        word2, valid2 = _greedy_with_backtrack(n, deadline_greedy)
        if valid2:
            if time.time() < deadline - 3:
                word2 = _delete_pass(word2, n, deadline - 3)
                if _verify_superpermutation(word2, n):
                    if len(word2) < best_len:
                        best_word = word2
                        best_len = len(word2)
    
    # Strategy 3: Simple greedy (faster, may be shorter for small n)
    if time.time() < deadline - 5:
        deadline_simple = deadline - 5
        word3, valid3 = _greedy_set_cover_superperm(n, deadline_simple)
        if valid3:
            if time.time() < deadline - 2:
                word3 = _delete_pass(word3, n, deadline - 2)
                if _verify_superpermutation(word3, n):
                    if len(word3) < best_len:
                        best_word = word3
                        best_len = len(word3)
    
    if best_word is None:
        # Fallback to standard
        word = _standard_superperm(n)
    else:
        word = best_word
    
    return "".join(map(str, word))

# Quick self-test
if __name__ == "__main__":
    for n in range(1, 6):
        w = construct(n)
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}")
    
    for n in [6, 7]:
        start = time.time()
        w = construct(n)
        elapsed = time.time() - start
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}, time={elapsed:.2f}s")
