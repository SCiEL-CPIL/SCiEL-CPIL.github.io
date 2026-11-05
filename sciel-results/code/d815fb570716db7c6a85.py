"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n <= 6:
        return _classic_superperm(n)
    
    k = n - 1
    word_k = _classic_superperm_list(k)
    
    # Extract unique permutations of length k from word_k in order of first appearance
    perms_k = []
    seen = set()
    for i in range(len(word_k) - k + 1):
        window = tuple(word_k[i:i + k])
        if window not in seen:
            if sorted(window) == list(range(1, k + 1)):
                seen.add(window)
                perms_k.append(window)
    
    num_perms = len(perms_k)
    expected = 1
    for x in range(1, k + 1):
        expected *= x
    if num_perms != expected:
        return _classic_superperm(n)
    
    # Precompute overlap matrix M[i][j] = longest suffix of perms_k[i] matching prefix of perms_k[j]
    M = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_k[i]
        for j in range(num_perms):
            if i == j:
                M[i][j] = k
                continue
            p2 = perms_k[j]
            max_ov = k - 1
            for o in range(max_ov, 0, -1):
                if p1[k - o:] == p2[:o]:
                    M[i][j] = o
                    break
    
    # Helper: compute total overlap score for an order
    def total_overlap(ord):
        s = 0
        for idx in range(len(ord) - 1):
            s += M[ord[idx]][ord[idx + 1]]
        return s
    
    # Helper: construct word from an order, using actual overlap computation
    def build_word(ord):
        word = []
        for idx, p_idx in enumerate(ord):
            p = perms_k[p_idx]
            block = list(p) + [n] + list(p)
            if idx == 0:
                word = block
            else:
                # Compute actual overlap: longest suffix of word matching prefix of block
                max_check = min(len(word), len(block))
                ov = 0
                for o in range(max_check, 0, -1):
                    if word[-o:] == block[:o]:
                        ov = o
                        break
                word += block[ov:]
        return word
    
    # Local search with incremental delta
    best_order = list(range(num_perms))
    best_score = total_overlap(best_order)
    
    random.seed(42)
    
    def try_move(cur_order, cur_score, i, j, move_type):
        """Try a move and return (new_order, new_score, delta) or None if invalid."""
        if move_type == 'swap':
            if i >= len(cur_order) - 1:
                return None
            a, b = i, i + 1
            pa, pb = cur_order[a], cur_order[b]
            # Remove contributions involving a and b
            delta = 0
            if a > 0:
                delta -= M[cur_order[a - 1]][pa]
            delta -= M[pa][pb]
            if b < len(cur_order) - 1:
                delta -= M[pb][cur_order[b + 1]]
            # Add new contributions
            if a > 0:
                delta += M[cur_order[a - 1]][pb]
            delta += M[pb][pa]
            if b < len(cur_order) - 1:
                delta += M[pa][cur_order[b + 1]]
            new_score = cur_score + delta
            new_order = cur_order[:]
            new_order[a], new_order[b] = new_order[b], new_order[a]
            return new_order, new_score, delta
        elif move_type == 'move':
            if i == j:
                return None
            item = cur_order[i]
            # Remove item from position i
            delta = 0
            if i > 0:
                delta -= M[cur_order[i - 1]][item]
            if i < len(cur_order) - 1:
                delta -= M[item][cur_order[i + 1]]
            # If i < j, after removal, item goes to position j-1 (before original j)
            # If i > j, after removal, item goes to position j
            # Let's just compute by constructing new order and scoring affected region
            # For simplicity with move, do partial recalculation
            new_order = cur_order[:]
            new_order.pop(i)
            # Adjust j if i < j
            if i < j:
                j -= 1
            new_order.insert(j, item)
            # Score only the affected positions: around i and around j in new_order
            # Positions affected in new_order: max(0, min(i, j) - 1) to min(len-1, max(i, j))
            lo = max(0, min(i, j) - 1)
            hi = min(len(new_order) - 1, max(i, j))
            # Compute old score contribution for these positions in cur_order
            old_contrib = 0
            for p in range(lo, hi):
                if p < len(cur_order) - 1:
                    old_contrib += M[cur_order[p]][cur_order[p + 1]]
            # But this is tricky because the mapping between old and new positions is complex.
            # For 'move', just do full recalc of the small region or full score.
            # Given N=720, full recalc is ~719 lookups. Let's just do it.
            new_score = total_overlap(new_order)
            return new_order, new_score, new_score - cur_score
        return None
    
    current_order = best_order[:]
    current_score = best_score
    
    while time.time() < deadline:
        if random.random() < 0.5:
            i = random.randint(0, num_perms - 2)
            res = try_move(current_order, current_score, i, i, 'swap')
        else:
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j:
                continue
            res = try_move(current_order, current_score, i, j, 'move')
        
        if res is None:
            continue
        new_order, new_score, _ = res
        
        if new_score > best_score:
            best_score = new_score
            best_order = new_order
        if new_score >= current_score:
            current_order = new_order
            current_score = new_score
    
    # Build final word with actual overlap verification
    final_word = build_word(best_order)
    
    # Verify it's a valid superpermutation
    if not _verify_superperm(final_word, n):
        # Fallback to classic
        return _classic_superperm(n)
    
    return "".join(map(str, final_word))

def _verify_superperm(word, n):
    """Verify that word contains every permutation of 1..n as a consecutive substring."""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n and min(window) == 1 and max(window) == n:
            seen.add(window)
    return len(seen) == _factorial(n)

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def _classic_superperm_list(n):
    """Returns the classic superpermutation for n as a list of integers."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _classic_superperm(n):
    """Returns the classic superpermutation for n as a string."""
    return "".join(map(str, _classic_superperm_list(n)))

# EVOLVE-BLOCK-END
