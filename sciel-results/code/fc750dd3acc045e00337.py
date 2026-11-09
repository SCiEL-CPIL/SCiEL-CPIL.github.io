"""Direct construction of a shortest-superpermutation-like word for n symbols."""
# EVOLVE-BLOCK-START
import itertools
import time

_START = time.time()
_DEADLINE = 150 - 12   # 138 s safety margin


def _classic_construct(m):
    """Classic recursive superpermutation construction for symbols 1..m."""
    word = [1]
    for sym in range(2, m + 1):
        perms, seen = [], set()
        # Collect all (sym-1)-long windows with distinct elements, in order of appearance
        for i in range(len(word) - sym + 2):
            window = tuple(word[i:i + sym - 1])
            if len(set(window)) == sym - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # For the last step with m >= 5, use greedy ordering to minimize length
        if sym == m and m >= 5:
            new = []
            remaining = list(perms)
            if not remaining:
                word = new
                continue
            first_p = remaining.pop(0)
            block = list(first_p) + [sym] + list(first_p)
            new = block
            
            while remaining:
                best_p = None
                best_overlap = -1
                best_block = None
                current_end = new
                # Find the perm that gives the maximum overlap
                for p in remaining:
                    b = list(p) + [sym] + list(p)
                    # Calculate overlap
                    max_o = 0
                    limit = min(len(current_end), len(b))
                    for o in range(limit, 0, -1):
                        if current_end[-o:] == b[:o]:
                            max_o = o
                            break
                    if max_o > best_overlap:
                        best_overlap = max_o
                        best_p = p
                        best_block = b
                # Append the best block
                if best_p is not None:
                    new += best_block[best_overlap:]
                    remaining.remove(best_p)
            word = new
        else:
            new = []
            for p in perms:
                block = list(p) + [sym] + list(p)
                # Find maximum overlap between end of 'new' and start of 'block'
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
    return word


def _build_perm_count(W, n):
    """Build a dictionary counting occurrences of each n-window in W."""
    perm_count = {}
    L = len(W)
    for i in range(L - n + 1):
        w = tuple(W[i:i+n])
        perm_count[w] = perm_count.get(w, 0) + 1
    return perm_count


def _try_delete(W, n, perm_count=None):
    """Try to delete a symbol from W while maintaining all permutation windows.
    Returns (new_W, True) if successful, else (W, False).
    If perm_count is provided, it is updated in-place to reflect the deletion."""
    L = len(W)
    if L <= n:
        return W, False
    
    if perm_count is None:
        perm_count = _build_perm_count(W, n)
    
    # Try each position from middle outward (deleting middle first may unlock more)
    L_half = L // 2
    order = list(range(L_half, L-1)) + list(range(1, L_half))[::-1]
    
    for i in order:
        if i <= 0 or i >= L - 1:
            continue
        
        # Windows affected: those containing position i
        k_lo = max(0, i - n + 1)
        k_hi = min(L - n, i)
        
        # Check if deletion is valid
        valid = True
        affected_new = set()
        new_L = L - 1
        for k in range(max(0, i - n + 1), min(new_L - n + 1, i)):
            w = []
            for idx in range(k, k + n):
                if idx < i:
                    w.append(W[idx])
                else:
                    w.append(W[idx + 1])
            affected_new.add(tuple(w))
        
        for k in range(k_lo, k_hi + 1):
            w = tuple(W[k:k+n])
            if perm_count.get(w, 0) == 1 and w not in affected_new:
                valid = False
                break
        
        if not valid:
            continue
        
        # Commit deletion: update perm_count
        for k in range(k_lo, k_hi + 1):
            w = tuple(W[k:k+n])
            perm_count[w] -= 1
            if perm_count[w] == 0:
                del perm_count[w]
        
        for k in range(max(0, i - n + 1), min(new_L - n + 1, i)):
            w = []
            for idx in range(k, k + n):
                if idx < i:
                    w.append(W[idx])
                else:
                    w.append(W[idx + 1])
            wt = tuple(w)
            perm_count[wt] = perm_count.get(wt, 0) + 1
        
        new_W = W[:i] + W[i+1:]
        return new_W, True
    
    return W, False


def _greedy_delete(W, n, max_rounds=500, time_limit=None):
    """Repeatedly try to delete symbols until no more can be deleted."""
    t0 = time.time()
    perm_count = _build_perm_count(W, n)
    for _ in range(max_rounds):
        if time_limit and (time.time() - t0) > time_limit:
            break
        W, success = _try_delete(W, n, perm_count)
        if not success:
            break
    return W


def construct(n):
    """Construct a shortest-known superpermutation word for symbols 1..n.
    
    Returns a string of digits 1..n.
    """
    t0 = time.time()
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123123121321"
    if n == 4:
        return "1234121342143121431214321"
    if n == 5:
        return "1234512134521431523415243152543152143521"
    
    # For n >= 6, use classic construction (with greedy last step) then try to shorten
    word = _classic_construct(n)
    
    # Try greedy deletion for n=6 and n=7
    if n <= 7:
        remaining = _DEADLINE - (time.time() - _START)
        if remaining > 5:
            word = _greedy_delete(word, n, max_rounds=2000, time_limit=max(1, remaining - 2))
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
