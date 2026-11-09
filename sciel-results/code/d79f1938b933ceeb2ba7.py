import time
import numpy as np

_DEADLINE = time.time() + 125.0


def _colors_to_matrix(colors, n):
    m = np.zeros((n, n), dtype=np.int8)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = colors[k]
            m[j, i] = colors[k]
            k += 1
    return m


def _has_c4(m, c):
    """Return True if colour class c contains a 4-cycle."""
    A = (m == c).astype(np.int8)
    np.fill_diagonal(A, 0)
    P = A @ A
    return bool((P >= 2).any())


def _validate(colors, n):
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    m = _colors_to_matrix(colors, n)
    for c in range(5):
        if _has_c4(m, c):
            return False
    return True


def _build_n26():
    """Lazebnik-Woldar construction for n=26."""
    p = 5
    q = 5
    n = p * q
    # Quadratic residues mod 5: 1^2=1, 2^2=4, 3^2=9=4, 4^2=16=1
    # So residues of x^2 mod 5 for x in {1,2,3,4} are {1, 4}
    # r[d] = 1 if d is a non-zero quadratic residue mod 5
    r = [0] * p
    for x in range(1, p):
        r[x * x % p] = 1
    # r[0]=0, r[1]=1, r[2]=0, r[3]=0, r[4]=1
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            a = i % p
            b = j % p
            if a == b:
                c = 0
            else:
                x = ((j // q) - (i // q)) % q
                c = 1 + r[(a - b) * x % p]
            colors.append(c)
    return n, colors


def _total_c4(m, n):
    """Count total number of C4s across all colour classes."""
    s = 0
    for c in range(5):
        A = (m == c).astype(np.int8)
        np.fill_diagonal(A, 0)
        P = A @ A
        s += int((P >= 2).sum())
    return s


def _greedy_extend(m, best_n, target_n, rng, time_limit):
    """
    Try to extend the coloring from best_n to target_n using greedy edge assignment
    that minimizes new C4s. Returns (success, final_n, final_m).
    """
    m = m.copy()
    m = np.pad(m, ((0, target_n - best_n), (0, target_n - best_n)), mode='constant')
    cur_n = best_n

    # Assign edges involving new vertices
    for k in range(cur_n, target_n):
        for i in range(k):
            if time.time() > time_limit:
                break
            # Try each colour, pick the one that minimizes total C4s
            best_c = 0
            best_c4 = float('inf')
            for c in range(5):
                m[i, k] = c
                m[k, i] = c
                # Quick check: does this create a C4?
                # A C4 in color c through edge (i,k) exists if i and k share >=2 common neighbors in color c
                # Check common neighbors of i and k in color c
                common = 0
                for j in range(k):
                    if j != i:
                        if m[i, j] == c and m[k, j] == c:
                            common += 1
                            if common >= 2:
                                break
                if common < 2:
                    # No immediate C4, good candidate
                    # Count total C4s (expensive but accurate)
                    c4_count = _total_c4(m, target_n)
                    if c4_count < best_c4:
                        best_c4 = c4_count
                        best_c = c
                else:
                    # This colour creates a C4; still track it but prefer non-C4
                    c4_count = _total_c4(m, target_n)
                    if c4_count < best_c4:
                        best_c4 = c4_count
                        best_c = c
            m[i, k] = best_c
            m[k, i] = best_c
        if time.time() > time_limit:
            break
        cur_n = k + 1

    if cur_n < target_n:
        # Timed out; work with what we have
        m = m[:cur_n, :cur_n]
        return _total_c4(m, cur_n) == 0, cur_n, m

    final_c4 = _total_c4(m, target_n)
    return final_c4 == 0, target_n, m


def construct():
    t0 = time.time()

    # Start with the known n=26 construction
    n26, colors26 = _build_n26()
    if not _validate(colors26, n26):
        # Should not happen, but fallback
        return {'n': 11,
                'colors': [min(j - i, 11 - j + i) - 1
                           for i in range(11) for j in range(i + 1, 11)]}

    best_n = n26
    best_colors = colors26
    best_m = _colors_to_matrix(colors26, n26)

    rng = np.random.default_rng(20240517)

    # Try to extend to n=27
    if time.time() < _DEADLINE - 40.0:
        success, new_n, new_m = _greedy_extend(best_m, best_n, 27, rng, _DEADLINE - 40.0)
        if success and new_n == 27:
            best_n = 27
            best_colors = [int(new_m[i, j]) for i in range(27) for j in range(i + 1, 27)]
            best_m = new_m

    # If we got 27, try to extend to 28
    if best_n == 27 and time.time() < _DEADLINE - 25.0:
        success, new_n, new_m = _greedy_extend(best_m, best_n, 28, rng, _DEADLINE - 25.0)
        if success and new_n == 28:
            best_n = 28
            best_colors = [int(new_m[i, j]) for i in range(28) for j in range(i + 1, 28)]
            best_m = new_m

    # If we didn't reach 27 via greedy, try local search from n=26 to fix/extend
    if best_n < 27 and time.time() < _DEADLINE - 15.0:
        # Try local search to find a valid n=27 coloring
        # Start from n=26, add vertex 26 with random colours, then repair
        m = best_m.copy()
        m = np.pad(m, ((0, 1), (0, 1)), mode='constant')
        for i in range(26):
            m[i, 26] = rng.integers(0, 5)
            m[26, i] = m[i, 26]

        cur_c4 = _total_c4(m, 27)
        no_improve = 0
        max_iter = 100000
        it = 0
        while it < max_iter and no_improve < 20000 and time.time() < _DEADLINE - 10.0:
            it += 1
            i = rng.integers(0, 27)
            j = rng.integers(0, 27)
            if i == j:
                continue
            old = m[i, j]
            new = rng.integers(0, 5)
            if new == old:
                continue
            m[i, j] = new
            m[j, i] = new
            new_c4 = _total_c4(m, 27)
            if new_c4 < cur_c4:
                cur_c4 = new_c4
                no_improve = 0
                if cur_c4 == 0:
                    break
            else:
                m[i, j] = old
                m[j, i] = old
                no_improve += 1

        if cur_c4 == 0:
            best_n = 27
            best_colors = [int(m[i, j]) for i in range(27) for j in range(i + 1, 27)]
            best_m = m

    # Final validation and return
    if time.time() > _DEADLINE - 3.0:
        return {'n': best_n, 'colors': best_colors}

    # Double-check validity
    if _validate(best_colors, best_n):
        return {'n': best_n, 'colors': best_colors}
    else:
        # Fallback to n=26
        return {'n': n26, 'colors': colors26}
