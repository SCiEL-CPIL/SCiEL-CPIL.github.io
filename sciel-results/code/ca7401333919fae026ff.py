# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    DEADLINE = 78.0  # safety margin before the 90 s kill

    # Precompute 2,3,5-smooth numbers up to 128.
    smooth = []
    smooth_set = set()
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
            smooth_set.add(x)
    
    smooth.sort()
    answers = []

    # Precompute antichain validity for pairs
    # div_free[a][b] = True if a does not divide b and b does not divide a (and a != b)
    # Actually we just need: for sorted terms, no earlier term divides a later term
    # Let's precompute: for each pair (i,j) with i<j, does smooth[i] divide smooth[j]?
    n_smooth = len(smooth)
    divides = [[False]*n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if smooth[j] % smooth[i] == 0:
                divides[i][j] = True

    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            answers.append([n])
            continue

        # 1. Check 1 term
        if n in smooth_set:
            answers.append([n])
            continue

        # 2. Check 2 terms
        found = False
        for i, a in enumerate(smooth):
            if a >= n:
                break
            b = n - a
            if b <= a:
                continue
            if b in smooth_set:
                j = smooth.index(b)
                # Check antichain: a does not divide b
                if not divides[i][j]:
                    answers.append([a, b])
                    found = True
                    break
        if found:
            continue

        # 3. Check 3 terms
        found = False
        for i, a in enumerate(smooth):
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c in smooth_set:
                    k = smooth.index(c)
                    # Check antichain: no earlier divides later
                    if not divides[i][j] and not divides[i][k] and not divides[j][k]:
                        answers.append([a, b, c])
                        found = True
                        break
            if found:
                break
        if found:
            continue

        # 4. Check 4 terms
        found = False
        for i, a in enumerate(smooth):
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth[k]
                    if a + b + c >= n:
                        break
                    d = n - a - b - c
                    if d <= c:
                        continue
                    if d in smooth_set:
                        l = smooth.index(d)
                        # Check antichain
                        if not divides[i][j] and not divides[i][k] and not divides[i][l] \
                           and not divides[j][k] and not divides[j][l] and not divides[k][l]:
                            answers.append([a, b, c, d])
                            found = True
                            break
                    if found: break
                if found: break
            if found: break
        if found:
            continue

        # 5. Generic DFS fallback for 5+ terms
        def dfs(remaining, last_idx, current_terms, term_indices):
            if remaining == 0:
                return current_terms
            for idx in range(last_idx + 1, n_smooth):
                val = smooth[idx]
                if val > remaining:
                    break
                # Check antichain with all existing terms
                valid = True
                for ti in term_indices:
                    if divides[ti][idx]:
                        valid = False
                        break
                if valid:
                    res = dfs(remaining - val, idx, current_terms + [val], term_indices + [idx])
                    if res:
                        return res
            return None

        res = dfs(n, -1, [], [])
        if res:
            answers.append(res)
        else:
            answers.append([n])

    return answers
# EVOLVE-BLOCK-END
