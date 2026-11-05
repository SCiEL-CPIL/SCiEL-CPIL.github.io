import time

MAX_N = 128
smooth = []
for x in range(1, MAX_N + 1):
    y = x
    for p in (2,3,5):
        while y % p == 0:
            y //= p
    if y == 1:
        smooth.append(x)
smooth_set = set(smooth)
smooth_list = sorted(smooth)

def antichain(terms):
    for i in range(len(terms)):
        for j in range(i+1, len(terms)):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

best = [None] * (MAX_N + 1)
start = time.time()
LIMIT = 75.0

for n in range(1, MAX_N + 1):
    if time.time() - start > LIMIT:
        break
    # k=1
    if n in smooth_set:
        best[n] = [n]
        continue
    # k=2
    found = False
    for i in range(len(smooth_list)):
        a = smooth_list[i]
        if a >= n:
            break
        b = n - a
        if b <= a:
            continue
        if b in smooth_set and b % a != 0:
            best[n] = [a, b]
            found = True
            break
    if found:
        continue
    # k=3
    found = False
    for i in range(len(smooth_list)):
        a = smooth_list[i]
        if a >= n:
            break
        for j in range(i+1, len(smooth_list)):
            b = smooth_list[j]
            if a + b >= n:
                break
            c = n - a - b
            if c <= b:
                continue
            if c not in smooth_set:
                continue
            if b % a == 0 or c % a == 0 or c % b == 0:
                continue
            best[n] = [a, b, c]
            found = True
            break
        if found:
            break
    if found:
        continue
    # k=4
    found = False
    for i in range(len(smooth_list)):
        a = smooth_list[i]
        if a >= n:
            break
        for j in range(i+1, len(smooth_list)):
            b = smooth_list[j]
            if a + b >= n:
                break
            for k in range(j+1, len(smooth_list)):
                c = smooth_list[k]
                if a + b + c >= n:
                    break
                d = n - a - b - c
                if d <= c:
                    continue
                if d not in smooth_set:
                    continue
                # antichain check for a<b<c<d
                if b % a == 0 or c % a == 0 or d % a == 0:
                    continue
                if c % b == 0 or d % b == 0:
                    continue
                if d % c == 0:
                    continue
                best[n] = [a, b, c, d]
                found = True
                break
            if found:
                break
        if found:
            break
    if found:
        continue
    # k=5 via DFS
    def dfs(idx, cur, s):
        if len(cur) == 5:
            if s == n and antichain(cur):
                return cur[:]
            return None
        for t in range(idx, len(smooth_list)):
            v = smooth_list[t]
            if s + v > n:
                break
            if v in cur:
                continue
            # antichain prune
            ok = True
            for e in cur:
                x, y = min(e, v), max(e, v)
                if y % x == 0:
                    ok = False
                    break
            if not ok:
                continue
            res = dfs(t+1, cur + [v], s + v)
            if res:
                return res
        return None
    res = dfs(0, [], 0)
    if res:
        best[n] = res
    else:
        # fallback: try k up to 30 with combinations (slow but safe)
        from itertools import combinations
        for cnt in range(5, 31):
            if time.time() - start > LIMIT:
                break
            for terms in combinations([x for x in smooth_list if x <= n], cnt):
                if sum(terms) == n and antichain(list(terms)):
                    best[n] = list(terms)
                    break
            if best[n]:
                break

# final safety fill
for n in range(1, MAX_N + 1):
    if best[n] is None:
        # last resort: brute force small
        from itertools import combinations
        for cnt in range(1, 31):
            for terms in combinations([x for x in smooth_list if x <= n], cnt):
                if sum(terms) == n and antichain(list(terms)):
                    best[n] = list(terms)
                    break
            if best[n]:
                break

def construct():
    return [best[i] for i in range(1, MAX_N + 1)]
