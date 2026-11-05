"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n):
    start = time.time()
    deadline = start + 138.0  # safety margin before the 150s wall-clock kill

    if n <= 1:
        return "1" if n == 1 else ""

    # ---- Classic recursive construction (valid for any n, fast for n<=6) ----
    def classic(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                w = tuple(word[i:i + m - 1])
                if len(set(w)) == m - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                ov = 0
                mx = min(len(new), len(block))
                for o in range(mx, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        ov = o
                        break
                new += block[ov:]
            word = new
        return word

    if n != 7:
        return "".join(map(str, classic(n)))

    # ---- Build SP(6) and the ordered list of 6-permutations ----
    word_6 = classic(6)
    perms_6, seen = [], set()
    for i in range(len(word_6) - 5):
        w = tuple(word_6[i:i + 6])
        if len(set(w)) == 6 and w not in seen:
            seen.add(w)
            perms_6.append(w)
    N = len(perms_6)  # 720

    # ---- Overlap matrix: M[i][j] = longest suffix of p_i that is a prefix of p_j ----
    M = [[0] * N for _ in range(N)]
    for i in range(N):
        a = perms_6[i]
        for j in range(N):
            b = perms_6[j]
            k = 0
            for kk in range(6, -1, -1):
                if kk == 0 or a[-kk:] == b[:kk]:
                    k = kk
                    break
            M[i][j] = k

    # ---- For a given ordering, maximize emit-lengths (suffix s_i, prefix f_i)
    # subject to the merge constraint s_i + f_{i+1} <= M[i][i+1].
    # Equivalent to a shortest-path DP over the chain. ----
    def optimize_emit(order):
        # DP over blocks; state = suffix length emitted by the last block (0..6).
        # dp[s] = best (max) total emit-length contribution up to current block,
        #         with the current block emitting suffix s, given its prefix already
        #         counted. We accumulate s_i + f_i for each block.
        INF = 10 ** 9
        # f_0 (prefix of first block) has no preceding constraint -> max 6.
        # We process block 0: choose s_0 (0..6); f_0 = 6 always (no constraint).
        # contribution of block i = f_i + s_i.
        # transition: for block i (i>=1), f_i <= M[order[i-1]][order[i]] - s_{i-1}.
        prev = [INF] * 7
        for s in range(7):
            prev[s] = 6 + s  # f_0=6, plus s_0
        for idx in range(1, N):
            cur = [INF] * 7
            mprev = M[order[idx - 1]][order[idx]]
            for s_prev in range(7):
                if prev[s_prev] >= INF:
                    continue
                base = prev[s_prev]
                # f in 0..min(6, mprev - s_prev)
                fmax = mprev - s_prev
                if fmax > 6:
                    fmax = 6
                if fmax < 0:
                    continue
                for s in range(7):
                    val = base + fmax + s  # f = fmax (best), plus s
                    if val > cur[s]:
                        cur[s] = val
            prev = cur
        # total = max over final suffix s (no trailing constraint)
        best = -INF
        for s in range(7):
            if prev[s] > best:
                best = prev[s]
        return best  # sum of (f_i + s_i)

    # cost(order) = 720*13 - 8 - 2*sum(s) - sum(f) = 9360 - 8 - (sum f + sum s)
    #            = 9352 - (sum f + sum s)
    def cost(order):
        return 9352 - optimize_emit(order)

    def build_word(order):
        # Reconstruct emit lengths greedily consistent with optimize_emit's optimum.
        # We just need a valid set of (s_i, f_i); re-run DP recording choices.
        INF = 10 ** 9
        prev = [INF] * 7
        for s in range(7):
            prev[s] = 6 + s
        # store best (f_i, s_{i-1}) achieving each state
        rec = [None] * N
        for idx in range(1, N):
            cur = [INF] * 7
            mprev = M[order[idx - 1]][order[idx]]
            rec_i = [None] * 7
            for s_prev in range(7):
                if prev[s_prev] >= INF:
                    continue
                base = prev[s_prev]
                fmax = mprev - s_prev
                if fmax > 6:
                    fmax = 6
                if fmax < 0:
                    continue
                for s in range(7):
                    val = base + fmax + s
                    if val > cur[s]:
                        cur[s] = val
                        rec_i[s] = (fmax, s_prev)
            prev = cur
            rec[idx] = rec_i
        # backtrack
        best_s = max(range(7), key=lambda s: prev[s])
        s_list = [0] * N
        f_list = [0] * N
        f_list[0] = 6
        s_list[0] = best_s
        for idx in range(N - 1, 0, -1):
            f_list[idx], s_list[idx - 1] = rec[idx][best_s]
            best_s = s_list[idx]
        # build word
        word = []
        for idx in range(N):
            p = perms_6[order[idx]]
            # block: prefix of p (f_i), then 7, then suffix of p (s_i)
            seg = list(p[:f_list[idx]]) + [7] + list(p[-s_list[idx]:] if s_list[idx] else [])
            if idx == 0:
                word = seg
            else:
                ov = 0
                mx = min(len(word), len(seg))
                for o in range(mx, -1, -1):
                    if word[len(word) - o:] == seg[:o]:
                        ov = o
                        break
                word += seg[ov:]
        return word

    # ---- Search over orderings ----
    def greedy_init():
        remaining = list(range(N))
        order = []
        # start from the natural first permutation
        order.append(0)
        remaining.remove(0)
        while remaining:
            last = order[-1]
            best_j = -1
            best_m = -1
            for j in remaining:
                if M[last][j] > best_m:
                    best_m = M[last][j]
                    best_j = j
            order.append(best_j)
            remaining.remove(best_j)
        return order

    def order_score(order):
        # quick proxy to guide search: sum of M along the chain
        return sum(M[order[i]][order[i + 1]] for i in range(N - 1))

    random.seed(12345)
    best_order = greedy_init()
    best_c = cost(best_order)

    # 2-opt polish on the order (uses exact cost, but that's O(N*7*7) ~ cheap)
    def two_opt(order, max_passes=4):
        cur = order[:]
        cur_c = cost(cur)
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 20:
            improved = False
            passes += 1
            for i in range(N - 1):
                if time.time() > deadline - 20:
                    return cur, cur_c
                j = i + 1
                # reverse segment [i..j]
                new = cur[:i] + cur[i:j + 1][::-1] + cur[j + 1:]
                nc = cost(new)
                if nc < cur_c:
                    cur = new
                    cur_c = nc
                    improved = True
        return cur, cur_c

    # Simulated-annealing on orderings using exact cost
    T = 30.0
    cur = best_order[:]
    cur_c = best_c
    while time.time() < deadline - 25:
        i = random.randint(0, N - 2)
        j = i + 1
        new = cur[:]
        new[i], new[j] = new[j], new[i]
        nc = cost(new)
        delta = nc - cur_c
        if delta <= 0 or random.random() < math.exp(-delta / T):
            cur = new
            cur_c = nc
            if cur_c < best_c:
                best_c = cur_c
                best_order = cur[:]
        T *= 0.9998
        if T < 0.5:
            break

    # final polish
    best_order, best_c = two_opt(best_order, max_passes=6)

    if time.time() > deadline - 8:
        # not enough time to build & verify safely; return classic (valid)
        return "".join(map(str, classic(7)))

    word = build_word(best_order)

    # ---- Verify: every permutation of 1..7 appears as 7 consecutive symbols ----
    if time.time() < deadline - 2:
        s = "".join(map(str, word))
        if len(s) < 5040 + 6:
            return s  # impossible to be valid; fall through
        found = set()
        for i in range(len(s) - 6):
            w = s[i:i + 7]
            if len(set(w)) == 7:
                found.add(w)
                if len(found) == 5040:
                    return s
        # not valid -> return classic
        return "".join(map(str, classic(7)))

    return "".join(map(str, word))
# EVOLVE-BLOCK-END
