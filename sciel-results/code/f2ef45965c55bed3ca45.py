import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 80.0  # 10s margin for the 90s wall-clock kill

    def time_left():
        return time_limit - (time.time() - start_time)

    # ---------- helpers ----------
    def compute_sums(seq):
        s = set()
        n = len(seq)
        for i in range(n):
            t = 0
            for j in range(i, n):
                t += seq[j]
                s.add(t)
        return s

    def count_dup(seq):
        """Number of excess occurrences of any block sum (0 iff all distinct)."""
        n = len(seq)
        c = {}
        for i in range(n):
            t = 0
            for j in range(i, n):
                t += seq[j]
                c[t] = c.get(t, 0) + 1
        dup = 0
        for v in c.values():
            if v > 1:
                dup += v - 1
        return dup

    # ---------- DFS (current design, kept as the structural core) ----------
    def dfs(start_val, seq, sums, suffix_sums):
        if time_left() <= 1.0:
            return
        if len(seq) > len(best_seq):
            best_seq[:] = seq
        # Prune: cannot beat best even taking every remaining value
        if len(seq) + (256 - start_val + 1) <= len(best_seq):
            return
        for x in range(start_val, 257):
            if time_left() <= 1.0:
                return
            new_suffix = [x]
            ok = True
            if x in sums:
                ok = False
            else:
                for s in suffix_sums:
                    ns = x + s
                    new_suffix.append(ns)
                    if ns in sums:
                        ok = False
                        break
            if not ok:
                continue
            dfs(x + 1, seq + [x], sums.union(new_suffix), new_suffix)

    # ---------- simulated annealing on the difference vector ----------
    def local_search(seq0, deadline):
        """Perturb the difference vector; accept moves by duplicate count (smooth
        surrogate for the nonsmooth 'all sums distinct' objective)."""
        rng = random.Random(12345 + int(time.time() * 1000) % 100000)
        cur = list(seq0)
        cur_dup = count_dup(cur)
        best_local = list(cur) if cur_dup == 0 else []
        best_local_len = len(best_local)
        T = 3.0
        cool = 0.99
        while time.time() < deadline:
            if T < 0.01:
                break
            n = len(cur)
            r = rng.random()
            if r < 0.45 and n > 1:
                # change one gap
                i = rng.randrange(n - 1)
                old = cur[i + 1] - cur[i]
                new = old + rng.choice([-2, -1, 1, 1, 2, 3])
                if new < 1:
                    continue
                nxt = cur[:]
                nxt[i + 1] = cur[i] + new
                for k in range(i + 2, n):
                    nxt[k] = nxt[k - 1] + (cur[k] - cur[k - 1])
                if nxt[-1] > 256:
                    continue
                nd = count_dup(nxt)
                if nd <= cur_dup or rng.random() < 0.25 * (1.0 - T / 3.0) and nd < cur_dup + 3:
                    cur, cur_dup = nxt, nd
                    if nd == 0 and len(cur) > best_local_len:
                        best_local, best_local_len = list(cur), len(cur)
            elif r < 0.7 and n > 0:
                # remove an element
                i = rng.randrange(n)
                nxt = cur[:i] + cur[i + 1:]
                nd = count_dup(nxt)
                if nd <= cur_dup:
                    cur, cur_dup = nxt, nd
            else:
                # insert an element at a random position
                i = rng.randrange(n + 1)
                lo = (cur[i - 1] + 1) if i > 0 else 1
                hi = (cur[i] - 1) if i < n else 256
                if hi < lo:
                    continue
                x = rng.randrange(lo, hi + 1)
                nxt = cur[:i] + [x] + cur[i:]
                nd = count_dup(nxt)
                if nd <= cur_dup or (nd < cur_dup + 2 and rng.random() < 0.15):
                    cur, cur_dup = nxt, nd
                    if nd == 0 and len(cur) > best_local_len:
                        best_local, best_local_len = list(cur), len(cur)
            T *= cool
        return best_local

    # ---------- main ----------
    best_seq = []

    # Phase 0: deterministic greedy for a solid baseline
    vals, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(vals):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            vals.append(x)
            sums.update(tails)
    best_seq = vals

    # Phase 1: DFS from greedy start (keeps the current search design)
    dfs(vals[0] + 1 if vals else 1, list(vals), compute_sums(vals),
        [sum(vals[len(vals)-i:]) for i in range(1, len(vals)+1)]) if vals else dfs(1, [], set(), [])

    # Phase 2: local search to escape local optima; feed valid longer results back as seeds
    deadline_ls = start_time + time_limit - 2.0
    improved = True
    while improved and time.time() < deadline_ls:
        improved = False
        ls = local_search(best_seq, deadline_ls)
        if ls and len(ls) > len(best_seq) and count_dup(ls) == 0:
            best_seq = ls
            improved = True
            # re-run DFS seeded from this longer sequence
            dfs(best_seq[-1] + 1, list(best_seq), compute_sums(best_seq),
                [sum(best_seq[len(best_seq)-i:]) for i in range(1, len(best_seq)+1)])

    # Final safety: return a guaranteed-valid sequence
    if best_seq and count_dup(best_seq) != 0:
        # fall back to the deterministic greedy (always valid)
        vals, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(vals):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                vals.append(x)
                sums.update(tails)
        best_seq = vals

    return best_seq

if __name__ == "__main__":
    print(construct())
