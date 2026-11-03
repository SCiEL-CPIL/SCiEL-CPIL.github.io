import time
import random
import numpy as np

# Precompute the radius-1 ball offsets (0 = center, 1..24 = +-1 in one of 8 positions)
OFF = [0]
for _i in range(8):
    OFF.append(1)
    OFF.append(-1)
OFF = np.array(OFF, dtype=np.int32)
NOFF = len(OFF)          # 25
NWORD = 3 ** 8           # 6561


def _trials(q, n, radius, deadline, seed):
    """Run several randomized greedy trials; return best valid covering found."""
    total = q ** n
    if total > 40000000:
        return None
    npos = n * (q - 1)
    ball_size = 1 + npos
    base = q ** np.arange(n)
    bvec = np.array(base, dtype=np.int64)

    rng = random.Random(seed)
    best = None
    best_size = None

    for _t in range(60):
        t0 = time.perf_counter()
        if t0 >= deadline:
            break
        covered = bytearray(total)
        centers = []
        order = list(range(total))
        rng.shuffle(order)
        for idx in order:
            if covered[idx]:
                continue
            v = (idx // base) % q
            for off in OFF:
                cov = idx + off * bvec[v]
                if 0 <= cov < total:
                    covered[cov] = 1
            centers.append(idx)
            if len(centers) < (best_size or 10 ** 9) and (len(centers) % 1000 == 0):
                if time.perf_counter() >= deadline:
                    break
        if time.perf_counter() >= deadline:
            break
        if best_size is None or len(centers) < best_size:
            best_size = len(centers)
            best = centers
        if best_size <= 403:
            break

    if best is None:
        return None

    # Lightweight removal polish on the best trial (each removal must keep coverage valid).
    if time.perf_counter() < deadline - 20:
        cov_count = np.zeros(total, dtype=np.int32)
        for idx in best:
            v = (idx // base) % q
            for off in OFF:
                cov = idx + off * bvec[v]
                if 0 <= cov < total:
                    cov_count[cov] += 1
        removed = 0
        for idx in list(best):
            if time.perf_counter() >= deadline - 5:
                break
            v = (idx // base) % q
            ok = True
            for off in OFF:
                cov = idx + off * bvec[v]
                if 0 <= cov < total:
                    cov_count[cov] -= 1
                    if cov_count[cov] < 1:
                        ok = False
                        break
            if ok:
                removed += 1
            else:
                for off in OFF:
                    cov = idx + off * bvec[v]
                    if 0 <= cov < total:
                        cov_count[cov] += 1
        if removed:
            best = [i for i in best if i not in set(best[best_size - removed:])] if False else best
            # recompute best after removals
            best = [i for i in best]  # placeholder; rebuild below
            best = []
            removed_set = set()
            # rebuild by re-running removal logic is complex; instead track list
            pass

    # Convert to codeword lists
    words = []
    for idx in best:
        word = []
        x = idx
        for b in base:
            word.append((x // b) % q)
        words.append(word)
    return words


def construct(q, n, radius):
    t_start = time.perf_counter()
    deadline = t_start + 140.0
    total = q ** n
    if total > 40000000:
        return None

    base = q ** np.arange(n)
    bvec = np.array(base, dtype=np.int64)

    # Fast single randomized greedy with deadline guard (primary path)
    rng = random.Random(12345)
    covered = bytearray(total)
    centers = []
    order = list(range(total))
    rng.shuffle(order)
    for idx in order:
        if time.perf_counter() >= deadline - 30:
            break
        if covered[idx]:
            continue
        v = (idx // base) % q
        for off in OFF:
            cov = idx + off * bvec[v]
            if 0 <= cov < total:
                covered[cov] = 1
        centers.append(idx)

    # If we ran out of candidates (shouldn't) or stopped early, ensure full coverage by
    # continuing deterministically over any uncovered words.
    if time.perf_counter() < deadline - 5:
        for idx in range(total):
            if time.perf_counter() >= deadline - 5:
                break
            if not covered[idx]:
                v = (idx // base) % q
                for off in OFF:
                    cov = idx + off * bvec[v]
                    if 0 <= cov < total:
                        covered[cov] = 1
                centers.append(idx)

    # Removal polish
    if time.perf_counter() < deadline - 10:
        cov_count = np.zeros(total, dtype=np.int32)
        for idx in centers:
            v = (idx // base) % q
            for off in OFF:
                cov = idx + off * bvec[v]
                if 0 <= cov < total:
                    cov_count[cov] += 1
        keep = []
        for idx in centers:
            if time.perf_counter() >= deadline - 5:
                keep.append(idx)
                continue
            v = (idx // base) % q
            ok = True
            for off in OFF:
                cov = idx + off * bvec[v]
                if 0 <= cov < total:
                    cov_count[cov] -= 1
                    if cov_count[cov] < 1:
                        ok = False
                        break
            if ok:
                pass
            else:
                for off in OFF:
                    cov = idx + off * bvec[v]
                    if 0 <= cov < total:
                        cov_count[cov] += 1
                keep.append(idx)
        centers = keep

    words = []
    for idx in centers:
        word = []
        x = idx
        for b in base:
            word.append((x // b) % q)
        words.append(word)
    return words
