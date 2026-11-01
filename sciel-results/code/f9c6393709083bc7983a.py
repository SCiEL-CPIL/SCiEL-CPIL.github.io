# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_TIME_BUDGET = 138.0


def construct(v, k, t):
    t0 = time.time()

    combos = list(itertools.combinations(range(v), t))
    ncomb = len(combos)
    nwords = (ncomb + 63) // 64
    word_of = np.arange(ncomb) // 64
    bit_of = np.arange(ncomb) % 64

    if t == 3:
        a = np.array([c[0] for c in combos], dtype=np.int32)
        b = np.array([c[1] for c in combos], dtype=np.int32)
        d = np.array([c[2] for c in combos], dtype=np.int32)
        i3 = np.empty((v, v, v), dtype=np.int32)
        i3[a, b, d] = np.arange(ncomb, dtype=np.int32)

        def idx_of(s):
            return int(i3[s[0], s[1], s[2]])
    else:
        # general fallback: build a dict
        dmap = {c: i for i, c in enumerate(combos)}

        def idx_of(s):
            return int(dmap[tuple(sorted(s))])

    def make_uncovered():
        m = np.ones(nwords, dtype=np.uint8)
        rem = ncomb % 64
        m.flat[nwords - 1] = (1 << rem) - 1 if rem else 255
        return m

    def popcount_u8(m):
        # m is uint8 array
        return int(sum(bin(int(x)).count("1") for x in m))

    def greedy(seed):
        rng = np.random.default_rng(seed)
        uncovered = make_uncovered()
        blocks = []
        while True:
            # find first set bit
            nz = np.nonzero(uncovered)[0]
            if nz.size == 0:
                break
            w = int(nz[0])
            word = int(uncovered[w])
            local = (word & -word)
            lbit = int(np.log2(local)) if local else 0
            base = w * 64 + lbit
            if base >= ncomb:
                break
            block = list(combos[base])
            while len(block) < k:
                inb = np.zeros(v, dtype=bool)
                inb[block] = True
                cands = np.where(~inb)[0]
                if cands.size == 0:
                    break
                if t == 3:
                    gain = np.zeros(v, dtype=np.int32)
                    for x, y in itertools.combinations(block, 2):
                        gain += np.bitwise_count(uncovered[x, y, :]) if hasattr(np, "bitwise_count") else popcount_u8(uncovered[x, y, :])
                else:
                    gain = np.zeros(v, dtype=np.int32)
                    for rest in itertools.combinations(block, t - 1):
                        for x in cands:
                            tri = tuple(sorted(rest + (int(x),)))
                            idx = idx_of(tri)
                            if (uncovered[word_of[idx]] >> bit_of[idx]) & 1:
                                gain[x] += 1
                r = rng.uniform(0.0, 1.0, size=v)
                key = gain.astype(np.float64) + r
                key[inb] = -1.0
                x = int(np.argmax(key))
                block.append(x)
            block.sort()
            blocks.append(block)
            for s in itertools.combinations(block, t):
                idx = idx_of(s)
                uncovered[word_of[idx]] &= ~(1 << bit_of[idx])
            if time.time() - t0 > _TIME_BUDGET:
                return None  # timed out mid-run; discard
        return blocks

    best = None
    best_len = 10 ** 9
    seed = 0
    while time.time() - t0 < _TIME_BUDGET:
        blocks = greedy(seed)
        if blocks is not None and len(blocks) < best_len:
            cov = np.zeros(nwords, dtype=np.uint8)
            for blk in blocks:
                for s in itertools.combinations(blk, t):
                    idx = idx_of(s)
                    cov[word_of[idx]] |= (1 << bit_of[idx])
            if bool(np.all(cov != 0)):
                best = blocks
                best_len = len(blocks)
        seed += 1
    if best is None:
        best = greedy(0)
    return best
# EVOLVE-BLOCK-END
