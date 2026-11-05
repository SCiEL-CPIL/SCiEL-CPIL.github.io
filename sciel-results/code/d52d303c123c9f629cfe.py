import itertools
import random
import time
import math
import sys

sys.setrecursionlimit(10000)

# ---------------- Precomputation ----------------
V = 19
K = 7
T = 4

subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)          # 3876
FULL_MASK = (1 << N_SUBSETS) - 1

all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
block_subsets = [[] for _ in range(N_BLOCKS)]  # placeholder, filled below

for block in itertools.combinations(range(V), K):
    mask = 0
    subs = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subs.append(idx)
    all_blocks.append(block)
    all_masks.append(mask)
    block_subsets.append(subs)

N_BLOCKS = len(all_blocks)
block_sizes = [len(s) for s in block_subsets]  # each = C(7,4)=35

# blocks sorted by index (default) for determinism
block_order = list(range(N_BLOCKS))


def _popcount(x):
    return x.bit_count()


def _iter_bits(x):
    while x:
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb


# ---------------- Packing-first construction ----------------
def packing_construct(deadline):
    """Greedy packing: pick blocks maximizing NEW distinct 4-subsets.
    Strict collision avoidance via the uncovered mask (a block is only
    chosen if it adds at least one new subset)."""
    counts = [0] * N_SUBSETS
    chosen = []
    uncovered = FULL_MASK

    while uncovered:
        if time.time() > deadline:
            break
        best = -1
        best_gain = 0
        for i in block_order:
            g = _popcount(all_masks[i] & uncovered)
            if g > best_gain:
                best_gain = g
                best = i
                if g == block_sizes[best]:
                    break  # can't do better than full 35
        if best == -1 or best_gain == 0:
            break
        chosen.append(best)
        uncovered &= ~all_masks[best]
        for b in _iter_bits(all_masks[best]):
            counts[b] += 1

    return chosen, counts, uncovered


def repair_construct(chosen, counts, uncovered, deadline):
    """Targeted repair: cover remaining uncovered 4-subsets with blocks that
    cover the *hardest* (least-covered) uncovered subsets, minimizing waste."""
    chosen = list(chosen)
    counts = list(counts)
    cur_uncovered = uncovered

    while cur_uncovered:
        if time.time() > deadline:
            break
        # pick the hardest uncovered subset (fewest blocks can cover it)
        target = -1
        target_deg = 10**9
        for b in _iter_bits(cur_uncovered):
            d = len(blocks_containing_subset[b])
            if d < target_deg:
                target_deg = d
                target = b
                if d == 1:
                    break
        if target == -1:
            break

        # among blocks covering target, pick the one with best (new gain / size)
        best = -1
        best_score = -1.0
        for c in blocks_containing_subset[target]:
            g = _popcount(all_masks[c] & cur_uncovered)
            if g == 0:
                continue
            # prefer blocks that also cover other hard subsets
            score = g / block_sizes[c]
            if score > best_score:
                best_score = score
                best = c
        if best == -1:
            break
        chosen.append(best)
        cur_uncovered &= ~all_masks[best]
        for b in _iter_bits(all_masks[best]):
            counts[b] += 1

    return chosen, counts, cur_uncovered


# ---------------- Local search ----------------
def local_search(chosen, counts, deadline):
    """Remove redundant blocks, then try swaps and 2-removals to shrink."""
    best = list(chosen)
    best_len = len(best)
    cur = list(chosen)
    cur_counts = list(counts)
    cur_masks = [all_masks[i] for i in cur]

    no_improve = 0
    max_no_improve = 40

    def counts_of(i):
        return cur_counts

    def remove_block(i):
        m = cur_masks[i]
        for b in _iter_bits(m):
            cur_counts[b] -= 1
        cur.pop(i)
        cur_masks.pop(i)

    def add_block(idx):
        cur.append(idx)
        cur_masks.append(all_masks[idx])
        m = all_masks[idx]
        for b in _iter_bits(m):
            cur_counts[b] += 1

    def is_redundant(i):
        m = cur_masks[i]
        for b in _iter_bits(m):
            if cur_counts[b] == 1:
                return False
        return True

    def try_remove_redundant():
        for i in range(len(cur)):
            if time.time() > deadline:
                return False
            if is_redundant(i):
                remove_block(i)
                if len(cur) < best_len:
                    nonlocal best, best_len
                    best = list(cur)
                    best_len = len(best)
                return True
        return False

    def try_swap():
        if len(cur) == 0:
            return False
        i = random.randrange(len(cur))
        m_i = cur_masks[i]
        uncovered_bits = [b for b in _iter_bits(m_i) if cur_counts[b] == 1]
        if not uncovered_bits:
            remove_block(i)
            if len(cur) < best_len:
                nonlocal best, best_len
                best = list(cur)
                best_len = len(best)
            return True
        if len(uncovered_bits) > 4:
            return False
        uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
        cands = set(blocks_containing_subset[uncovered_bits[0]])
        for b in uncovered_bits[1:]:
            cands &= set(blocks_containing_subset[b])
            if not cands:
                return False
        valid = [c for c in cands if c != i]
        if not valid:
            return False
        j = random.choice(valid)
        remove_block(i)
        add_block(j)
        return True

    def try_remove_two():
        n = len(cur)
        if n < 2:
            return False
        for _ in range(4):
            if time.time() > deadline:
                return False
            i, j = random.sample(range(n), 2)
            combined = cur_masks[i] | cur_masks[j]
            ok = True
            for b in _iter_bits(combined):
                if cur_counts[b] <= 2:
                    ok = False
                    break
            if ok:
                hi, lo = (i, j) if i > j else (j, i)
                remove_block(hi)
                remove_block(lo)
                if len(cur) < best_len:
                    nonlocal best, best_len
                    best = list(cur)
                    best_len = len(best)
                return True
        return False

    while time.time() < deadline:
        progress = False
        for _ in range(25):
            if time.time() > deadline:
                break
            if try_remove_redundant():
                progress = True
            else:
                break
        if progress:
            no_improve = 0
            continue
        if try_remove_two():
            no_improve = 0
            continue
        if try_swap():
            no_improve = 0
            continue
        no_improve += 1
        if no_improve > max_no_improve:
            c = random.randrange(N_BLOCKS)
            if c not in cur:
                add_block(c)
            no_improve = 0

    return best


def _finalize(chosen):
    """Remove any remaining redundant blocks; return list of point-lists."""
    counts = [0] * N_SUBSETS
    for idx in chosen:
        for b in _iter_bits(all_masks[idx]):
            counts[b] += 1
    final = list(chosen)
    for i in range(len(final) - 1, -1, -1):
        m = all_masks[final[i]]
        redundant = True
        for b in _iter_bits(m):
            if counts[b] == 1:
                redundant = False
                break
        if redundant:
            for b in _iter_bits(m):
                counts[b] -= 1
            final.pop(i)
    return [list(all_blocks[i]) for i in final]


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback generic greedy for unexpected parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k = list(itertools.combinations(range(v), k))
        random.shuffle(all_k)
        for b in all_k:
            if not uncovered:
                break
            cov = set(itertools.combinations(b, t))
            if uncovered & cov:
                blocks.append(list(b))
                uncovered -= cov
        return blocks

    start = time.time()
    deadline = start + 112  # safety margin under 120s

    best = None
    best_len = float('inf')

    # Multiple random restarts with different block orderings
    rng = random.Random(12345)
    for restart in range(8):
        if time.time() > deadline:
            break
        order = list(range(N_BLOCKS))
        if restart == 0:
            # deterministic packing order
            pass
        else:
            # shuffle but keep a bias: sort by a random key for diversity
            rng.shuffle(order)
        # temporarily use this order
        global block_order
        block_order = order

        chosen, counts, unc = packing_construct(deadline)
        chosen, counts, unc = repair_construct(chosen, counts, unc, deadline)
        if len(chosen) < best_len:
            best_len = len(chosen)
            best = list(chosen)

        if time.time() < deadline - 5:
            improved = local_search(chosen, counts, deadline)
            if len(improved) < best_len:
                best_len = len(improved)
                best = list(improved)

    if best is None:
        return []
    return _finalize(best)
