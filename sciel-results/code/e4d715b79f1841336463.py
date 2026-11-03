import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # --- Precompute triple indexing ---
    all_triples = list(itertools.combinations(range(v), t))
    triple_count = len(all_triples)
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    num_words = (triple_count + 63) // 64

    def block_to_mask(block):
        mask = np.zeros(num_words, dtype=np.uint64)
        for tr in itertools.combinations(block, t):
            idx = triple_index[tr]
            mask[idx // 64] |= (1 << (idx % 64))
        return mask

    def popcount_u64(word):
        # Efficient popcount for a single uint64
        word = np.uint64(word)
        word = word - ((word >> np.uint64(1)) & np.uint64(0x5555555555555555))
        word = (word & np.uint64(0x3333333333333333)) + ((word >> np.uint64(2)) & np.uint64(0x3333333333333333))
        word = (word + (word >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return int((word * np.uint64(0x0101010101010101)) >> np.uint64(56))

    def mask_popcount(mask):
        total = 0
        for w in mask:
            total += popcount_u64(w)
        return total

    def masks_equal(a, b):
        return np.array_equal(a, b)

    full_universe = np.ones(num_words, dtype=np.uint64)

    # --- Phase 1: Build a candidate pool ---
    # We need a diverse pool for SA to explore.
    # Strategy: 
    #   a) A deterministic greedy covering (gives valid baseline + good blocks)
    #   b) Random blocks seeded around uncovered triples
    #   c) Random blocks with biased point selection

    rng = random.Random(12345)
    pool_set = set()
    pool = []

    def add_block(block):
        b = tuple(sorted(block))
        if b not in pool_set:
            pool_set.add(b)
            pool.append(b)

    # a) Greedy covering for baseline
    uncovered = set(all_triples)
    while uncovered:
        first_triple = next(iter(uncovered))
        block = list(first_triple)
        while len(block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for subset in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(subset + (x,)))
                    if new_triple in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                break
            block.append(best_point)
        block.sort()
        add_block(block)
        for tr in itertools.combinations(block, t):
            uncovered.discard(tr)

    baseline_blocks = pool[:]

    # b) Expand pool with random blocks
    target_pool = 800
    while len(pool) < target_pool:
        # Start from a random triple, fill with random points
        r_triple = tuple(sorted(rng.sample(range(v), t)))
        block = list(r_triple)
        while len(block) < k:
            remaining = [x for x in range(v) if x not in block]
            block.append(rng.choice(remaining))
        block.sort()
        add_block(block)

    # c) Add some "perturbed" blocks: take existing blocks, swap 1-2 points
    existing = pool[:]
    for b in existing[:200]:
        for _ in range(3):
            new_b = list(b)
            for _ in range(rng.randint(1, 2)):
                if new_b:
                    i = rng.randrange(len(new_b))
                    new_b[i] = rng.randrange(v)
            new_b.sort()
            add_block(new_b)

    pool_size = len(pool)
    pool_masks = [block_to_mask(b) for b in pool]

    # --- Phase 2: Greedy set cover for initial SA starting point ---
    # Use a fast greedy to get a valid starting solution
    covered = np.zeros(num_words, dtype=np.uint64)
    selected = []
    available = list(range(pool_size))

    while not masks_equal(covered, full_universe):
        if time.time() > deadline - 40:
            break
        best_idx = -1
        best_gain = -1
        for idx in available:
            # marginal gain = popcount(pool_masks[idx] & ~covered)
            gain = 0
            for wi in range(num_words):
                gain += popcount_u64(pool_masks[idx][wi] & ~covered[wi])
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx == -1 or best_gain == 0:
            break
        selected.append(best_idx)
        for wi in range(num_words):
            covered[wi] |= pool_masks[best_idx][wi]
        available.remove(best_idx)

    if not masks_equal(covered, full_universe):
        return [list(b) for b in baseline_blocks]

    best_solution = selected[:]
    best_size = len(selected)

    # --- Phase 3: Simulated Annealing ---
    # State: selected list of pool indices
    # Moves: 
    #   - Remove a redundant block (if all its triples are covered by others)
    #   - Add a random block
    #   - Swap: remove one, add another
    # Energy = len(selected) (minimize)
    # We accept worse solutions with probability exp(-delta_E / T)

    # Maintain marginal coverage: for each triple, how many selected blocks cover it
    # This lets us check redundancy in O(triples_in_block) instead of O(pool_size * triples)
    
    def compute_coverage_counts(sel):
        cc = np.zeros(triple_count, dtype=np.int32)
        for idx in sel:
            m = pool_masks[idx]
            for wi in range(num_words):
                w = m[wi]
                if w == 0:
                    continue
                base = wi * 64
                ww = w
                while ww:
                    bit = (ww & -ww).bit_length() - 1
                    ti = base + bit
                    if ti < triple_count:
                        cc[ti] += 1
                    ww &= ww - 1
        return cc

    def is_redundant(idx, cc):
        """Check if block idx can be removed (all its triples have coverage > 1)."""
        m = pool_masks[idx]
        for wi in range(num_words):
            w = m[wi]
            if w == 0:
                continue
            base = wi * 64
            ww = w
            while ww:
                bit = (ww & -ww).bit_length() - 1
                ti = base + bit
                if ti < triple_count and cc[ti] <= 1:
                    return False
                ww &= ww - 1
        return True

    def try_remove(sel, idx, cc):
        """Remove block idx from sel, update cc. Returns True if valid (no triple goes to 0)."""
        m = pool_masks[idx]
        for wi in range(num_words):
            w = m[wi]
            if w == 0:
                continue
            base = wi * 64
            ww = w
            while ww:
                bit = (ww & -ww).bit_length() - 1
                ti = base + bit
                if ti < triple_count:
                    cc[ti] -= 1
                    if cc[ti] < 0:
                        # Undo
                        cc[ti] += 1
                        return False
                ww &= ww - 1
        sel.remove(idx)
        return True

    def try_add(sel, idx, cc):
        """Add block idx to sel, update cc."""
        m = pool_masks[idx]
        for wi in range(num_words):
            w = m[wi]
            if w == 0:
                continue
            base = wi * 64
            ww = w
            while ww:
                bit = (ww & -ww).bit_length() - 1
                ti = base + bit
                if ti < triple_count:
                    cc[ti] += 1
                ww &= ww - 1
        sel.append(idx)

    def is_valid(sel, cc):
        """Check all triples have coverage >= 1."""
        for ti in range(triple_count):
            if cc[ti] < 1:
                return False
        return True

    # SA parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.9995
    max_steps = 200000
    current_T = T_start

    sel = selected[:]
    cc = compute_coverage_counts(sel)
    current_energy = len(sel)

    # Precompute: for each block, list of triple indices it covers (for fast iteration)
    block_triples = []
    for idx in range(pool_size):
        m = pool_masks[idx]
        tlist = []
        for wi in range(num_words):
            w = m[wi]
            if w == 0:
                continue
            base = wi * 64
            ww = w
            while ww:
                bit = (ww & -ww).bit_length() - 1
                ti = base + bit
                if ti < triple_count:
                    tlist.append(ti)
                ww &= ww - 1
        block_triples.append(tlist)

    step = 0
    while step < max_steps and time.time() < deadline - 10:
        step += 1
        if step % 5000 == 0:
            current_T *= cooling_rate

        # Choose a move
        move_type = rng.random()
        if move_type < 0.3 and len(sel) > 1:
            # Try remove
            idx = sel[rng.randrange(len(sel))]
            if is_redundant(idx, cc):
                # Definitely accept (energy decreases)
                try_remove(sel, idx, cc)
                current_energy -= 1
                if current_energy < best_size:
                    best_size = current_energy
                    best_solution = sel[:]
            else:
                # Try to add back a block that covers the critical triples
                # Find critical triples
                critical = []
                m = pool_masks[idx]
                for wi in range(num_words):
                    w = m[wi]
                    if w == 0:
                        continue
                    base = wi * 64
                    ww = w
                    while ww:
                        bit = (ww & -ww).bit_length() - 1
                        ti = base + bit
                        if ti < triple_count and cc[ti] <= 1:
                            critical.append(ti)
                        ww &= ww - 1
                if critical:
                    # Find a block that covers all critical triples
                    found = False
                    for j in range(pool_size):
                        if j in sel:
                            continue
                        mj = pool_masks[j]
                        ok = True
                        for ti in critical:
                            if not (mj[ti // 64] & (1 << (ti % 64))):
                                ok = False
                                break
                        if ok:
                            # Swap: remove idx, add j
                            try_remove(sel, idx, cc)
                            try_add(sel, j, cc)
                            # Energy unchanged (swap)
                            if not is_valid(sel, cc):
                                # Undo
                                sel.remove(j)
                                m_j = pool_masks[j]
                                for wi in range(num_words):
                                    w = m_j[wi]
                                    if w == 0:
                                        continue
                                    base = wi * 64
                                    ww = w
                                    while ww:
                                        bit = (ww & -ww).bit_length() - 1
                                        ti = base + bit
                                        if ti < triple_count:
                                            cc[ti] -= 1
                                        ww &= ww - 1
                                try_add(sel, idx, cc)
                            else:
                                found = True
                            break
                    if not found and not is_valid(sel, cc):
                        # Undo the removal
                        try_add(sel, idx, cc)
                        current_energy += 1
        elif move_type < 0.5:
            # Try add a random block
            j = rng.randrange(pool_size)
            if j not in sel:
                try_add(sel, j, cc)
                delta_e = 1
                if rng.random() < np.exp(-delta_e / max(current_T, 1e-10)):
                    current_energy += 1
                else:
                    sel.remove(j)
                    m = pool_masks[j]
                    for wi in range(num_words):
                        w = m[wi]
                        if w == 0:
                            continue
                        base = wi * 64
                        ww = w
                        while ww:
                            bit = (ww & -ww).bit_length() - 1
                            ti = base + bit
                            if ti < triple_count:
                                cc[ti] -= 1
                            ww &= ww - 1
                    current_energy = len(sel)
            else:
                pass
        else:
            # Swap: remove one random, add one random
            if len(sel) > 0:
                i_pos = rng.randrange(len(sel))
                idx = sel[i_pos]
                j = rng.randrange(pool_size)
                if j == idx:
                    continue
                if j in sel:
                    continue
                # Remove idx, add j
                try_remove(sel, idx, cc)
                try_add(sel, j, cc)
                if is_valid(sel, cc):
                    current_energy = len(sel)
                    if current_energy < best_size:
                        best_size = current_energy
                        best_solution = sel[:]
                else:
                    # Undo
                    sel.remove(j)
                    m_j = pool_masks[j]
                    for wi in range(num_words):
                        w = m_j[wi]
                        if w == 0:
                            continue
                        base = wi * 64
                        ww = w
                        while ww:
                            bit = (ww & -ww).bit_length() - 1
                            ti = base + bit
                            if ti < triple_count:
                                cc[ti] -= 1
                            ww &= ww - 1
                    try_add(sel, idx, cc)

    # --- Final: Greedy cleanup of best_solution ---
    # Remove any redundant blocks from best_solution
    sel = best_solution[:]
    cc = compute_coverage_counts(sel)
    changed = True
    while changed:
        changed = False
        for i in range(len(sel) - 1, -1, -1):
            if is_redundant(sel[i], cc):
                try_remove(sel, sel[i], cc)
                changed = True
                if len(sel) < best_size:
                    best_size = len(sel)
                    best_solution = sel[:]

    # Verify
    final_blocks = [list(pool[i]) for i in best_solution]
    covered_mask = np.zeros(num_words, dtype=np.uint64)
    for b in final_blocks:
        covered_mask |= block_to_mask(b)

    if not masks_equal(covered_mask, full_universe):
        return [list(b) for b in baseline_blocks]

    return final_blocks
