import time
import random

N_POINTS = 16
N_MASKS = 1 << N_POINTS
ALL_MASKS = range(N_MASKS)

# Precompute popcount
POP_COUNT = [0] * N_MASKS
for i in range(1, N_MASKS):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

# Precompute all triples (masks)
TRIPLES = []
TRIPLE_TO_IDX = {}
for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        for c in range(b + 1, N_POINTS):
            mask = (1 << a) | (1 << b) | (1 << c)
            TRIPLES.append(mask)
            TRIPLE_TO_IDX[mask] = len(TRIPLES) - 1

N_TRIPLES = len(TRIPLES)

# Precompute supersets for each triple
SUPERSETS = [None] * N_TRIPLES
for i, t in enumerate(TRIPLES):
    comp = ((1 << N_POINTS) - 1) ^ t
    sups = []
    sub = comp
    while True:
        sups.append(t | sub)
        if sub == 0:
            break
        sub = (sub - 1) & comp
    SUPERSETS[i] = sups

# Sort masks by popcount descending for quick alpha lookup
SORTED_MASKS = sorted(ALL_MASKS, key=lambda x: POP_COUNT[x], reverse=True)


def construct():
    deadline = time.time() + 110.0
    _random = random.Random(42)

    # State: which triples are in the system
    # Use a set for O(1) lookup
    triples_set = set()
    triples_list = []

    # Independent mask tracking: independent[mask] = True if no triple in system is subset of mask
    independent = [True] * N_MASKS

    def add_triple(t):
        """Add triple t to the system, update independent array."""
        idx = TRIPLE_TO_IDX[t]
        triples_set.add(t)
        triples_list.append(t)
        for s in SUPERSETS[idx]:
            independent[s] = False

    def remove_triple(t):
        """Remove triple t from the system, update independent array."""
        idx = TRIPLE_TO_IDX[t]
        triples_set.discard(t)
        triples_list.remove(t)
        # Need to recompute which masks become independent again
        # A mask s becomes independent if no other triple in system is subset of s
        # This is expensive; instead, we'll mark it as potentially independent and verify
        for s in SUPERSETS[idx]:
            # Check if any other triple in system is subset of s
            still_blocked = False
            for other in triples_set:
                if other != t and (other & s) == other:
                    still_blocked = True
                    break
            if not still_blocked:
                independent[s] = True

    def calc_alpha_and_set():
        """Find the largest independent set."""
        for s in SORTED_MASKS:
            if independent[s]:
                return POP_COUNT[s], s
        return 0, 0

    def can_add(new_t):
        """Check if new_t can be added without violating the pairwise intersection constraint."""
        for t in triples_set:
            if (t & new_t).bit_count() >= 2:
                return False
        return True

    # Initialize with greedy maximal system
    available = TRIPLES[:]
    _random.shuffle(available)
    for t in available:
        if can_add(t):
            add_triple(t)

    current_alpha, current_set = calc_alpha_and_set()
    best_alpha = current_alpha
    best_triples_set = set(triples_set)
    best_triples_list = triples_list[:]

    def snapshot():
        return triples_set.copy(), triples_list[:]

    def restore(snap_set, snap_list):
        nonlocal triples_set, triples_list
        # Clear current
        for t in triples_list:
            triples_set.discard(t)
        triples_list.clear()
        # Reset independent array
        for i in range(N_MASKS):
            independent[i] = True
        # Restore
        for t in snap_list:
            add_triple(t)

    # Local search
    while time.time() < deadline - 5.0:
        # Find candidates: triples that are subsets of current independent set
        candidates = []
        for t in TRIPLES:
            if t not in triples_set and (t & ~current_set) == 0:
                if can_add(t):
                    candidates.append(t)

        _random.shuffle(candidates)

        moved = False
        for t in candidates:
            # Try adding t
            old_alpha = current_alpha
            add_triple(t)
            new_alpha, new_set = calc_alpha_and_set()
            if new_alpha < old_alpha:
                current_alpha = new_alpha
                current_set = new_set
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples_set = triples_set.copy()
                    best_triples_list = triples_list[:]
                moved = True
                break
            else:
                remove_triple(t)

        if moved:
            continue

        # Try swaps: remove one triple, add a candidate
        if candidates:
            t_in = _random.choice(candidates)
            # Find conflicting triples
            conflicting = [t for t in triples_list if (t & t_in).bit_count() >= 2]
            if conflicting:
                _random.shuffle(conflicting)
                for t_out in conflicting:
                    old_alpha = current_alpha
                    remove_triple(t_out)
                    if can_add(t_in):
                        add_triple(t_in)
                        new_alpha, new_set = calc_alpha_and_set()
                        if new_alpha <= old_alpha:
                            current_alpha = new_alpha
                            current_set = new_set
                            if new_alpha < best_alpha:
                                best_alpha = new_alpha
                                best_triples_set = triples_set.copy()
                                best_triples_list = triples_list[:]
                            moved = True
                            break
                        else:
                            remove_triple(t_in)
                            add_triple(t_out)
                    else:
                        add_triple(t_out)

        if not moved and _random.random() < 0.3:
            # Random restart with perturbation
            snap_set, snap_list = snapshot()
            # Remove 2-3 random triples
            for _ in range(_random.randint(2, 3)):
                if triples_list:
                    t = _random.choice(triples_list)
                    remove_triple(t)
            # Add 1-2 random non-conflicting triples
            for _ in range(_random.randint(1, 2)):
                t = _random.choice(TRIPLES)
                if t not in triples_set and can_add(t):
                    add_triple(t)
            new_alpha, new_set = calc_alpha_and_set()
            if new_alpha > current_alpha:
                # Revert
                restore(snap_set, snap_list)
                current_alpha, current_set = calc_alpha_and_set()

    # Final polish
    while time.time() < deadline:
        candidates = []
        for t in TRIPLES:
            if t not in triples_set and (t & ~current_set) == 0:
                if can_add(t):
                    candidates.append(t)

        _random.shuffle(candidates)
        improved = False
        for t in candidates:
            old_alpha = current_alpha
            add_triple(t)
            new_alpha, new_set = calc_alpha_and_set()
            if new_alpha < old_alpha:
                current_alpha = new_alpha
                current_set = new_set
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples_set = triples_set.copy()
                    best_triples_list = triples_list[:]
                improved = True
                break
            else:
                remove_triple(t)

        if improved:
            continue
        break

    # Return best
    result = []
    for mask in best_triples_list:
        pts = []
        for i in range(N_POINTS):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
    return result
