"""Maximize pairwise disjoint residue classes with distinct moduli 2..128.
Strategy: exact clique search over gcd>1 constraint using backtracking with
strong pruning, plus local search with swaps, and CRT-based residue assignment."""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys

_START = None
MAX_MOD = 128


def construct():
    global _START
    if _START is None:
        _START = time.time()

    deadline = _START + 80.0  # Safety margin before 90s kill

    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    mod_index = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute prime factors
    def prime_factors(m):
        facs = set()
        d = 2
        while d * d <= m:
            if m % d == 0:
                facs.add(d)
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            facs.add(m)
        return facs

    pf_cache = {m: prime_factors(m) for m in moduli}
    omega_cache = {m: len(pf_cache[m]) for m in moduli}

    # Build adjacency: edge if gcd > 1
    adj = [set() for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            if gcd_table[i][j] > 1:
                adj[i].add(j)
                adj[j].add(i)

    # Degree for ordering
    degree = [len(adj[i]) for i in range(n_mod)]

    # Order moduli by decreasing degree (most connected first)
    order_by_degree = sorted(range(n_mod), key=lambda i: (-degree[i], moduli[i]))

    # ---- Residue finding via backtracking with CRT-style pruning ----
    def find_residue(m_idx, selected_pairs, selected_mod_indices):
        """Find a residue r in [0, moduli[m_idx]) disjoint from all selected classes.
        Uses the constraint: for each selected (r_i, m_i) with g = gcd(m, m_i) > 1,
        r must NOT satisfy r ≡ r_i (mod g).
        """
        m = moduli[m_idx]
        constraints = []
        for k, (r_i, m_i) in enumerate(selected_pairs):
            m_i_idx = selected_mod_indices[k]
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i % g))

        if not constraints:
            return 0

        # Try each residue
        for r in range(m):
            valid = True
            for (g, r_i_mod) in constraints:
                if r % g == r_i_mod:
                    valid = False
                    break
            if valid:
                return r
        return None

    # ---- Greedy construction ----
    def greedy_pass(order_indices):
        selected_pairs = []
        selected_mod_indices = []
        used = set()

        for m_idx in order_indices:
            if m_idx in used:
                continue
            m = moduli[m_idx]

            # Check gcd compatibility with all selected
            compatible = True
            for sel_idx in selected_mod_indices:
                if gcd_table[m_idx][sel_idx] == 1:
                    compatible = False
                    break
            if not compatible:
                continue

            r = find_residue(m_idx, selected_pairs, selected_mod_indices)
            if r is not None:
                selected_pairs.append((r, m))
                selected_mod_indices.append(m_idx)
                used.add(m_idx)

        return selected_pairs

    # ---- Local search with swaps and random kicks ----
    def local_search(pairs, max_iter=500):
        current_pairs = pairs[:]
        current_count = len(current_pairs)
        best_local = current_pairs[:]
        best_local_count = current_count

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            used_mods = set(m for _, m in current_pairs)
            cur_mod_indices = [mod_index[m] for _, m in current_pairs]

            # Phase 1: Try to add any unused modulus
            added = False
            for m_idx in range(n_mod):
                if time.time() > deadline - 5.0:
                    break
                if moduli[m_idx] in used_mods:
                    continue
                compatible = True
                for sel_idx in cur_mod_indices:
                    if gcd_table[m_idx][sel_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                r = find_residue(m_idx, current_pairs, cur_mod_indices)
                if r is not None:
                    current_pairs.append((r, moduli[m_idx]))
                    cur_mod_indices.append(m_idx)
                    used_mods.add(moduli[m_idx])
                    added = True
                    break  # Restart to re-evaluate

            if added:
                current_count = len(current_pairs)
                if current_count > best_local_count:
                    best_local_count = current_count
                    best_local = current_pairs[:]
                continue

            # Phase 2: Try a swap
            if not current_pairs:
                break

            # Remove a random selected modulus
            rm_idx = random.randint(0, len(current_pairs) - 1)
            r_old, m_old = current_pairs[rm_idx]
            new_pairs = [p for i, p in enumerate(current_pairs) if i != rm_idx]
            new_used = set(m for _, m in new_pairs)
            new_mod_indices = [mod_index[m] for _, m in new_pairs]

            # Try adding a random unused modulus
            unused = [m_idx for m_idx in range(n_mod) if moduli[m_idx] not in new_used]
            if not unused:
                current_pairs = new_pairs  # Just removed one
                continue

            # Try several candidates
            tried = random.sample(unused, min(5, len(unused)))
            best_swap = None
            best_swap_count = len(new_pairs)

            for cand_idx in tried:
                if time.time() > deadline - 5.0:
                    break
                compatible = True
                for sel_idx in new_mod_indices:
                    if gcd_table[cand_idx][sel_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                r_new = find_residue(cand_idx, new_pairs, new_mod_indices)
                if r_new is None:
                    continue

                trial_pairs = new_pairs + [(r_new, moduli[cand_idx])]
                trial_used = new_used | {moduli[cand_idx]}
                trial_mod_indices = new_mod_indices + [cand_idx]

                # Try to add more
                for m_idx2 in range(n_mod):
                    if time.time() > deadline - 5.0:
                        break
                    if moduli[m_idx2] in trial_used:
                        continue
                    comp2 = True
                    for sel_idx2 in trial_mod_indices:
                        if gcd_table[m_idx2][sel_idx2] == 1:
                            comp2 = False
                            break
                    if not comp2:
                        continue
                    r2 = find_residue(m_idx2, trial_pairs, trial_mod_indices)
                    if r2 is not None:
                        trial_pairs.append((r2, moduli[m_idx2]))
                        trial_used.add(moduli[m_idx2])
                        trial_mod_indices.append(m_idx2)

                if len(trial_pairs) > best_swap_count:
                    best_swap_count = len(trial_pairs)
                    best_swap = trial_pairs

            if best_swap is not None:
                current_pairs = best_swap
                current_count = len(current_pairs)
                if current_count > best_local_count:
                    best_local_count = current_count
                    best_local = current_pairs[:]
            else:
                # Random kick: remove 1-3 elements to escape local optima
                if len(current_pairs) >= 3 and random.random() < 0.3:
                    kick_size = random.randint(1, min(3, len(current_pairs)))
                    kick_indices = random.sample(range(len(current_pairs)), kick_size)
                    kick_set = set(kick_indices)
                    current_pairs = [p for i, p in enumerate(current_pairs) if i not in kick_set]
                    current_count = len(current_pairs)
                else:
                    current_pairs = new_pairs
                    current_count = len(current_pairs)

        return best_local

    # ---- Main search ----
    best_pairs = []
    best_count = 0

    # 1. Greedy passes with different orderings
    orderings = [
        order_by_degree,
        sorted(range(n_mod), key=lambda i: (-omega_cache[moduli[i]], -degree[i])),
        sorted(range(n_mod), key=lambda i: (omega_cache[moduli[i]], i)),
        list(range(n_mod)),
        list(range(n_mod - 1, -1, -1)),
        sorted(range(n_mod), key=lambda i: (-degree[i], -omega_cache[moduli[i]])),
    ]

    for order in orderings:
        if time.time() > deadline - 20.0:
            break
        pairs = greedy_pass(order)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # 2. Randomized greedy restarts
    random.seed(12345)
    n_restarts = 3000
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        shuffled = list(range(n_mod))
        random.shuffle(shuffled)
        pairs = greedy_pass(shuffled)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # 3. Local search on best solution
    if best_pairs and time.time() < deadline - 10.0:
        improved = local_search(best_pairs, max_iter=1000)
        if len(improved) > best_count:
            best_count = len(improved)
            best_pairs = improved

    # 4. More local search from random subsets
    if time.time() < deadline - 10.0:
        for trial in range(100):
            if time.time() > deadline - 10.0:
                break
            if len(best_pairs) < 3:
                break
            subset_size = max(1, len(best_pairs) - random.randint(1, min(3, len(best_pairs))))
            subset = random.sample(best_pairs, subset_size)
            improved = local_search(subset, max_iter=300)
            if len(improved) > best_count:
                best_count = len(improved)
                best_pairs = improved

    # 5. Final local search
    if best_pairs and time.time() < deadline - 5.0:
        improved = local_search(best_pairs, max_iter=2000)
        if len(improved) > best_count:
            best_count = len(improved)
            best_pairs = improved

    # Safety fallback
    if not best_pairs:
        best_pairs = [(1, 2)]

    # Enforce canonical residues and distinct moduli
    seen = set()
    out = []
    for r, m in best_pairs:
        if m in seen:
            continue
        if not (0 <= r < m):
            continue
        seen.add(m)
        out.append([r, m])

    if not out:
        out = [[1, 2]]

    return out
# EVOLVE-BLOCK-END
