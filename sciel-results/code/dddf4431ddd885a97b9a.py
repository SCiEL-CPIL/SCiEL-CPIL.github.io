"""Improved search for pairwise disjoint residue classes with distinct moduli in [20, 200]."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Precompute gcd table
    gcd_table = {}
    for m1 in range(20, 201):
        for m2 in range(20, 201):
            if m1 != m2:
                gcd_table[(m1, m2)] = gcd(m1, m2)

    def compute_score(selected):
        """Compute sum of 1/m."""
        return float(sum(Fraction(1, m) for m, r in selected))

    def find_valid_residue(m_new, selected):
        """Find a valid residue for m_new given selected classes. Returns None if no valid residue."""
        for r in range(m_new):
            conflict = False
            for m2, r2 in selected:
                g = gcd_table.get((m_new, m2), gcd(m_new, m2))
                if r % g == r2 % g:
                    conflict = True
                    break
            if not conflict:
                return r
        return None

    def greedy_fill(selected, order):
        """Greedy add moduli in given order."""
        existing = {m for m, r in selected}
        sel_list = list(selected)
        for m in order:
            if m in existing:
                continue
            r = find_valid_residue(m, sel_list)
            if r is not None:
                sel_list.append([m, r])
                existing.add(m)
        return sel_list

    def try_add_greedy(selected, m_add, fill_order):
        """Try adding m_add then greedy fill."""
        if m_add in {m for m, r in selected}:
            return None
        sel_list = [x[:] for x in selected]
        r = find_valid_residue(m_add, sel_list)
        if r is None:
            return None
        sel_list.append([m_add, r])
        existing = {m for m, r in sel_list}
        for m in fill_order:
            if m in existing:
                continue
            r2 = find_valid_residue(m, sel_list)
            if r2 is not None:
                sel_list.append([m, r2])
                existing.add(m)
        return sel_list

    def try_swap(selected, m_out, m_in, fill_order):
        """Remove m_out, add m_in, then greedy fill."""
        sel_list = [x[:] for x in selected if x[0] != m_out]
        r = find_valid_residue(m_in, sel_list)
        if r is None:
            return None
        sel_list.append([m_in, r])
        existing = {m for m, r in sel_list}
        for m in fill_order:
            if m in existing:
                continue
            r2 = find_valid_residue(m, sel_list)
            if r2 is not None:
                sel_list.append([m, r2])
                existing.add(m)
        return sel_list

    # Phase 1: Multiple greedy starts with different strategies
    best_selected = []
    best_score = 0.0

    random.seed(42)

    # Strategy 1: Increasing order (small moduli first, high value)
    order = list(range(20, 201))
    selected = greedy_fill([], order)
    best_selected = [x[:] for x in selected]
    best_score = compute_score(selected)

    # Strategy 2: Decreasing order
    order = list(range(200, 19, -1))
    selected = greedy_fill([], order)
    score = compute_score(selected)
    if score > best_score:
        best_score = score
        best_selected = [x[:] for x in selected]

    # Strategy 3-12: Randomized orders with bias toward small moduli
    for trial in range(10):
        if time.time() > deadline - 40:
            break
        order = list(range(20, 201))
        # Weighted shuffle: smaller moduli more likely to come first
        weights = [1.0 / m for m in order]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        shuffled = []
        remaining = list(order)
        rem_weights = list(weights)
        for _ in range(len(order)):
            r = random.random()
            cumsum = 0
            idx = 0
            for i, w in enumerate(rem_weights):
                cumsum += w
                if r <= cumsum:
                    idx = i
                    break
            shuffled.append(remaining[idx])
            rem_weights.pop(idx)
            remaining.pop(idx)
        selected = greedy_fill([], shuffled)
        score = compute_score(selected)
        if score > best_score:
            best_score = score
            best_selected = [x[:] for x in selected]

    if time.time() > deadline - 15:
        return best_selected

    # Phase 2: Local search with add and swap moves
    current_selected = [x[:] for x in best_selected]
    current_score = best_score
    no_improve = 0
    max_no_improve = 50
    fill_order = list(range(20, 201))  # Fill in increasing order for best greedy completion

    while time.time() < deadline - 5:
        improved = False

        # Try adding any missing modulus (prioritize small ones)
        existing = {m for m, r in current_selected}
        candidates = sorted([m for m in range(20, 201) if m not in existing])
        
        for m_add in candidates:
            temp = try_add_greedy(current_selected, m_add, fill_order)
            if temp is not None:
                new_score = compute_score(temp)
                if new_score > current_score:
                    current_selected = temp
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_selected = [x[:] for x in current_selected]
                    improved = True
                    no_improve = 0
                    break

        if improved:
            continue

        # Try single swaps: remove large, add small
        existing = {m for m, r in current_selected}
        candidates_in = sorted([m for m in range(20, 201) if m not in existing])
        candidates_out = sorted([x[0] for x in current_selected], reverse=True)
        
        swap_done = False
        for m_out in candidates_out:
            if swap_done or time.time() > deadline - 5:
                break
            for m_in in candidates_in:
                if m_in >= m_out:
                    break
                temp = try_swap(current_selected, m_out, m_in, fill_order)
                if temp is not None:
                    new_score = compute_score(temp)
                    if new_score > current_score:
                        current_selected = temp
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_selected = [x[:] for x in current_selected]
                        swap_done = True
                        no_improve = 0
                        break

        if swap_done:
            continue

        # Perturbation: remove random subset and re-greedy
        if len(current_selected) > 5:
            n_remove = random.randint(1, min(4, len(current_selected) - 2))
            indices = list(range(len(current_selected)))
            # Bias removal toward larger moduli (lower value)
            weights = [1.0 / current_selected[i][0] for i in indices]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            to_remove = set(random.choices(indices, k=n_remove, weights=weights))
            
            temp_selected = [current_selected[i][:] for i in indices if i not in to_remove]
            temp_selected = greedy_fill(temp_selected, fill_order)
            new_score = compute_score(temp_selected)
            
            if new_score > current_score:
                current_selected = temp_selected
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_selected = [x[:] for x in current_selected]
                no_improve = 0
                continue

        no_improve += 1
        if no_improve >= max_no_improve:
            # Reset to best and try aggressive perturbation
            current_selected = [x[:] for x in best_selected]
            current_score = best_score
            no_improve = 0
            
            n_remove = random.randint(2, min(6, len(current_selected) - 2))
            indices = list(range(len(current_selected)))
            random.shuffle(indices)
            to_remove = set(indices[:n_remove])
            
            temp_selected = [current_selected[i][:] for i in indices if i not in to_remove]
            temp_selected = greedy_fill(temp_selected, fill_order)
            new_score = compute_score(temp_selected)
            if new_score > current_score:
                current_selected = temp_selected
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_selected = [x[:] for x in current_selected]

    return best_selected
# EVOLVE-BLOCK-END
