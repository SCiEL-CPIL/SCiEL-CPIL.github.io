"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # All candidate moduli
    moduli = list(range(20, 201))
    n_mod = len(moduli)

    # Precompute gcd pairs
    gcd_table = {}
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            gcd_table[(i, j)] = math.gcd(moduli[i], moduli[j])

    # We work with a set of (modulus_index, residue) pairs
    # State: dict mapping modulus_index -> residue (or None if not selected)
    # We want to maximize sum of 1/m for selected moduli

    def is_compatible(m1_idx, r1, m2_idx, r2):
        """Check if two selections are disjoint."""
        if m1_idx == m2_idx:
            return False  # same modulus, different residues would be needed but we pick one per modulus
        g = gcd_table.get((min(m1_idx, m2_idx), max(m1_idx, m2_idx)))
        return (r1 % g) != (r2 % g)

    def score_of(selection):
        """Compute score as float sum of 1/m for selected moduli."""
        total = 0.0
        for m_idx, r in selection.items():
            if r is not None:
                total += 1.0 / moduli[m_idx]
        return total

    def find_valid_residue(m_idx, current_selection):
        """Find a residue for modulus m_idx that's compatible with all current selections."""
        for r in range(moduli[m_idx]):
            valid = True
            for other_idx, other_r in current_selection.items():
                if other_r is None or other_idx == m_idx:
                    continue
                if not is_compatible(m_idx, r, other_idx, other_r):
                    valid = False
                    break
            if valid:
                return r
        return None

    def greedy_fill(selection, randomize=False):
        """Try to add moduli with valid residues, prioritizing small moduli (high 1/m)."""
        # Sort by 1/m descending (small modulus first)
        candidates = list(range(n_mod))
        if randomize:
            random.shuffle(candidates)
        else:
            candidates.sort(key=lambda i: moduli[i])

        improved = True
        while improved:
            improved = False
            for m_idx in candidates:
                if selection.get(m_idx) is not None:
                    continue
                r = find_valid_residue(m_idx, selection)
                if r is not None:
                    selection[m_idx] = r
                    improved = True

    def try_add_modulus(selection, m_idx):
        """Try to add a specific modulus with a valid residue."""
        r = find_valid_residue(m_idx, selection)
        if r is not None:
            selection[m_idx] = r
            return True
        return False

    def remove_modulus(selection, m_idx):
        """Remove a modulus from selection."""
        selection[m_idx] = None

    # Initialize with greedy
    selection = {}
    greedy_fill(selection, randomize=False)
    best_score = score_of(selection)
    best_selection = dict(selection)

    # Simulated annealing
    current_score = best_score
    temperature = 1.0
    cooling = 0.9995
    min_temp = 1e-6

    # Count how many moduli are selected
    selected_count = sum(1 for v in selection.values() if v is not None)

    while time.time() < deadline:
        # Periodic greedy fill to consolidate
        if random.random() < 0.05:
            greedy_fill(selection, randomize=True)
            current_score = score_of(selection)
            if current_score > best_score:
                best_score = current_score
                best_selection = dict(selection)
            continue

        # Pick a random selected modulus to reassign or remove
        selected_mods = [i for i, v in selection.items() if v is not None]
        if not selected_mods:
            break

        action = random.random()
        if action < 0.6:
            # Reassign residue of a selected modulus
            m_idx = random.choice(selected_mods)
            old_r = selection[m_idx]
            # Try a different residue
            new_r = random.randint(0, moduli[m_idx] - 1)
            if new_r == old_r:
                continue
            # Check compatibility with all others
            valid = True
            for other_idx, other_r in selection.items():
                if other_r is None or other_idx == m_idx:
                    continue
                if not is_compatible(m_idx, new_r, other_idx, other_r):
                    valid = False
                    break
            if not valid:
                # Try to find a valid residue
                new_r = find_valid_residue(m_idx, selection)
                if new_r is None or new_r == old_r:
                    continue
            selection[m_idx] = new_r
            new_score = score_of(selection)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-10)):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selection = dict(selection)
            else:
                selection[m_idx] = old_r

        elif action < 0.8:
            # Remove a selected modulus
            m_idx = random.choice(selected_mods)
            selection[m_idx] = None
            new_score = score_of(selection)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-10)):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selection = dict(selection)
            else:
                selection[m_idx] = best_selection.get(m_idx)

        else:
            # Add a new modulus
            unselected = [i for i in range(n_mod) if selection.get(i) is None]
            if unselected:
                # Prefer small moduli
                unselected.sort(key=lambda i: moduli[i])
                # Pick from the smaller half with some randomness
                k = max(1, len(unselected) // 2)
                m_idx = random.choice(unselected[:k])
                r = find_valid_residue(m_idx, selection)
                if r is not None:
                    selection[m_idx] = r
                    new_score = score_of(selection)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-10)):
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_selection = dict(selection)
                    else:
                        selection[m_idx] = None

        temperature *= cooling
        if temperature < min_temp:
            temperature = min_temp

    # Final greedy fill on best selection
    final_selection = dict(best_selection)
    greedy_fill(final_selection, randomize=False)
    final_score = score_of(final_selection)
    if final_score > best_score:
        best_score = final_score
        best_selection = final_selection

    # Convert to output format
    result = []
    for m_idx, r in best_selection.items():
        if r is not None:
            result.append([moduli[m_idx], r])

    # Verify pairwise disjointness
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            m1, r1 = result[i]
            m2, r2 = result[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                # Invalid - remove the one with smaller 1/m
                if 1.0 / m1 < 1.0 / m2:
                    result.pop(i)
                else:
                    result.pop(j)
                break
        else:
            continue
        break

    return result
# EVOLVE-BLOCK-END
