"""Optimized program: greedy initialization + local search (add/swap) + SA + polish."""
import math
import time
import random

# EVOLVE-BLOCK-START

def _is_compatible(m1, r1, m2, r2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _find_valid_residue(m, others):
    """
    Find a residue r for modulus m such that (m, r) is compatible with all (m2, r2) in others.
    Returns r or None.
    """
    for r in range(m):
        valid = True
        for m2, r2 in others:
            if not _is_compatible(m, r, m2, r2):
                valid = False
                break
        if valid:
            return r
    return None

def _score(chosen):
    return sum(1.0 / m for m, _ in chosen)

def _add_pass(chosen, deadline):
    """Try to add any valid modulus, repeatedly until no more can be added."""
    moduli_in_set = {m for m, _ in chosen}
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for m in range(20, 201):
            if time.time() > deadline:
                break
            if m in moduli_in_set:
                continue
            r = _find_valid_residue(m, chosen)
            if r is not None:
                chosen.append([m, r])
                moduli_in_set.add(m)
                improved = True
                break  # restart to catch chain reactions
    return chosen

def _swap_pass(chosen, deadline):
    """Try swap moves: remove one member, add another (smaller modulus preferred)."""
    moduli_in_set = {m for m, _ in chosen}
    improved = True
    while improved and time.time() < deadline:
        improved = False
        sorted_members = sorted(chosen, key=lambda x: x[0], reverse=True)
        for m_out, r_out in sorted_members:
            if time.time() > deadline:
                break
            for m_in in range(20, m_out):
                if m_in in moduli_in_set:
                    continue
                others = [p for p in chosen if p[0] != m_out]
                r_in = _find_valid_residue(m_in, others)
                if r_in is not None:
                    chosen = [p for p in chosen if p[0] != m_out]
                    moduli_in_set.discard(m_out)
                    chosen.append([m_in, r_in])
                    moduli_in_set.add(m_in)
                    improved = True
                    break
            if improved:
                break
    return chosen

def _reassign_pass(chosen, deadline):
    """Try to find better residues for existing moduli to enable more additions."""
    moduli_in_set = {m for m, _ in chosen}
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try changing residue of each member
        for i, (m, r) in enumerate(chosen):
            if time.time() > deadline:
                break
            others = [p for j, p in enumerate(chosen) if j != i]
            # Try all residues, prefer ones that allow more future additions
            best_r = None
            best_future = -1
            for r_try in range(m):
                if not all(_is_compatible(m, r_try, m2, r2) for m2, r2 in others):
                    continue
                # Count how many new moduli could be added
                temp_chosen = chosen[:i] + [[m, r_try]] + chosen[i+1:]
                temp_moduli = {mm for mm, _ in temp_chosen}
                future = 0
                for m_new in range(20, 201):
                    if m_new in temp_moduli:
                        continue
                    if _find_valid_residue(m_new, temp_chosen) is not None:
                        future += 1
                if future > best_future:
                    best_future = future
                    best_r = r_try
            if best_r is not None and best_r != r:
                chosen[i] = [m, best_r]
                improved = True
                break
    return chosen

def _sa_pass(chosen, deadline):
    """Simulated annealing with add/remove/swap moves."""
    moduli_in_set = {m for m, _ in chosen}
    current_score = _score(chosen)
    best_chosen = [list(p) for p in chosen]
    best_score = current_score
    T = 0.5  # initial temperature
    cooling = 0.999
    min_T = 0.001
    
    while time.time() < deadline and T > min_T:
        # Random move
        move_type = random.random()
        if move_type < 0.4:
            # Add
            available = [m for m in range(20, 201) if m not in moduli_in_set]
            if available:
                m_in = random.choice(available)
                r_in = _find_valid_residue(m_in, chosen)
                if r_in is not None:
                    new_chosen = chosen + [[m_in, r_in]]
                    new_score = _score(new_chosen)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / T):
                        chosen = new_chosen
                        moduli_in_set.add(m_in)
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_chosen = [list(p) for p in chosen]
        elif move_type < 0.7:
            # Remove
            if len(chosen) > 1:
                idx = random.randrange(len(chosen))
                m_out = chosen[idx][0]
                new_chosen = chosen[:idx] + chosen[idx+1:]
                new_score = _score(new_chosen)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    chosen = new_chosen
                    moduli_in_set.discard(m_out)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = [list(p) for p in chosen]
        else:
            # Swap: remove one, add another
            if len(chosen) > 0:
                idx = random.randrange(len(chosen))
                m_out = chosen[idx][0]
                available = [m for m in range(20, 201) if m not in moduli_in_set]
                if available:
                    m_in = random.choice(available)
                    others = [p for j, p in enumerate(chosen) if j != idx]
                    r_in = _find_valid_residue(m_in, others)
                    if r_in is not None:
                        new_chosen = chosen[:idx] + [[m_in, r_in]] + chosen[idx+1:]
                        new_score = _score(new_chosen)
                        delta = new_score - current_score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            chosen = new_chosen
                            moduli_in_set.discard(m_out)
                            moduli_in_set.add(m_in)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_chosen = [list(p) for p in chosen]
        T *= cooling
        if time.time() > deadline - 5.0:
            break
    
    # Final polish on best
    if time.time() < deadline - 2.0:
        best_chosen = _add_pass(best_chosen, deadline - 1.0)
        best_chosen = _swap_pass(best_chosen, deadline - 0.5)
    
    return best_chosen

def _polish_pass(chosen, deadline):
    """Final high-precision polish: try all single-residue changes and small swaps."""
    moduli_in_set = {m for m, _ in chosen}
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try changing residue of each member
        for i, (m, r) in enumerate(chosen):
            if time.time() > deadline:
                break
            others = [p for j, p in enumerate(chosen) if j != i]
            for r_try in range(m):
                if r_try == r:
                    continue
                if not all(_is_compatible(m, r_try, m2, r2) for m2, r2 in others):
                    continue
                # Check if this enables adding a new modulus
                temp_chosen = chosen[:i] + [[m, r_try]] + chosen[i+1:]
                temp_moduli = {mm for mm, _ in temp_chosen}
                for m_new in range(20, 201):
                    if m_new in temp_moduli:
                        continue
                    r_new = _find_valid_residue(m_new, temp_chosen)
                    if r_new is not None:
                        # Apply the residue change and add
                        chosen[i] = [m, r_try]
                        chosen.append([m_new, r_new])
                        moduli_in_set.add(m_new)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return chosen

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under the 150 s wall-clock limit
    
    # 1. Initial Greedy Construction
    moduli = list(range(20, 201))
    chosen = []
    for m in moduli:
        if time.time() > deadline:
            break
        best_r = None
        best_conf = m + 1
        for r in range(m):
            conf = 0
            for (m2, r2) in chosen:
                g = math.gcd(m, m2)
                if (r - r2) % g == 0:
                    conf += 1
                    if conf >= best_conf:
                        break
            if conf < best_conf:
                best_conf = conf
                best_r = r
                if best_conf == 0:
                    break
        if best_conf == 0:
            chosen.append([m, best_r])
    
    # 2. Add pass
    if time.time() < deadline - 30.0:
        chosen = _add_pass(chosen, deadline - 30.0)
    
    # 3. Swap pass
    if time.time() < deadline - 25.0:
        chosen = _swap_pass(chosen, deadline - 25.0)
    
    # 4. Reassign pass
    if time.time() < deadline - 20.0:
        chosen = _reassign_pass(chosen, deadline - 20.0)
    
    # 5. Simulated annealing
    if time.time() < deadline - 10.0:
        chosen = _sa_pass(chosen, deadline - 10.0)
    
    # 6. Final polish
    if time.time() < deadline:
        chosen = _polish_pass(chosen, deadline)
    
    return chosen

# EVOLVE-BLOCK-END
