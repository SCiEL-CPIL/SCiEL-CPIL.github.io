import time
import random
from math import gcd

def construct():
    """
    Maximize the number of pairwise-disjoint infinite residue classes with
    distinct moduli in [2, 128].

    Two classes (r1,m1) and (r2,m2) are disjoint iff (r1-r2) % gcd(m1,m2) != 0.

    Structural approach:
      1. Build a pool of candidate moduli via greedy multi-order initialization
         (this gives a large, valid starting set).
      2. Refine with a Simulated Annealing-style local search where the state
         is an ORDERED list of selected moduli.  Moves are add / remove / swap.
         After each move we run a fast local repair that greedily re-assigns
         residues (with randomized tie-breaking) so that the set has as few
         conflicts as possible.  The energy function is
             energy = -len(selected) + 10 * num_conflicts
         so that conflict-free sets are strongly preferred and, among them,
         larger sets are preferred.  This lets us escape local optima that a
         single greedy pass gets stuck in.
    """
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin before the 90s kill

    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    n = len(all_moduli)

    # ---------- Precompute gcd table (indexed by modulus value) ----------
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for m1 in range(2, MAX_MOD + 1):
        for m2 in range(2, MAX_MOD + 1):
            gcd_table[m1][m2] = gcd(m1, m2)

    # ---------- Fast local repair: assign residues to an ordered list of
    # moduli to minimize conflicts.  Returns (assignment dict, num_conflicts).
    # The order matters: earlier moduli are fixed first; later ones adapt.
    # ----------
    def repair(order, rng, deterministic=False):
        assignment = {}
        for m in order:
            best_r = None
            best_conf = None
            if deterministic:
                # Deterministic: try residues in order, pick first with min conflicts
                for r in range(m):
                    conf = 0
                    for m2, r2 in assignment.items():
                        g = gcd_table[m][m2]
                        if (r - r2) % g == 0:
                            conf += 1
                    if best_conf is None or conf < best_conf:
                        best_conf = conf
                        best_r = r
                assignment[m] = best_r
            else:
                # Randomized: shuffle residues, keep best seen, small chance to
                # accept a slightly worse one to escape local minima
                residues = list(range(m))
                rng.shuffle(residues)
                best_conf = None
                for r in residues:
                    conf = 0
                    for m2, r2 in assignment.items():
                        g = gcd_table[m][m2]
                        if (r - r2) % g == 0:
                            conf += 1
                    if best_conf is None or conf < best_conf:
                        best_conf = conf
                        best_r = r
                    elif conf == best_conf and rng.random() < 0.1:
                        best_r = r  # tie-break randomization
                assignment[m] = best_r
        # Count total conflicts
        conflicts = 0
        mods = list(assignment.keys())
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                m1, m2 = mods[i], mods[j]
                g = gcd_table[m1][m2]
                if (assignment[m1] - assignment[m2]) % g == 0:
                    conflicts += 1
        return assignment, conflicts

    # ---------- Energy: prefer conflict-free, then larger ----------
    def energy(order, assignment, conflicts):
        if conflicts > 0:
            return 10 * conflicts - len(order)
        return -len(order)

    # ---------- Greedy initialization: build a large valid set ----------
    def greedy_init(order, rng, deterministic=False):
        chosen = []
        for m in order:
            # Check coprime constraint: if gcd(m, m_old)==1 for any chosen,
            # the classes always intersect, so skip m.
            compatible = True
            for _, m_old in chosen:
                if gcd_table[m][m_old] == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            # Find a residue compatible with all chosen
            residues = list(range(m))
            if not deterministic:
                rng.shuffle(residues)
            found = None
            for r in residues:
                ok = True
                for r_old, m_old in chosen:
                    g = gcd_table[m][m_old]
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    found = r
                    break
            if found is not None:
                chosen.append((found, m))
        return chosen

    rng = random.Random(12345)

    best_set = []
    best_count = 0

    def update_best(chosen):
        nonlocal best_set, best_count
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_set = [list(c) for c in chosen]

    # ---------- Phase 1: Greedy initialization with many orderings ----------
    orderings = []
    # Heuristic orderings
    orderings.append(sorted(all_moduli))
    orderings.append(sorted(all_moduli, reverse=True))
    orderings.append(sorted(all_moduli, key=lambda m: -m.bit_count()))
    orderings.append(sorted(all_moduli, key=lambda m: m.bit_count()))
    # Random orderings
    for _ in range(200):
        o = all_moduli[:]
        rng.shuffle(o)
        orderings.append(o)

    for order in orderings:
        if time.time() > deadline - 20:
            break
        chosen = greedy_init(order, rng, deterministic=False)
        update_best(chosen)

    # ---------- Phase 2: Simulated Annealing refinement ----------
    # Start from the best greedy result.  Represent state as an ordered list
    # of (residue, modulus) pairs.  Moves: add a random unselected modulus,
    # remove a random selected modulus, swap two selected moduli.  After each
    # move, run repair to re-assign residues and minimize conflicts.
    if best_set:
        # Convert best_set to an ordered list of moduli
        state_mods = [m for _, m in best_set]
        # Order by modulus for stable repair
        state_mods.sort()
        state_assign, state_conflicts = repair(state_mods, rng, deterministic=False)
        state_energy = energy(state_mods, state_assign, state_conflicts)

        # Track best conflict-free solution found during SA
        sa_best_mods = state_mods[:]
        sa_best_assign = dict(state_assign)
        sa_best_count = len(state_mods) if state_conflicts == 0 else 0

        # SA parameters
        T = 5.0
        T_min = 0.05
        cooling = 0.9995
        max_sa_time = deadline - 5

        # Precompute set of all moduli for fast membership
        all_mod_set = set(all_moduli)

        while time.time() < max_sa_time and T > T_min:
            mods_set = set(state_mods)

            # Choose a move
            move = rng.random()
            new_mods = state_mods[:]

            if move < 0.4 and len(state_mods) < n:
                # Add a random unselected modulus
                candidates = [m for m in all_moduli if m not in mods_set]
                if candidates:
                    m_add = rng.choice(candidates)
                    pos = rng.randint(0, len(new_mods))
                    new_mods.insert(pos, m_add)
            elif move < 0.6 and len(state_mods) > 1:
                # Remove a random selected modulus
                pos = rng.randint(0, len(new_mods) - 1)
                new_mods.pop(pos)
            elif move < 0.8 and len(state_mods) >= 2:
                # Swap two selected moduli
                i, j = rng.sample(range(len(new_mods)), 2)
                new_mods[i], new_mods[j] = new_mods[j], new_mods[i]
            else:
                # Perturb order: random swap or move
                if len(new_mods) >= 2:
                    i, j = rng.sample(range(len(new_mods)), 2)
                    new_mods[i], new_mods[j] = new_mods[j], new_mods[i]

            # Repair the new state
            new_assign, new_conflicts = repair(new_mods, rng, deterministic=False)
            new_energy = energy(new_mods, new_assign, new_conflicts)

            # Accept if better, or with probability exp(-delta/T)
            delta = new_energy - state_energy
            if delta < 0 or rng.random() < pow(2.718281828459045, -delta / T):
                state_mods = new_mods
                state_assign = new_assign
                state_conflicts = new_conflicts
                state_energy = new_energy

                # Update SA best if conflict-free and larger
                if state_conflicts == 0 and len(state_mods) > sa_best_count:
                    sa_best_count = len(state_mods)
                    sa_best_mods = state_mods[:]
                    sa_best_assign = dict(state_assign)

            T *= cooling

        # After SA, update global best with the best conflict-free SA result
        if sa_best_count > best_count:
            best_count = sa_best_count
            best_set = [[sa_best_assign[m], m] for m in sa_best_mods]

    # ---------- Phase 3: Final greedy add to the best result ----------
    # Try to add any remaining moduli to the best set
    if time.time() < deadline - 2:
        chosen_mod_set = set(m for _, m in best_set)
        for m in all_moduli:
            if m in chosen_mod_set:
                continue
            # Check coprime constraint
            compatible = True
            for _, m_old in best_set:
                if gcd_table[m][m_old] == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            # Find a residue
            found = None
            for r in range(m):
                ok = True
                for r_old, m_old in best_set:
                    g = gcd_table[m][m_old]
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    found = r
                    break
            if found is not None:
                best_set.append([found, m])
                chosen_mod_set.add(m)

    # ---------- Ensure non-empty valid result ----------
    if not best_set:
        best_set = [[0, 2]]

    return best_set
