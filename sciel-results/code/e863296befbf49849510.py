import time
import math
import random

def construct():
    # Safety margin: The evaluator kills at 150s. We use 140s to be safe.
    deadline = time.time() + 140
    
    # Moduli in [20, 200]. 
    # The task says distinct moduli. 
    # Using only even moduli was a heuristic in the previous versions.
    # However, the task allows any distinct moduli in [20, 200].
    # Let's check if including odd moduli helps.
    # 1/m is larger for smaller m.
    # GCD constraints: r1 != r2 mod gcd(m1, m2).
    # If m1, m2 are coprime, gcd=1, r1 != r2 mod 1 is always true (0!=0 is false? No, 0==0).
    # Wait, r % 1 is always 0. So if gcd=1, 0 == 0 is True, so they CONFLICT?
    # Let's re-read the condition: "pairwise disjoint (r_1 != r_2 modulo gcd(m_1, m_2))".
    # If gcd(m1, m2) = 1, then r1 % 1 == 0 and r2 % 1 == 0.
    # So 0 == 0 is True. The condition requires r1 != r2 mod gcd.
    # So if gcd=1, we require 0 != 0, which is FALSE.
    # Therefore, we CANNOT have two coprime moduli?
    # Let's verify.
    # If m1=20, m2=21. gcd=1.
    # r1=0, r2=0. 0 % 1 = 0. 0 % 1 = 0. 0 == 0. Condition "r1 != r2 mod gcd" fails.
    # So they are NOT disjoint?
    # The problem states: "Pairwise disjoint residue classes".
    # Usually, disjoint residue classes $A \pmod m$ and $B \pmod n$ means $A \cap B = \emptyset$.
    # $x \equiv r_1 \pmod{m_1}$ and $x \equiv r_2 \pmod{m_2}$.
    # By CRT, a solution exists iff $r_1 \equiv r_2 \pmod{\gcd(m_1, m_2)}$.
    # If a solution exists, the sets are NOT disjoint (they intersect).
    # So for them to be disjoint, NO solution should exist.
    # Therefore, we require $r_1 \not\equiv r_2 \pmod{\gcd(m_1, m_2)}$.
    # Yes, this matches the prompt: "r_1 != r_2 modulo gcd(m_1, m_2)".
    # So if gcd=1, we need $r_1 \not\equiv r_2 \pmod 1$.
    # Since any integer is congruent to 0 mod 1, $0 \not\equiv 0$ is false.
    # So we CANNOT use coprime moduli?
    # This implies we must use moduli that share a common factor > 1.
    # This is why the previous programs used even moduli!
    # If we pick two odd numbers, they might be coprime.
    # If we pick 20 and 21, gcd=1, impossible to be disjoint.
    # So we must restrict to moduli that are NOT pairwise coprime.
    # The set of even numbers works well because gcd(m1, m2) >= 2.
    # Can we mix even and odd?
    # If m1 is even, m2 is odd. gcd can be 1.
    # If gcd=1, impossible.
    # So we can only include pairs where gcd > 1.
    # This is a constraint graph. We need to find a set of moduli such that for every pair, gcd > 1 AND we can assign residues.
    # Actually, if gcd > 1, we have $g$ residue classes mod $g$. We just need to pick different ones.
    # If we use all even moduli, gcd is at least 2.
    # If we use multiples of 3, gcd is at least 3.
    # If we mix, e.g., 20 (even) and 21 (odd, mult of 3). gcd=1. Bad.
    # 20 and 24. gcd=4. Good.
    # 21 and 24. gcd=3. Good.
    # 20 and 25. gcd=5. Good.
    # 21 and 25. gcd=1. Bad.
    # So we can't just take all numbers. We must take a clique in the graph where edge exists if gcd > 1.
    # And we must be able to assign residues.
    # The previous programs restricted to even moduli. This guarantees gcd >= 2.
    # Is it better to mix?
    # If we drop some even moduli to include odd moduli with high 1/m (small odds), it might help.
    # Small odds: 21, 23, 25, 27, 29...
    # 21 (gcd with evens: gcd(21, 20)=1, gcd(21, 22)=1, gcd(21, 24)=3).
    # So 21 can coexist with 24, 26 (gcd 1? 21,26=1), 28 (1), 30 (3).
    # So 21 can only coexist with evens that are multiples of 3 or 7?
    # Multiples of 3 in even range: 24, 30, 36...
    # Multiples of 7 in even range: 28, 42...
    # So 21 is very constrained.
    # Given the complexity, sticking to even moduli is a strong heuristic that was used in the top scoring inspirations.
    # I will stick to even moduli to ensure a dense valid graph.
    
    moduli = list(range(20, 201, 2))  # 100 even moduli
    n = len(moduli)
    idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Helper to check validity of a pair of (m, r) against a list of classes
    def is_pair_valid(m_new, r_new, classes):
        mi_new = idx.get(m_new)
        if mi_new is None:
            return False
        for m_old, r_old in classes:
            mi_old = idx.get(m_old)
            if mi_old is None:
                continue
            g = gcd_table[mi_new][mi_old]
            if r_new % g == r_old % g:
                return False
        return True

    def try_greedy(order):
        """Build a solution by adding moduli in the given order, first valid residue."""
        classes = []
        present = set()
        for m in order:
            if m in present:
                continue
            found = False
            # Try residues in a specific order? 
            # For even moduli, residues 0..m-1.
            # Usually, lower residues are "better" if they leave more room? 
            # Not necessarily. But first valid is a good heuristic.
            for r in range(m):
                if is_pair_valid(m, r, classes):
                    classes.append([m, r])
                    present.add(m)
                    found = True
                    break
            if not found:
                # If no residue works, we skip this modulus.
                # This can happen if it's incompatible with the current set.
                pass
        return classes

    def score_of(classes):
        return sum(1.0 / m for m, _ in classes)

    def local_search(classes, max_iters=50000):
        """
        Improve via:
        A: Direct add of smallest missing modulus.
        B: Reassign one existing class to free a slot for the smallest missing modulus.
        C: Random Kick.
        """
        classes = [c[:] for c in classes]
        score = score_of(classes)
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        missing.sort()  # Smallest first (highest 1/m)
        
        rng = random.Random(12345)

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False

            # Move A: Direct add of smallest missing modulus
            added = False
            if missing:
                m_target = missing[0]
                mi_target = idx[m_target]
                for r in range(m_target):
                    if is_pair_valid(m_target, r, classes):
                        classes.append([m_target, r])
                        present.add(m_target)
                        missing.remove(m_target)
                        score += 1.0 / m_target
                        added = True
                        break
            if added:
                continue

            # Move B: Reassign one existing class to enable adding the smallest missing modulus
            if not missing:
                break
            
            m_target = missing[0]
            mi_target = idx[m_target]
            
            # Find residue for m_target with minimum conflicts
            best_r_target = None
            min_conf = float('inf')
            # Optimization: We don't need to check all r if we find 0 conflicts
            for r in range(m_target):
                conf = 0
                for m2, r2 in classes:
                    g = gcd_table[mi_target][idx[m2]]
                    if r % g == r2 % g:
                        conf += 1
                        # Early break if conf > 1? We want min conf.
                        # If conf > 1, it's unlikely to be the min if we find a 0 or 1 later.
                        # But we need the *min*.
                if conf < min_conf:
                    min_conf = conf
                    best_r_target = r
                    if conf == 0:
                        # This shouldn't happen if Move A failed, but just in case
                        # (maybe Move A failed due to bug or state change?)
                        # Actually if conf is 0, Move A should have added it.
                        # So min_conf should be >= 1.
                        pass
            
            if min_conf <= 1:
                # Identify conflicting classes
                conf_js = []
                for j, (m2, r2) in enumerate(classes):
                    g = gcd_table[mi_target][idx[m2]]
                    if best_r_target % g == r2 % g:
                        conf_js.append(j)
                
                # Try to reassign one of the conflicting classes to a different valid residue
                found_move = False
                for j in conf_js:
                    mj, rj = classes[j]
                    mj_i = idx[mj]
                    g_jt = gcd_table[mj_i][mi_target]
                    target_bad = best_r_target % g_jt
                    
                    # Try to find a new_r for mj such that:
                    # 1. new_r % g_jt != target_bad (so it doesn't conflict with m_target anymore)
                    # 2. new_r is valid against all OTHER classes
                    found_new_r = False
                    for new_r in range(mj):
                        if new_r % g_jt == target_bad:
                            continue
                        # Check against all other classes
                        ok = True
                        for k, (m2, r2) in enumerate(classes):
                            if k == j:
                                continue
                            g = gcd_table[mj_i][idx[m2]]
                            if new_r % g == r2 % g:
                                ok = False
                                break
                        if ok:
                            # Apply move
                            classes[j][1] = new_r
                            classes.append([m_target, best_r_target])
                            present.add(m_target)
                            missing.remove(m_target)
                            score += 1.0 / m_target
                            found_move = True
                            found_new_r = True
                            break
                    if found_move:
                        improved = True
                        break
            
            if improved:
                continue

            # Move C: Random Kick
            # Perturb a random existing class to a random valid residue
            if classes:
                j = rng.randint(0, len(classes) - 1)
                mj, rj = classes[j]
                mj_i = idx[mj]
                # Try a few random new residues
                for _ in range(20):
                    new_r = rng.randint(0, mj - 1)
                    if new_r == rj:
                        continue
                    ok = True
                    for k, (m2, r2) in enumerate(classes):
                        if k == j:
                            continue
                        g = gcd_table[mj_i][idx[m2]]
                        if new_r % g == r2 % g:
                            ok = False
                            break
                    if ok:
                        classes[j][1] = new_r
                        break
            else:
                # If no classes, add the smallest modulus
                classes.append([moduli[0], 0])
                present.add(moduli[0])
                missing.remove(moduli[0])
                score += 1.0 / moduli[0]

        return classes, score

    # Initial greedy with smallest-first
    best_classes = try_greedy(sorted(moduli))
    best_score = score_of(best_classes)

    # Local search from the initial greedy
    best_classes, best_score = local_search(best_classes)

    # Multi-start randomized greedy with local search
    rng = random.Random(42)
    # We have ~140s. 
    # Initial greedy + LS takes some time.
    # Let's do several restarts.
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        # Weighted random order
        # Bias towards smaller moduli
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        # Sort by a key that mixes m and random noise
        # Smaller m should come earlier on average
        weighted = sorted(shuffled, key=lambda m: (m * 0.5 + rng.random() * 50))
        
        cand = try_greedy(weighted)
        s = score_of(cand)
        
        # Only run local search if the initial score is promising
        # Or just run it, as LS is fast enough
        if s > best_score * 0.9: # Heuristic: only polish good candidates
            cand_classes, cand_score = local_search(cand)
            if cand_score > best_score:
                best_classes = cand_classes
                best_score = cand_score

    # Final polish: try to add any remaining small moduli directly
    present = set(m for m, _ in best_classes)
    missing = [m for m in moduli if m not in present]
    missing.sort()
    for m in missing:
        if time.time() > deadline - 2:
            break
        mi = idx[m]
        for r in range(m):
            if is_pair_valid(m, r, best_classes):
                best_classes.append([m, r])
                present.add(m)
                best_score += 1.0 / m
                break

    return best_classes
