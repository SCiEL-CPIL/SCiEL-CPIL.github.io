def construct():
    import time
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    from fractions import Fraction
    from math import isqrt, gcd
    from collections import defaultdict

    target = Fraction(5, 31)

    best_max_den = 10000001
    best_solution = None

    # Phase 1: Exhaustive 3-term search via algebraic factor-pair method
    for a in range(7, 19):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break

        num = 5 * a - 31
        if num <= 0:
            continue
        den = 31 * a

        g = gcd(num, den)
        n = num // g
        d = den // g

        d2 = d * d
        limit = isqrt(d2)

        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            v = d2 // u

            bu = u + d
            bv = v + d

            if bu % n == 0 and bv % n == 0:
                b = bu // n
                c = bv // n

                if b <= 0 or c <= 0:
                    continue
                if b == a or c == a or b == c:
                    continue
                if b > 10000000 or c > 10000000:
                    continue

                m = max(a, b, c)
                if m < best_max_den:
                    check = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                    if check == target:
                        best_max_den = m
                        best_solution = [a, b, c]

    # Phase 2: Meet-in-the-middle for 4, 5, 6 terms
    # Strategy: For each bound B, generate all k-term partial sums (with distinct denominators in [lo, B])
    # and look for complementary (total-k)-term sums that complete the target.
    
    def find_mitm(total_terms, max_bound, min_den=7):
        """Find total_terms distinct denominators in [min_den, max_bound] summing reciprocally to target."""
        nonlocal best_max_den, best_solution
        if time.time() - start_time > DEADLINE:
            return
        if max_bound < min_den or max_bound >= best_max_den:
            return
        
        half = total_terms // 2
        other = total_terms - half
        
        # Generate all 'half'-term combinations with sum, tracking the max denominator used
        # and the set of denominators used
        half_sums = defaultdict(list)  # Fraction -> list of (max_den_in_combo, sorted_tuple_of_dens)
        
        # For half terms: enumerate combinations of size 'half' from [min_den, max_bound]
        # This is C(max_bound-min_den+1, half) which can be large
        # Instead, use recursive generation with pruning
        
        def gen_combos(k, start, current_sum, current_dens, max_so_far):
            if time.time() - start_time > DEADLINE:
                return
            if k == 0:
                half_sums[current_sum].append((max_so_far, tuple(current_dens)))
                return
            # Bounds: need k more terms, each >= start, <= max_bound
            # Sum of k terms each >= 1/max_bound => current_sum + k/max_bound >= ... 
            # Also current_sum + k/start <= target (roughly)
            # Prune: remaining sum needed = target - current_sum
            # Each remaining term contributes at most 1/start, so k/start >= target - current_sum
            # And each contributes at least 1/max_bound, so k/max_bound <= target - current_sum
            rem_needed = target - current_sum
            if rem_needed < 0:
                return
            if rem_needed > k / start:  # even k terms of 1/start can't reach
                return
            if rem_needed < k / max_bound:  # even k terms of 1/max_bound exceed needed
                # Actually this means we need terms larger than max_bound, which isn't possible
                # Wait: if rem_needed < k/max_bound, it means we need terms > max_bound on average
                # So this is a dead end
                return
            
            for x in range(start, max_bound + 1):
                if x < min_den:
                    continue
                new_sum = current_sum + Fraction(1, x)
                if new_sum > target:
                    break
                current_dens.append(x)
                gen_combos(k - 1, x + 1, new_sum, current_dens, max(max_so_far, x))
                current_dens.pop()
        
        gen_combos(half, min_den, Fraction(0), [], min_den)
        
        if time.time() - start_time > DEADLINE:
            return
        
        # Now for each half-sum, we need (target - half_sum) expressed as 'other' terms
        # with denominators > max denominator in the half-combo, <= max_bound, distinct from half-combo
        for hsum, combos in half_sums.items():
            if time.time() - start_time > DEADLINE:
                return
            rem = target - hsum
            if rem < 0:
                continue
            
            # Generate 'other'-term combos for 'rem' with denominators > max_den_of_half
            # and <= max_bound, not in the half combo
            # We'll do this per-combo since the min_den constraint depends on the combo
            
            for max_h, dens_tuple in combos:
                if time.time() - start_time > DEADLINE:
                    return
                if max_h >= best_max_den:
                    continue
                
                # Find 'other' terms for 'rem' with denominators in (max_h, max_bound], distinct from dens_tuple
                used = set(dens_tuple)
                
                def gen_other(k, start, cur_sum, cur_dens, max_sofar):
                    if time.time() - start_time > DEADLINE:
                        return False
                    if k == 0:
                        if cur_sum == rem:
                            all_dens = list(dens_tuple) + cur_dens
                            m = max(all_dens)
                            if m < best_max_den and m <= max_bound:
                                # Verify
                                s = sum(Fraction(1, x) for x in all_dens)
                                if s == target and len(set(all_dens)) == len(all_dens):
                                    best_max_den = m
                                    best_solution = sorted(all_dens)
                                return True
                        return False
                    if cur_sum > rem:
                        return False
                    rem_needed = rem - cur_sum
                    if rem_needed <= 0:
                        return False
                    if rem_needed > k / start:
                        return False
                    if rem_needed < k / max_bound:
                        return False
                    
                    for x in range(start, max_bound + 1):
                        if x in used:
                            continue
                        if x < min_den:
                            continue
                        new_sum = cur_sum + Fraction(1, x)
                        if new_sum > rem:
                            break
                        cur_dens.append(x)
                        used.add(x)
                        if gen_other(k - 1, x + 1, new_sum, cur_dens, max(max_sofar, x)):
                            return True
                        cur_dens.pop()
                        used.discard(x)
                    return False
                
                if gen_other(other, max_h + 1, Fraction(0), [], max_h):
                    return
        
        return
    
    # Try increasing bounds with meet-in-the-middle for 4, 5, 6 terms
    for total_terms in [4, 5, 6]:
        if time.time() - start_time > DEADLINE:
            break
        for max_bound in range(15, min(best_max_den, 200) + 1):
            if time.time() - start_time > DEADLINE:
                break
            if max_bound >= best_max_den:
                break
            find_mitm(total_terms, max_bound)
            if best_max_den <= max_bound:
                break

    if best_solution is not None:
        return best_solution

    # Fallback: greedy
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d_val = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d_val)
        r -= Fraction(1, d_val)
    return values
