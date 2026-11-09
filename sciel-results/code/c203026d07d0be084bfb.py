        if remaining_terms == 2:
            # rem_sum = 1/a + 1/b
            # b = a / (rem_sum * a - 1)
            # a must be such that rem_sum * a - 1 divides a
            # Also b > a >= min_d
            # And b >= min_d
            # a must be >= min_d
            # Also 1/a < rem_sum => a > 1/rem_sum
            # And 1/a + 1/b = rem_sum => 1/a < rem_sum/2 is not required, but 1/a < rem_sum.
            # Actually, since b > a, 1/b < 1/a, so rem_sum < 2/a => a < 2/rem_sum.
            # So a is in (1/rem_sum, 2/rem_sum).
            a_min = max(min_d, int(1 / rem_sum) + 1)
            a_max = int(2 / rem_sum)
            
            for a in range(a_min, a_max + 1):
                # Check time
                if time.time() > deadline:
                    return None
                # b = a / (rem_sum * a - 1)
                # rem_sum is Fraction p/q
                # rem_sum * a = p*a / q
                # rem_sum * a - 1 = (p*a - q) / q
                # b = a / ((p*a - q) / q) = a * q / (p*a - q)
                # So p*a - q must divide a*q
                # Let denom = p*a - q
                # if denom <= 0: continue (a too small)
                # if (a * q) % denom != 0: continue
                # b = (a * q) // denom
                # if b <= a: continue (we want distinct and b > a to avoid duplicates? No, just distinct. But we enforce a < b by range? No, we enforce a >= min_d. We should enforce a < b to maintain order and distinctness.)
                # Actually, if we enforce a < b, we cover all pairs {a,b} exactly once.
                
                p = rem_sum.numerator
                q = rem_sum.denominator
                denom = p * a - q
                if denom <= 0:
                    continue
                if (a * q) % denom != 0:
                    continue
                b = (a * q) // denom
                if b > a and b >= min_d:
                    return current_list + [a, b]
            return None
    