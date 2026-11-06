            min_d = last_denom + 1
            if min_d < 5:
                min_d = 5
            # Ensure 1/min_d <= rem
            while Fraction(1, min_d) > rem:
                min_d += 1
