            for terms in constraints_6:
                s = sum(sign * voltages[idx] for idx, sign in terms)
                if s % m == 0:
                    valid = False
                    break
