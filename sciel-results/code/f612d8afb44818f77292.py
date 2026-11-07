                # Pre-check
                invalid = False
                for s in current_sums:
                    if candidate % s == 0 or s % candidate == 0:
                        invalid = True
                        break
                if invalid:
                    continue
