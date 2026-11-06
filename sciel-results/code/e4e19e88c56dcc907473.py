                valid = True
                new_diffs = set()
                for a in current_set:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
