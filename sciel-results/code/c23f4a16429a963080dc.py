                new_val_found = None
                for v in range(min_val, min(current_val, min_val + 50)):
                    valid = True
                    new_diffs = []
                    seen_new = set()
                    for a in row:
                        if a != current_val:
                            d = abs(a - v)
                            if d > MAX_SCOPE or used[d] or d in seen_new:
                                valid = False
                                break
                            new_diffs.append(d)
                            seen_new.add(d)
                    if valid:
                        new_val_found = v
                        break
