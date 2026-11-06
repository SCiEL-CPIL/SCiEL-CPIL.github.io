            if key not in valid_cache:
                if quick_check(candidate):
                    if is_valid(candidate):
                        valid_cache[key] = True
                        best_net = candidate
                        stuck = 0
                    else:
                        valid_cache[key] = False
                        stuck += 1
                else:
                    valid_cache[key] = False
                    stuck += 1
