            new_row = [0]
            success = True
            for step in range(1, k + 1):
                x = new_row[-1] + 1
                found = False
                limit = max(current_scope, best_scope) + 20
                
                while x <= limit:
                    new_diffs = {x - a for a in new_row}
                    # Check if all new diffs are available (count 0)
                    valid = True
                    for d in new_diffs:
                        if d >= len(diff_counts) or diff_counts[d] > 0:
                            valid = False
                            break
                    if valid:
                        new_row.append(x)
                        found = True
                        break
                    x += 1
