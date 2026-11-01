        else:
            # Fallback to greedy
            row = [0]
            local_used = set()
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not (new_diffs & global_used) and not (new_diffs & local_used):
                    row.append(x)
                    local_used.update(new_diffs)
                x += 1
            # Do NOT update global_used here.
            return row
