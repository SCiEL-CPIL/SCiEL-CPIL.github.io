    def dfs(pos, last_val):
        if pos == k + 1:
            # We have placed indices 1 to k
            rulers.append({
                'row': tuple(current_row),
                'diffs': frozenset(current_diffs),
                'max_val': current_row[-1]
            })
            return
        
        # Try all possible values for current_row[pos]
        for v in range(last_val + 1, M + 1):
            new_diffs = set()
            valid = True
            for prev in current_row:
                d = v - prev
                if d in current_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                current_row.append(v)
                current_diffs.update(new_diffs)
                
                dfs(pos + 1, v)
                
                current_row.pop()
                current_diffs.difference_update(new_diffs)
                
                # Pruning: need k + 1 - pos - 1 more elements?
                # We are at pos. Next is pos+1.
                # Elements to place: pos, pos+1, ..., k.
                # Count: k - pos + 1.
                # We placed one. Remaining: k - pos.
                # Space needed: k - pos.
                if M - v < k - pos:
                    break
