    def build_greedy(order):
        selected = []
        # Max sum is 3 * 4096 = 12288
        max_sum = 3 * N
        sums_present = [False] * (max_sum + 1)
        
        for x in order:
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 5:
                break
            
            valid = True
            
            # Check 3x
            if sums_present[3 * x]:
                continue
            
            # Check 2x + y
            for y in selected:
                if sums_present[2 * x + y]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check x + y + z
            for i in range(len(selected)):
                y = selected[i]
                for z in selected[i:]:
                    if sums_present[x + y + z]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Add x
            selected.append(x)
            
            # Update sums
            sums_present[3 * x] = True
            
            # Add 2x + y
            for y in selected[:-1]:
                sums_present[2 * x + y] = True
            
            # Add x + y + z
            old_len = len(selected) - 1
            for i in range(old_len):
                y = selected[i]
                for j in range(i, old_len):
                    z = selected[j]
                    sums_present[x + y + z] = True
        
        return selected
