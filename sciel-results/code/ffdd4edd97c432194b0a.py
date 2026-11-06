            new_sums = [3*x]
            for y in cs:
                new_sums.append(2*x + y)
                new_sums.append(x + 2*y)
            for i in range(n):
                for j in range(i+1, n):
                    new_sums.append(x + cs[i] + cs[j])
            
            # Check for duplicates in new_sums
            if len(new_sums) != len(set(new_sums)):
                continue
                
            # Check against existing sums
            # We can do: if any(s in sums for s in new_sums): continue
            if any(s in sums for s in new_sums):
                continue
                
            current_set.append(x)
            sums.update(new_sums)
