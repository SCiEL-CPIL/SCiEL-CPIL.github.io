    def greedy_construct(candidates, time_limit):
        current_sums = set()
        current_elements = []
        available = candidates.copy()
        
        while available and time.time() < time_limit:
            # Sort available by value descending, take top 20
            # This assumes larger numbers are "safer"
            available.sort(reverse=True)
            candidates_to_check = available[:20]
            
            valid_candidates = []
            for x in candidates_to_check:
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + x)
                new_sums.add(x)
                
                if check_divisibility(new_sums):
                    valid_candidates.append((x, new_sums))
                    
            if not valid_candidates:
                # If none of the top 20 work, try the rest? Or break?
                # Let's try the rest but with a stricter time check
                remaining = available[20:]
                for x in remaining:
                    if time.time() > time_limit:
                        break
                    new_sums = set(current_sums)
                    for s in current_sums:
                        new_sums.add(s + x)
                    new_sums.add(x)
                    if check_divisibility(new_sums):
                        valid_candidates.append((x, new_sums))
                        break # Just find one
            
            if not valid_candidates:
                break
                
            # Pick randomly from valid
            x, new_sums = random.choice(valid_candidates)
            current_sums = new_sums
            current_elements.append(x)
            available.remove(x)
            
        return current_elements
