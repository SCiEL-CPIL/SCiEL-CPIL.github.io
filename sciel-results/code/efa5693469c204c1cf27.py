            # Generate ALL valid x in [last_val+1, M]
            # But M can be up to 100+.
            # Let's iterate x from last_val+1 to M.
            # Check Sidon.
            # Score by cover_count.
            candidates = []
            for x in range(last_val + 1, M + 1):
                valid = True
                new_diffs = []
                for s in current_set:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if not valid:
                    continue
                cover_count = 0
                for d in new_diffs:
                    if d in needed:
                        cover_count += 1
                candidates.append((cover_count, x, new_diffs))
            
            if not candidates:
                return False
                
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit to top 50
            if len(candidates) > 50:
                candidates = candidates[:50]
