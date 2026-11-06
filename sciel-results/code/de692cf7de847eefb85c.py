            for u in range(N):
                if time.time() > DEADLINE - 3:
                    break
                if not ((current_mask >> u) & 1):
                    continue
                # ...
