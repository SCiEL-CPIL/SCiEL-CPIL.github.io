        # Greedy cover
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            if state.uncovered == 0:
                break
            # Pick a random uncovered target
            unc_targets = target_range[state.counts[target_start:target_end+1] == 0]
            if len(unc_targets) == 0:
                break
            t = random.choice(unc_targets)
            # Find a representation
            found = False
            # Try random a
            for _ in range(10):
                a = random.randint(0, 128)
                b = t - a
                if 0 <= b <= 128:
                    if a not in state.A:
                        state.add(a)
                    if b not in state.A:
                        state.add(b)
                    found = True
                    break
            if not found:
                # Fallback: try all a
                for a in range(129):
                    b = t - a
                    if 0 <= b <= 128:
                        if a not in state.A:
                            state.add(a)
                        if b not in state.A:
                            state.add(b)
                        found = True
                        break
            if not found:
                break
