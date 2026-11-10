    def greedy_shuffled_fast():
        random.shuffle(candidates)
        selected = []
        sums = [False] * (2 * N + 1)
        for x in candidates:
            if sums[2 * x]:
                continue
            
            conflict = False
            for a in selected:
                if sums[x + a]:
                    conflict = True
                    break
            
            if not conflict:
                selected.append(x)
                sums[2 * x] = True
                for a in selected:
                    sums[x + a] = True
        return selected
