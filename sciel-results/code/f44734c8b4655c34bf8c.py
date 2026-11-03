    def is_valid(rows):
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    all_diffs.append(r[j] - r[i])
        return len(set(all_diffs)) == len(all_diffs)
