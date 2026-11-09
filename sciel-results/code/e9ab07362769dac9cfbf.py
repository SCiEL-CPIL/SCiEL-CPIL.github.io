            uncovered = [c for c in cliques if not (c & test_base)]
            if not uncovered:
                continue
            common = set(uncovered[0])
            for c in uncovered[1:]:
                common &= c
            for w in common:
                if w not in selected_set:
                    swaps_found.append((v, w))
