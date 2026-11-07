def construct():
    sum_card_map = {0: {0}}  # maps sum value -> set of cardinalities achieving it
    result = []
    for v in range(1, 257):
        # Tentatively add v and check validity
        new_map = {s: set(cards) for s, cards in sum_card_map.items()}
        valid = True
        for s, cards in list(sum_card_map.items()):
            new_s = s + v
            new_cards = {c + 1 for c in cards}
            if new_s in new_map:
                new_map[new_s].update(new_cards)
            else:
                new_map[new_s] = new_cards
            if len(new_map[new_s]) > 1:
                valid = False
                break
        if valid:
            sum_card_map = new_map
            result.append(v)
    return result
