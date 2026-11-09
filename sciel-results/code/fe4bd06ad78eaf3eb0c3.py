import time

def construct():
    start = time.time()
    deadline = start + 85.0

    def is_valid(elements):
        n = len(elements)
        if n > 20:
            return False
        sums_by_card = [set() for _ in range(n + 1)]
        sums_by_card[0].add(0)
        for x in elements:
            for k in range(n - 1, -1, -1):
                if sums_by_card[k]:
                    for s in list(sums_by_card[k]):
                        sums_by_card[k + 1].add(s + x)
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums_by_card[i] & sums_by_card[j]:
                    return False
        return True

    best = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Greedy: add smallest valid element each step
    elements = []
    for _ in range(60):
        if time.time() > deadline - 5.0:
            break
        found = False
        for candidate in range(1, 257):
            if candidate in elements:
                continue
            new_elements = elements + [candidate]
            if is_valid(new_elements):
                elements = new_elements
                found = True
                break
        if not found:
            break

    if len(elements) > len(best):
        best = elements

    return list(best)
