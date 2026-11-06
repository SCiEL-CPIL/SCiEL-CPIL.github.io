import time
import sys

sys.setrecursionlimit(3000)

N = 256
recip = [0.0] * (N + 1)
for i in range(1, N + 1):
    recip[i] = 1.0 / i

suffix = [0.0] * (N + 2)
for i in range(N, 0, -1):
    suffix[i] = suffix[i + 1] + recip[i]

best_val = 0.0
best_set = []
deadline = 0.0

def construct():
    global best_val, best_set, deadline
    
    best_val = 0.0
    best_set = []
    deadline = time.time() + 80.0

    def search(index, selected, forbidden, current_sum):
        global best_val, best_set, deadline

        if time.time() > deadline:
            return

        if index > N:
            if current_sum > best_val:
                best_val = current_sum
                best_set = list(selected)
            return

        # Loose upper bound
        if current_sum + suffix[index] <= best_val:
            return

        if index in forbidden:
            search(index + 1, selected, forbidden, current_sum)
            return

        # Try including index
        valid = True
        for a in selected:
            if (2 * a - index) in selected or (2 * index - a) in selected:
                valid = False
                break

        if valid:
            new_forbidden = set(forbidden)
            for a in selected:
                y1 = 2 * a - index
                if 1 <= y1 <= N:
                    new_forbidden.add(y1)
                y2 = 2 * index - a
                if 1 <= y2 <= N:
                    new_forbidden.add(y2)
                s = a + index
                if s % 2 == 0:
                    y3 = s // 2
                    if 1 <= y3 <= N:
                        new_forbidden.add(y3)

            new_selected = set(selected)
            new_selected.add(index)
            
            # Tight bound check for include branch
            sum_rem = 0.0
            for j in range(index + 1, N + 1):
                if j not in new_forbidden:
                    sum_rem += recip[j]
            
            if current_sum + recip[index] + sum_rem > best_val:
                search(index + 1, new_selected, new_forbidden, current_sum + recip[index])

        # Try excluding index
        # Tight bound check for exclude branch
        sum_rem_ex = 0.0
        for j in range(index + 1, N + 1):
            if j not in forbidden:
                sum_rem_ex += recip[j]
        
        if current_sum + sum_rem_ex > best_val:
            search(index + 1, selected, forbidden, current_sum)

    search(1, set(), set(), 0.0)
    
    if not best_set:
        selected = []
        for x in range(1, N + 1):
            if all(2*b-a != x for a in selected for b in selected if a < b):
                selected.append(x)
        return selected
        
    return best_set
