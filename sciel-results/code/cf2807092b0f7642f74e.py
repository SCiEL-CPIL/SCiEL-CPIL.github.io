import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0

    def is_valid_fast(pos, color, colors):
        # Case 1: pos is 4th term
        for d in range(1, pos // 3 + 1):
            a = pos - 3 * d
            if a >= 0:
                if len({colors[a], colors[a+d], colors[a+2*d], color}) < 3:
                    return False

        # Case 2: pos is 3rd term
        for d in range(1, pos // 2 + 1):
            a = pos - 2 * d
            if a >= 0 and a + 3 * d < 128:
                c3 = colors[a + 3 * d]
                if c3 != -1:
                    if len({colors[a], colors[a+d], color, c3}) < 3:
                        return False

        # Case 3: pos is 2nd term
        for d in range(1, pos + 1):
            a = pos - d
            if a >= 0 and a + 3 * d < 128:
                c2 = colors[a + 2 * d]
                c3 = colors[a + 3 * d]
                if c2 != -1 and c3 != -1:
                    if len({colors[a], color, c2, c3}) < 3:
                        return False

        # Case 4: pos is 1st term
        for d in range(1, (127 - pos) // 3 + 1):
            if pos + 3 * d < 128:
                c1 = colors[pos + d]
                c2 = colors[pos + 2 * d]
                c3 = colors[pos + 3 * d]
                if c1 != -1 and c2 != -1 and c3 != -1:
                    if len({color, c1, c2, c3}) < 3:
                        return False

        return True

    for num_colors in range(3, 129):
        if time.time() - start_time > timeout - 5:
            break

        for attempt in range(200):
            if time.time() - start_time > timeout - 2:
                break

            colors = [-1] * 128
            color_order = list(range(num_colors))
            random.shuffle(color_order)

            def backtrack(pos):
                if time.time() - start_time > timeout - 1:
                    return False
                if pos == 128:
                    return True
                for c in color_order:
                    if is_valid_fast(pos, c, colors):
                        colors[pos] = c
                        if backtrack(pos + 1):
                            return True
                        colors[pos] = -1
                return False

            if backtrack(0):
                return colors

    # Fallback
    colors = [0] * 128
    for idx in range(128):
        for c in range(128):
            valid = True
            for d in range(1, idx // 3 + 1):
                if len({c, colors[idx-d], colors[idx-2*d], colors[idx-3*d]}) < 3:
                    valid = False
                    break
            if valid:
                colors[idx] = c
                break
    return colors
