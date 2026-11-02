<<<<<<< SEARCH
                val = 0
                for c in range(rank):
                    val += U[u_row][c] * V[v_row][c] * W[w_row][c]
                if val != 1:
                    return False
=======
                val = 0.0
                for c in range(rank):
                    val += float(U[u_row][c]) * float(V[v_row][c]) * float(W[w_row][c])
                if abs(val - 1.0) > 1e-9:
                    return False
>>>>>>> REPLACE
