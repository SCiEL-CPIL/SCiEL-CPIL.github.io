def construct():
          products = {1}
          selected = []
          for x in range(2, 41):
              new_products = set(products)
              for p in products:
                  new_products.add(x * p)
              if len(new_products) == 2 * len(products):
                  products = new_products
                  selected.append(x)
          return selected
