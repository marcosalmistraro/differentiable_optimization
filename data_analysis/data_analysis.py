from functools import reduce

losses = []

with open('losses.txt', 'r') as file:
    for line in file:
        loss = float(line[:-1])
        losses.append(loss)

avg = reduce(lambda x, y : x + y, losses)/len(losses)
print(f"Average loss = {avg:.3f}")

min = min(losses)
print(f"Minimum loss = {min:.3f}")