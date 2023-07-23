from functools import reduce

def analyze_file(file_name : str):
    losses = []

    with open(file_name, "r") as file:
        for line in file:
            loss = float(line[:-1])
            losses.append(loss)

    avg = reduce(lambda x, y : x + y, losses)/len(losses)
    print(f"Average loss = {avg:.3f}")

    minimum = min(losses)
    print(f"Minimum loss = {minimum:.3f}")

# Analyze results for CVXPY
print("Results for CVXPY")
analyze_file("losses_cvxpy.txt")
print("*"*20)

# Analyze results for CVXPY_DIFF
print("Results for CVXPY_DIFF")
analyze_file("losses_cvxpy_diff.txt")
print("*"*20)