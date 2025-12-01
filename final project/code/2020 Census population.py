import pandas as pd
import math

# Load dataset
df = pd.read_csv("population_2020.csv")

# Convert population to integers (remove commas if present)
df["Population"] = df["Population"].astype(str).str.replace(",", "")
df["Population"] = df["Population"].astype(int)

states = df["State"].tolist()
pop = df["Population"].tolist()
S = 435

# ---------------- Hamilton ----------------
quota = [S * p / sum(pop) for p in pop]
hamilton = [math.floor(q) for q in quota]
remaining = S - sum(hamilton)
remainders = sorted(
    list(enumerate([q - math.floor(q) for q in quota])),
    key=lambda x: x[1],
    reverse=True
)
for i in range(remaining):
    hamilton[remainders[i][0]] += 1

# ---------------- Jefferson ----------------
def jefferson(pop, S):
    seats = [0] * len(pop)
    for _ in range(S):
        idx = max(range(len(pop)), key=lambda i: pop[i] / (seats[i] + 1))
        seats[idx] += 1
    return seats

jefferson_seats = jefferson(pop, S)

# ---------------- Webster ----------------
def webster(pop, S):
    seats = [0] * len(pop)
    for _ in range(S):
        idx = max(range(len(pop)), key=lambda i: pop[i] / (2 * seats[i] + 1))
        seats[idx] += 1
    return seats

webster_seats = webster(pop, S)

# ---------------- Huntington–Hill ----------------
def huntington_hill(pop, S):
    seats = [1] * len(pop)
    for _ in range(S - len(pop)):
        idx = max(
            range(len(pop)),
            key=lambda i: pop[i] / math.sqrt(seats[i] * (seats[i] + 1))
        )
        seats[idx] += 1
    return seats

hill_seats = huntington_hill(pop, S)

# ---------------- Build dataframe ----------------
out = pd.DataFrame({
    "State": states,
    "Hamilton": hamilton,
    "Jefferson": jefferson_seats,
    "Webster": webster_seats,
    "HuntingtonHill": hill_seats
})

# Remove totals row if present
out = out[out["State"].str.contains("TOTAL") == False]

# ---------------- Output ----------------
print("\n===== Seat Allocation Results =====\n")
print(out.to_string(index=False))

print("\n===== States with Seat Differences Across Methods =====\n")
diff = out[(out.max(axis=1, numeric_only=True) - out.min(axis=1, numeric_only=True)) != 0]
print(diff.to_string(index=False))

print("\n===== Deviation from Huntington–Hill (current US method) =====\n")
for m in ["Hamilton", "Jefferson", "Webster"]:
    out[m + "_Diff_from_Hill"] = out[m] - out["HuntingtonHill"]

print(out[[
    "State",
    "Hamilton_Diff_from_Hill",
    "Jefferson_Diff_from_Hill",
    "Webster_Diff_from_Hill"
]].sort_values(by=[
    "Hamilton_Diff_from_Hill",
    "Jefferson_Diff_from_Hill",
    "Webster_Diff_from_Hill"
]))

print("\nDone.")
