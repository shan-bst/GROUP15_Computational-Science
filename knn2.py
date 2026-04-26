import numpy as np
import matplotlib.pyplot as plt

#DATA
data = np.genfromtxt("diabetes-k-nn.csv", delimiter=",", skip_header=1)

X = data[:, :-1]
y = data[:, -1]

columns = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","Pedigree","Age"
]
#MEDIAN
cols_with_zero = [1,2,3,4,5]  # Glucose, BP, Skin, Insulin, BMI

print("=== TABLE 1: ZERO VALUE HANDLING ===")
print(f"{'Feature':<20}{'Before':<10}{'After':<10}")

for col in cols_with_zero:
    before = np.sum(X[:, col] == 0)
    
    median = np.median(X[X[:, col] != 0, col])
    X[X[:, col] == 0, col] = median
    
    after = np.sum(X[:, col] == 0)
    
    print(f"{columns[col]:<20}{before:<10}{after:<10}")

#STANDARDIZATION
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_scaled = (X - mean) / std

print("\n=== TABLE 2: SUMMARY STATISTICS ===")
print(f"{'Feature':<20}{'Mean(Scaled)':<15}{'Std(Scaled)':<15}")

for i, col in enumerate(columns):
    print(f"{col:<20}{np.mean(X_scaled[:,i]):<15.2f}{np.std(X_scaled[:,i]):<15.2f}")

#TEST 668
test_index = 668
test_point = X_scaled[test_index]

X_train = np.delete(X_scaled, test_index, axis=0)
y_train = np.delete(y, test_index)

#DISTANCE FUNCTION
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

distances = []

for i in range(len(X_train)):
    d = euclidean_distance(test_point, X_train[i])
    distances.append((i, d, int(y_train[i])))

distances.sort(key=lambda x: x[1])

print("\n=== DISTANCE TABLE (TOP 10) ===")
print(f"{'Rank':<5}{'Index':<10}{'Distance':<15}{'Class':<10}")

for i, (idx, dist, label) in enumerate(distances[:10], start=1):
    print(f"{i:<5}{idx:<10}{dist:<15.4f}{label:<10}")

#KNN WHERE K=5
k = 5
neighbors = distances[:k]

votes = [label for _, _, label in neighbors]
prediction = max(set(votes), key=votes.count)

print("\n=== K=5 NEIGHBORS ===")
print(f"{'Index':<10}{'Distance':<15}{'Class':<10}")

for idx, dist, label in neighbors:
    print(f"{idx:<10}{dist:<15.4f}{label:<10}")

print("\n=== FINAL PREDICTION ===")
print(f"Test Instance: {test_index}")
print(f"Predicted Class: {prediction} ({'Diabetic' if prediction==1 else 'Non-diabetic'})")

#Visualization
glucose_idx = 1
bmi_idx = 5

plt.figure(figsize=(8,6))

# Class 0
plt.scatter(X_scaled[y==0][:,glucose_idx],
            X_scaled[y==0][:,bmi_idx],
            alpha=0.4, label="Non-diabetic (Class 0)")

# Class 1
plt.scatter(X_scaled[y==1][:,glucose_idx],
            X_scaled[y==1][:,bmi_idx],
            alpha=0.4, label="Diabetic (Class 1)")

# Test point
plt.scatter(test_point[glucose_idx], test_point[bmi_idx],
            marker='*', s=200,
            label=f"Test Instance {test_index} (Pred: {prediction})")

# Highlight neighbors
for idx, _, _ in neighbors:
    point = X_train[idx]
    plt.scatter(point[glucose_idx], point[bmi_idx],
                edgecolors='black', s=120)

# Closest neighbor
closest_idx = neighbors[0][0]
closest_point = X_train[closest_idx]

plt.scatter(closest_point[glucose_idx], closest_point[bmi_idx],
            s=150, label=f"Nearest Neighbor {closest_idx}")

# Circle radius
radius = neighbors[-1][1]
circle = plt.Circle(
    (test_point[glucose_idx], test_point[bmi_idx]),
    radius,
    fill=False,
    linestyle='dashed',
    label="K=5 Neighborhood"
)

plt.gca().add_patch(circle)

plt.xlabel("Standardized Glucose")
plt.ylabel("Standardized BMI")
plt.title("KNN Feature Space: Glucose vs BMI (Standardized)")
plt.legend()
plt.grid()

plt.show()