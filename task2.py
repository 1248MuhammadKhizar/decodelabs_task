from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print("Dataset Shape:", X.shape)
print("Classes:", target_names)
print("Samples per class:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

error_rates = []
k_range = range(1, 31)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - f1_score(y_test, preds, average='weighted'))

plt.figure(figsize=(10, 5))
plt.plot(k_range, error_rates, marker='o', color='navy')
plt.title('Tuning K: Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.savefig('k_tuning_elbow.png', dpi=150)
plt.close()
print("Elbow plot saved as k_tuning_elbow.png")

optimal_k = k_range[np.argmin(error_rates)]
print(f"\nOptimal K from elbow method: {optimal_k}")

model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

f1 = f1_score(y_test, predictions, average='weighted')
print(f"\nF1 Score (weighted): {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=target_names))

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names
)
plt.title(f'Confusion Matrix (K={optimal_k})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix saved as confusion_matrix.png")

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample)
result = model.predict(sample_scaled)
print(f"\nSample prediction for {sample[0]}: {target_names[result[0]]}")