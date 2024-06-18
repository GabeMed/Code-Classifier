import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from preprocess import X, y

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Criação do classificador RandomForest
clf = RandomForestClassifier(n_estimators=100)

# Definição do método de validação cruzada
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Avaliação do modelo usando validação cruzada
scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

# Impressão dos scores
print(f"Scores de validação cruzada: {scores}")
print(f"Mean accuracy: {scores.mean():.2f}")
print(f"Standard deviation: {scores.std():.2f}")

# Plotar a distribuição dos scores
plt.figure()
plt.plot(range(1, len(scores) + 1), scores, marker="o", linestyle="--")
plt.title("Cross-Validation Accuracy Scores")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid()
plt.show()
