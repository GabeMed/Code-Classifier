from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from preprocess import X, y

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Criação do classificador RandomForest
clf = RandomForestClassifier(n_estimators=100)

# Treinamento do modelo
clf.fit(X_train, y_train)

# Previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Cálculo e exibição da acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Matriz de confusão
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
