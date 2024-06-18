import ast
import os
from ast2vec import ast2vec
from ast2vec import python_ast_utils
import numpy as np

# Configurações de caminho e inicialização de listas
path = "./data"
trees = []
X = []
y = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
]

# Leitura dos códigos e parsing para AST
for file_name in os.listdir(path):
    if file_name.endswith(".py"):  # Verifica se o arquivo é Python
        with open(os.path.join(path, file_name), "r") as file:
            program = file.read()
        trees.append(python_ast_utils.ast_to_tree(ast.parse(program)))

# Carregamento do modelo
model = ast2vec.load_model()

# Geração dos vetores de código
for tree in trees:
    X.append(model.encode(tree).detach().numpy())

# Convertendo para tipo numpy array
y = np.array(y)
X = np.array(X)
