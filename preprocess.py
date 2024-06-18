import ast
import os
from ast2vec import ast2vec
from ast2vec import python_ast_utils

path = "./data"
trees = []  # A list where we are going to store the AST's
X = []  # A list with all the vectors
y = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
]  # Vectors representing the labels

# Reading all the codes as .txt and parsing them
for programs in os.listdir(path):
    with open(path + "/" + programs, "r") as file:
        program = file.read()
    trees.append(python_ast_utils.ast_to_tree(ast.parse(program)))

# Loading the model
model = ast2vec.load_model()

# Generating the code vectors and storing in X
for tree in trees:
    X.append(model.encode(tree).detach().numpy())
