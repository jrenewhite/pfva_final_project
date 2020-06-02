#https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
import argparse

inputs = argparse.ArgumentParser()
inputs.add_argument("-p", "--processors", type=str, required=True,
				help="number of processors to use")
args = vars(inputs.parse_args())

from sklearn.decomposition import PCA
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = args["processors"]


A = np.array([  [10, 4, 7],
				[ 1, 6,11],
				[23, 2,14],
				[ 5, 8,12]])

print("\n 1. Input Matrix A:\n",A,"\n")


# ------------------- #
#|Method from SCRATCH|#
# ------------------- #
print("\n 2. Method from SCRATCH\n")
M = np.mean(A,axis=0)
print(" Mean Matrix:\n",M,"\n")

C = A-M
print(" Mean Centered Matrix:\n",C,"\n")

V = np.cov(C.T)
print(" Covariance Matrix:\n",V,"\n")

values,vectors = np.linalg.eig(V)
print(" Eigenvectors:\n",vectors,"\n")
print(" Eigenvalues (unsorted):\n",values,"\n")

P = vectors.T.dot(C.T)
print(" Matrix A Transformed into its PCA:\n",P.T,"\n")


# -------------------- #
#|Method using SKLEARN|#
# -------------------- #
print("\n 3. Method using SKLEARN\n")
pcaDimensions = len(A[0])
pca = PCA(pcaDimensions)
pca.fit(A)

print(" Eigenvectors:\n",pca.components_,"\n")
print(" Eigenvalues (sorted):\n",pca.explained_variance_,"\n")

B = pca.transform(A)
print(" Matrix A Transformed into its PCA (D = %d):\n"%(pcaDimensions),B,"\n")
