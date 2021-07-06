import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a = np.loadtxt('../Dane/dane1.txt')

x = a[:,[0]]
y = a[:,[1]]

trainingSetX, testSetX, trainingSetY, testSetY = train_test_split(x, y, test_size=0.5)

c = np.hstack([trainingSetX, np.ones(trainingSetX.shape)]) # model 1 liniowy
c1 = np.hstack([trainingSetX**3, trainingSetX**2, trainingSetX, np.ones(trainingSetX.shape)]) # model x^3
#c2 = np.hstack([trainingSetX**4, trainingSetX**3, trainingSetX**2, trainingSetX, np.ones(trainingSetX.shape)]) # model x^4

v = np.linalg.pinv(c) @ trainingSetY
v1 = np.linalg.pinv(c1) @ trainingSetY
#v2 = np.linalg.pinv(c2) @ trainingSetY

eTraining1 = sum((trainingSetY - (v1[0] * trainingSetX + v1[1])) ** 2)/trainingSetY.size
eTest1 = sum((testSetY - (v1[0] * testSetX + v1[1])) ** 2)/testSetY.size
eTraining2 = sum((trainingSetY - (v1[0] * trainingSetX**3 + v1[1] * trainingSetX**2 + v1[2] * trainingSetX + v1[3]))**2)/trainingSetY.size
eTest2 = sum((testSetY - (v1[0] * testSetX**3 + v1[1] * testSetX**2 + v1[2] * testSetX + v1[3]))**2)/trainingSetY.size
#eTraining3 = sum(abs(trainingSetY - (v2[0] * trainingSetX**4 + v2[1] * trainingSetX**3 + v2[2] * trainingSetX**2 + v2[3] * trainingSetX + v2[4]))**3)/trainingSetY.size
#eTest3 = sum(abs(testSetY - (v2[0] * testSetX**4 + v2[1] * testSetX**3 + v2[2] * testSetX**2 + v2[3] * testSetX + v2[4]))**3)/trainingSetY.size

print("Training x(avg err) =", eTraining1, ", Test x(avg err) =", eTest1)
print("Training x^3(avg err) = ", eTraining2, "Test x^3(avg err) = ", eTest2)
#print("Training x^4(avg err) = ", eTraining3, "Test x^4(avg err) = ", eTest3)

plt.plot(x, y, 'ro')
plt.plot(x, v1[0] * x**3 + v[1] * x**2 + v1[2] * x + v1[3])
plt.plot(x, v[0] * x + v[1])
#plt.plot(x, v2[0] * x**4 + v1[0] * x**3 + v2[2] * x**2 + v1[3] * x + v2[3])
plt.show()

























