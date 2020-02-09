##################################################################################
# import matplotlib/fit a graph/show it
import matplotlib.pyplot as plt
plt.plot([1,2,3], [4,5,6])
plt.show()
##################################################################################
# Labels/Legends/titles
x = [1, 2, 3]
y = [4, 2, 1]
x2 = [1, 2, 3]
y2 = [14, 12, 11]

plt.plot(x1, y2, label='First line')
plt.plot(x2, y2, label='Second line')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Graph title\nSubtitle')
plt.legend()
plt.show()
##################################################################################
# Barcharts
x = [2 , 4, 6, 8, 10]
y  = [6,7,8,2,4]
plt.bar(x, y, label='Bars1', color='r')
##################################################################################
# Histograms
