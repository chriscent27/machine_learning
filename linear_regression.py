"""Module to learn Linear Regression Algorithm
"""
import matplotlib.pyplot as plt

FILE_PATH = "./portlandOregonHousingPrices.txt"

class LinearRegression(object):
	""" Class defined to simulate the linear regression algorithm.

		We use the gradient descent optimisation technique to train.

	"""
	def __init__(self, file_path):
		self.data = {
			"x1":[], "x2":[], "y":[], "hx":[]}
		self.theta = [0,1]
		self.alpha = 0.0001
		self.m = None
		self.cost = {"theta0":[], "theta1":[], "cost": []}

		self.fetch_data(file_path)
		self.set_input_plot()

	def fetch_data(self, file_path):
		data = []

		with open(file_path, 'r') as file:
			data = file.readlines()

		for index, item in enumerate(data):
			data = item.replace("\n", "").replace(" ", "").split(",")
			self.data["x1"].append(int(data[0]))
			self.data["x2"].append(int(data[1])) 
			self.data["y"].append(int(data[2])) 

		self.m = len(self.data["x1"])

	def set_input_plot(self):
		plt.figure(figsize = (15,4),dpi=100)
		values = [121, "Size of house (X1)", "Price (Y)", self.data["x1"], self.data["y"]]
		self.plot_graph(values)


	def plot_graph(self, values):
		plt.subplot(values[0])
		plt.xlabel(values[1])
		plt.ylabel(values[2])
		plt.scatter(values[3], values[4])

	def get_hypothesis(self, index):
		return self.theta[0] + self.theta[1]*self.data["x1"][index]

	def set_cost(self):
		cost = 0
		self.data["hx"] = []
		for index, item in enumerate(self.data["x1"]):
			hx = self.get_hypothesis(index)
			self.data["hx"].append(hx)
			cost += (hx - self.data["y"][index]) ** 2
			print("hx", hx,"y", self.data["y"][index])

		cost /= 2.0 * self.m
		self.cost["theta0"].append(self.theta[0])
		self.cost["theta1"].append(self.theta[1])
		self.cost["cost"].append(cost)
		# values = [122, "Number of Bedrooms (X2)", "Price (Y)", self.data["x2"], self.data["y"]]
		# self.plotGraph(values)

	def calculate_descent(self):
		sum_of_difference = 0
		difference_times_x = 0
		for index, item in enumerate(self.data["x1"]):
			difference = self.data["hx"][index] - self.data["y"][index]
			sum_of_difference += difference
			difference_times_x += difference * self.data["x1"][index]
		print ("sum_of_difference, difference_times_x", sum_of_difference, difference_times_x)
		descent0 = sum_of_difference * self.alpha / self.m
		descent1 = difference_times_x * self.alpha / self.m
		print("[descent0, descent1] ",[descent0, descent1])
		return [descent0, descent1]
		
	def gradient_descent(self):
		self.set_cost()
		descent = []
		descent = self.calculate_descent()
		limit = 0.00001
		counter = 0
		while abs(descent[0]) > limit or  abs(descent[1]) > limit:
			if counter > 1:
				break
			else:
				counter += 1

			self.theta[0] -= descent[0]
			self.theta[1] -= descent[1]
			self.set_cost()
			descent = self.calculate_descent()
			print("data",self.data)
			print("self.theta[0]",self.theta[0], "self.theta[1]", self.theta[1])
			print("self.cost", self.cost)


		values = [122, "Theta Value", "Cost (J)", self.cost["theta1"], self.cost["cost"]]
		self.plot_graph(values)
		plt.show()

if __name__ == "__main__":
	lr = LinearRegression(FILE_PATH)
	lr.gradient_descent()