"""Module to learn Linear Regression Algorithm
"""
import matplotlib.pyplot as plt

FILE_PATH = "./portlandOregonHousingPrices.txt"
ALPHA = 0.0000001

class LinearRegression(object):
    """ Class defined to simulate the linear regression algorithm.

        We use the gradient descent optimisation technique to train.

    """
    def __init__(self, file_path, alpha):
        """ The constructor method of the class.
        """
        self.data = {
            "x1": [],
            "x2": [],
            "y": [],
            "hx": []
        }
        self.theta = [71270,1]
        self.alpha = alpha
        self.m = None
        self.cost = {"theta0":[], "theta1":[], "cost": []}

        self.fetch_data(file_path)
        self.set_input_plot()

    def fetch_data(self, file_path):
        """ Reads the input data from the file.

            Reads the data and sets the data class variables.

            Args: file_path(str): A string representing the filepath of 
                                  input file.
        """
        with open(file_path, 'r') as file:
            data = file.readlines()

        for index, item in enumerate(data):
            data = item.replace("\n", "").replace(" ", "").split(",")
            self.data["x1"].append(int(data[0]))
            self.data["x2"].append(int(data[1])) 
            self.data["y"].append(int(data[2])) 

        self.m = len(self.data["x1"])

    def set_input_plot(self):
        """ Sets the values of the input graph plot.
        """
        plt.figure(figsize = (15,4),dpi=100)
        values = [121, "Size of house (X1)", "Price (Y)", self.data["x1"], self.data["y"]]
        self.plot_graph(values)

    def plot_graph(self, values):
        """ Plots the graph with the given values.

            Sets the subplot values, labels and plot values.

            Args 
                values(list): A list of values to plot the graph.
        """
        plt.subplot(values[0])
        plt.xlabel(values[1])
        plt.ylabel(values[2])
        plt.scatter(values[3], values[4])

    def get_hypothesis(self, index):
        """ Returns the value of hypothesis function.

            Inputs the index to get the theta and data values and calculates
            the hypothesis function.

            Args
                index(int): Denotes the row in the table to fetch the values.
        """
        return self.theta[0] + self.theta[1]*self.data["x1"][index]

    def set_cost(self):
        """ Sets the cost class variable.

            Calculates the cost of each row and stores it in a list.
        """
        cost = 0
        self.data["hx"] = []
        for index, item in enumerate(self.data["x1"]):
            hx = self.get_hypothesis(index)
            self.data["hx"].append(hx)
            cost += (hx - self.data["y"][index]) ** 2

        cost /= 2.0 * self.m
        self.cost["theta0"].append(self.theta[0])
        self.cost["theta1"].append(self.theta[1])
        self.cost["cost"].append(cost)

    def calculate_descent(self):
        """ Returns the decent values.

            Calculates the decent value to modify theta for the next iteration.
        """
        sum_of_difference = 0
        difference_times_x = 0
        for index, item in enumerate(self.data["x1"]):
            difference = self.data["hx"][index] - self.data["y"][index]
            sum_of_difference += difference
            difference_times_x += difference * self.data["x1"][index]
        
        descent0 = sum_of_difference * self.alpha / self.m
        descent1 = difference_times_x * self.alpha / self.m
        return [descent0, descent1]
        
    def gradient_descent(self):
        """ The gradient descent algorithm is run.

            Performs the gradient descent and optimises the theta value 
            in each iteration.
        """
        self.set_cost()
        descent = self.calculate_descent()
        limit = self.alpha/10000
        counter = 0
        while abs(descent[0]) > limit or  abs(descent[1]) > limit:
            if counter > 100:
                break
            else:
                counter += 1

            self.theta[0] -= descent[0]
            self.theta[1] -= descent[1]
            self.set_cost()
            descent = self.calculate_descent()


        print("theta0: ",self.theta[0], "\ntheta1: ", self.theta[1])
        print("\nInitial Cost: ", self.cost["cost"][0],"\nFinal Cost: ", self.cost["cost"][-1])
        values = [122, "Theta 1 Value", "Cost (J)", self.cost["theta1"], self.cost["cost"]]
        self.plot_graph(values)
        plt.subplot(121)
        plt.plot(self.data["x1"], self.data["hx"], "b")
        plt.show()

if __name__ == "__main__":
    lr = LinearRegression(FILE_PATH, ALPHA)
    lr.gradient_descent()