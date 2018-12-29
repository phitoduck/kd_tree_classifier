# nearest_neighbor.py

"""

Nearest Neighbor Search

by Eric Riddoch
Oct 20, 2018

Description:

    (A) Nearest Neighbor Problem:

    In this .py file, we solve the Nearest Neighbor problem:
    given a set of points in Euclidean space, if we pick a single point
    from the set, which other point is closest to the point we chose?

    We compare 2 methods:
        1) exhaustive search: calculate all of the distances!
        2) building a K neighbors classifier tree (a multi-dimensional binary tree)
    
    Method 2 requires us to assemble a data structure, so of course it is
    slower in the short term. However, provided we have the tree, the search
    goes from O(n) to O(logn) time.

    (B) Machine Learning Application:

    We want our computer to correctly label a digit given a 32x32 pixel image. 
    To do this, we stack each image into a 32**2 length vector, and treat them as
    nodes in scipy's version of a KD Tree.

    Once the tree is built from training data, we act as if we are going to insert
    a new node into the tree. Then, within a fixed distance, we count the number
    of 'neighbor' nodes in this proximity associated with each digit 0-9.

    Whichever digit has the most neighbors wins!




    File Contents:
        exaustive_search()   - brute force approach that calculates the distance
                               to all nodes in a set to a target node
        KDTNode              - individual node for KDTree
        KDTree               - tree with methods to find, insert, and query
        KNeighborsClassifier - Fits a training set of nodes with labels to an 
                               instance of scipy's KDTree. Has a predict function
                               to predict which label should be assigned to 
                               a new "target node"
        run()                - Uses a KNeighborsClassifier to build a KDTree
                               from our data set of digits and outputs the accuracy.

"""

import numpy as np
from scipy import linalg as la
import math
from collections import deque
from scipy.spatial import KDTree
from scipy.stats import mode

# for KDTree.draw()
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
from timeit import default_timer as time

def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """

    # minimum distance:
    d_star = math.inf

    # closest post office
    x_star = np.array([0, 0])

    # compute closest post office:
    for i in range(X.shape[0]):
        x = X[i, :]
        distance = la.norm(x - z)

        if distance < d_star:
            x_star = x
            d_star = distance

    return x_star, d_star

class KDTNode:
    """KDTNode simulates a node on a KD Tree"""

    def __init__(self, data):
        """Take in data and construct node"""

        if type(data) is not np.ndarray:
            raise TypeError("KDTNODE: Don't be dumb!")

        # Initialize
        self.value = data
        self.left = None
        self.right = None
        self.pivot = None

class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """

    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()

    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """

        # is the tree empty?
        if self.root == None:

            self.root = KDTNode(data)
            self.k = data.shape[0]
            self.root.pivot = 0

            return

        # is the proposed data valid?
        if data.shape[0] != self.k:
            raise ValueError(
                f"Tried to insert node of {data.shape[0]} into tree of {self.k}")

        # insert new node with iterative approach
        stack = deque()
        stack.append(self.root)

        while len(stack) > 0:
            current = stack.pop()

            if np.allclose(data, current.value):
                raise ValueError(f"{data} is already in the tree!")

            # left?
            if data[current.pivot] < current.value[current.pivot]:

                # insert?
                if current.left == None:
                    current.left = KDTNode(data)
                    current.left.pivot = (current.pivot + 1) % self.k
                    break
                else:
                    stack.append(current.left)

            # right?
            elif data[current.pivot] >= current.value[current.pivot]:

                # insert?
                if current.right == None:
                    current.right = KDTNode(data)
                    current.right.pivot = (current.pivot + 1) % self.k
                    break
                else:
                    stack.append(current.right)

    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """

        def KDSearch(current, nearest, d_star):

            # end of the tree!
            if current is None:
                return nearest, d_star

            x = current.value
            i = current.pivot
            distance = la.norm(x - z)

            # is current the closest so far?
            if distance < d_star:
                nearest = current
                d_star = distance

            # where to next?
            if z[i] < x[i]:
                nearest, d_star = KDSearch(current.left, nearest, d_star)

                # do we need to check the other path on our way back up?
                if z[i] + d_star >= x[i]:
                    nearest, d_star = KDSearch(current.right, nearest, d_star)

            else: #if z[i] >= x[i]:
                nearest, d_star = KDSearch(current.right, nearest, d_star)

                if z[i] - d_star <= x[i]:
                    nearest, d_star = KDSearch(current.left, nearest, d_star)

            return nearest, d_star

        nearest, d_star = KDSearch(self.root, self.root, la.norm(self.root.value - z))

        return nearest.value, d_star

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)

class KNeighborsClassifier:
    """Allows construction of a KD Tree to solve a machine learning problem."""

    def __init__(self, n_neighbors): 
        """
         @params: 
            n_neighbors (int): k nearest neighbors
        """
        self.k = n_neighbors
        self.tree = None
        self.labels = None

    def fit(self, X, y):
        """accept an m × k NumPy array X (the training set) and 
        a 1-dimensional NumPy array y with m entries 
        (the training labels). As in Problems 1 and 4, each of 
        the m rows of X represents a point in Rk. 
        Here yi is the label corresponding to row i of X.
        
        @params:
            X (m × k np.array): data points in R^k
            y (m × 1 np.array): label for each data point
            """
        
        # Build scipy.spatial.KDTree out of datapoints
        self.tree = KDTree(X)
        self.labels = y

    def predict(self, z):
        """
        Find the k nearest neighbors of z in the tree

        @params:
            z (1 × k np.array)
        """

        # indices: which nodes/labels are closest to z?
        _, indices = self.tree.query(z, self.k)

        # return most common label scipy.stats.mode()
        return mode(self.labels[indices])[0][0]

def view_img(X_test, n=1):
    """Displays n images from the data set"""
    for i in range(n):
        plt.imshow(X_test[i].reshape((28,28)), cmap="gray")
        plt.axis("off")
        plt.show()

def run(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """

    # Load in the data!
    data = np.load(filename)
    X_train = data["X_train"].astype(np.float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float)
    y_test = data["y_test"]

    # Try viewing a sample image
    view_img(X_test)

    # build KDTree out of test set and labels
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(X_train, y_train)

    total_accurate = 0
    total_train = len(y_test)

    # test each label
    for i in range(len(X_test)):
        test_target = X_test[i]

        # make prediction
        prediction = classifier.predict(test_target)

        # check accuracy
        if prediction == y_test[i]:
            total_accurate += 1
    
    # return accuracy ratio
    return total_accurate / total_train


# example
if __name__ == "__main__":
    run(10)



