Nearest Neighbor Search

by Eric Riddoch
Oct 20, 2018

Description:

    (A) Nearest Neighbor Problem:

    In this class .py file, we solve the Nearest Neighbor problem:
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