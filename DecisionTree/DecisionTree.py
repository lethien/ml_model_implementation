import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, criterion='entropy', max_depth=20):
        self.tree = None
        self.criterion = criterion
        self.max_depth = max_depth
    
    def fit(self, X, y):
        X_df, y_np = pd.DataFrame(X), np.array(y)        
        self.tree = self.build_node(X_df, y_np, np.array(X_df.columns), 'Node 0.', 'Root node', None, 0)
    
    def test(self, X, y):
        X_df, y_np = pd.DataFrame(X), np.array(y)
        y_pred = np.array(self.predict(X_df))
        right_predictions = (y_pred == y_np)
        accuracy = right_predictions.sum() / len(y_np)
        return accuracy, y_pred
    
    def predict(self, X):
        X_df, y_df = pd.DataFrame(X), pd.DataFrame({'classes': np.zeros(len(X))})
        self.assign_classes(X_df, y_df, self.tree, None)
        return y_df['classes']
    
    def build_node(self, X, y, available_attributes, node_name, node_input_description, input_function, current_depth):
        # Initialize a variable for current node
        current_node = Node(node_name, node_input_description, input_function)
        # Current impurity value
        current_impurity = self.calculate_impurity(y)
        
        # Check for stop conditions:
        #   - data is pure, all objects belong to one class
        #   - used all available attributes to split
        #   - reach max depth
        if (current_impurity == 0) or (len(available_attributes) == 0) or (current_depth >= self.max_depth):
            current_node.is_leaf = True            
            current_node.classified_as = self.get_mode(y)
        # If stop condition not met, continue spliting from this node        
        else:
            # Find the best attribute to split
            best_attr = available_attributes[0] # Initialize with the first attribute, will likely be updated after
            best_attr_split_val = -1 # Holder of split threshold for numerical attribute
            best_gain = 0
            for attr in available_attributes:
                gain_ratio, value_to_split_by = self.calculate_gain_ratio(X, y, attr, current_impurity) # Calculate gain ratio
                if gain_ratio >= best_gain: # Update best attribute to split
                    best_attr = attr
                    best_gain = gain_ratio
                    best_attr_split_val = value_to_split_by
            current_node.split_attribute = best_attr
            
            # Split current node into child nodes by the chosen attribute
            if best_attr_split_val == -1: # Categorical attribute
                unique_vals = np.unique(X[best_attr])
                for i, unique_val in enumerate(unique_vals):
                    # Filter out the data suitable for each child node
                    split_func = self.get_split_function(best_attr, unique_val, 'eq')
                    X_filtered = X[split_func(X)]
                    y_filtered = y[split_func(X)]  
                    
                    # Build the child node (recursively) and add to the current node's children list
                    child_node = self.build_node(X_filtered, y_filtered, np.setdiff1d(available_attributes, [best_attr]), 
                                           node_name + str(i + 1) + '.', best_attr + '==' + str(unique_val), split_func, 
                                                 current_depth + 1)
                    current_node.children = np.append(current_node.children, child_node)
            else: # Numerical attribute
                # Filter out the data suitable for left (val < threshold) child node
                split_func_left = self.get_split_function(best_attr, best_attr_split_val, 'lt')
                X_filtered_left = X[split_func_left(X)]
                y_filtered_left = y[split_func_left(X)]
                
                # *** If there is no object left after filtered, stop the split and make current node a leaf node
                if(len(y_filtered_left) == 0):
                    current_node.is_leaf = True            
                    current_node.classified_as = self.get_mode(y)
                else:
                    # Build the left child node (recursively) and add to the current node's children list
                    child_node_left = self.build_node(X_filtered_left, y_filtered_left, np.setdiff1d(available_attributes, [best_attr]), 
                                           node_name + '1.', best_attr + '<' + str(best_attr_split_val), split_func_left, 
                                                     current_depth + 1)
                    current_node.children = np.append(current_node.children, child_node_left)

                    # Filter out the data suitable for right (val >= threshold) child node
                    split_func_right = self.get_split_function(best_attr, best_attr_split_val, 'ge')
                    X_filtered_right = X[split_func_right(X)]
                    y_filtered_right = y[split_func_right(X)]

                    # Build the right child node (recursively) and add to the current node's children list
                    child_node_right = self.build_node(X_filtered_right, y_filtered_right, np.setdiff1d(available_attributes, [best_attr]), 
                                           node_name + '2.', best_attr + '>=' + str(best_attr_split_val), split_func_right, 
                                                      current_depth + 1)
                    current_node.children = np.append(current_node.children, child_node_right)             
            
        return current_node

    def assign_classes(self, X, y, current_node, apply_condition):
        # Check for stop condition: current node is a leaf node
        if current_node.is_leaf:
            # Update the labels as this node classified
            y[apply_condition] = current_node.classified_as
        else:
            # Continue to go down the tree branches and update the apply condition
            for child_node in current_node.children:
                if apply_condition is None:
                    self.assign_classes(X, y, child_node, child_node.input_function(X).values)
                else:
                    self.assign_classes(X, y, child_node, apply_condition * child_node.input_function(X).values)
    
    def calculate_impurity(self, y):
        impurity = 1 # Initialize impurity value
        
        # Calculate the proportion of all possible classes
        unique, counts = np.unique(y, return_counts=True)
        ps = counts / counts.sum()
        
        # Calculate impurity
        if self.criterion == 'entropy':
            es = - ps * np.log2(ps, where = (ps > 0.0), out=np.zeros_like(ps))
            impurity = es.sum()
        elif self.criterion == 'gini':            
            impurity = 1 - (np.square(ps)).sum()        
        return impurity
    
    def calculate_gain_ratio(self, X, y, attr, current_impurity):
        # Initialize variables for calculating gain ratio
        gain = current_impurity
        split_information = 0
        value_to_split_by = -1 # Init with -1 and keep being -1 for categorical attribute, will be updated if numerical
        
        attr_type = X.dtypes[attr]
        if (attr_type == np.object) or (attr_type == np.bool): # Categorical 
            unique_vals, counts = np.unique(X[attr], return_counts=True)
            ps = counts / counts.sum() # proportion of all possible values of this attribute
            for i, unique_val in enumerate(unique_vals): # calculate impurity reduction for each possible values
                gain = gain - ps[i] * self.calculate_impurity(y[X[attr] == unique_val])
            split_information = split_information - (ps * np.log2(ps, where = (ps > 0.0), out=np.zeros_like(ps))).sum()
        else: # Numerical
            # min to max iteration through all possible values to find the best value to split
            values_sorted = np.sort(np.unique(X[attr]))
            value_to_split_by = values_sorted[0]
            best_reduction = 0            
            for i, val in enumerate(values_sorted[1:len(values_sorted)]): # for each value, try to use as a threshold to split
                y_ge = y[X[attr] >= val] # attribute value in [val, max]
                y_lt = y[X[attr] < val]  # attribute value in [min, val)
                ps = np.array([len(y_ge) / len(y), len(y_lt) / len(y)]) # Proportion of each region
                reduction = ps[0] * self.calculate_impurity(y_ge) + ps[1] * self.calculate_impurity(y_lt) # Calculate impurity reduction
                if reduction >= best_reduction: # Update best value to split
                    best_reduction = reduction
                    value_to_split_by = val
                    split_information = split_information - (ps * np.log2(ps, where = (ps > 0.0), out=np.zeros_like(ps))).sum()
        
        # Return the gain ratio value
        gain_ratio = gain / split_information
        return gain_ratio, value_to_split_by
    
    def get_mode(self, y):
        unique, counts = np.unique(y, return_counts=True)
        mode = unique[np.argmax(counts)]
        return mode
    
    def get_split_function(self, attr, value, operator):
        split_func = None
        if operator == 'eq':
            split_func = lambda X_df: X_df[attr] == value
        elif operator == 'ge':
            split_func = lambda X_df: X_df[attr] >= value
        elif operator == 'lt':
            split_func = lambda X_df: X_df[attr] < value        
        return split_func
    
    def display_node(self, start_node=None):
        from_node = start_node if start_node else self.tree
        print(from_node.display_as_string())
        for child in from_node.children:
            self.display_node(child)
    
class Node:
    def __init__(self, name, input_description, input_function, is_leaf=False):
        self.name = name
        self.is_leaf = is_leaf
        self.input_description = input_description
        self.input_function = input_function
        self.split_attribute = None
        self.children = np.array([])
        self.classified_as = None
    
    def display_as_string(self):
        if self.is_leaf:
            string_format = '{} {}, Classified As: {}'
            return string_format.format(self.name, self.input_description, self.classified_as)
        else:
            string_format = '{} {}, Split Attribute: {}'
            return string_format.format(self.name, self.input_description, self.split_attribute)