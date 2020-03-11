import numpy as np
import pandas as pd

class NaiveBayes():
    def __init__(self, priors=None, laplace_smoothing_val=10**-9):
        self.priors = priors # array of classes' prior probability
        self.laplace_smoothing_val = laplace_smoothing_val # to avoid probability is 0.0
        self.classes = None # array of classes
        self.attributes = None # array of attributes
        self.likelihood_calculators = {} # dictionary of likelihood calculator for each attribute
    
    def fit(self, X, y):
        X_df, y_np = pd.DataFrame(X), np.array(y)
        
        # Get all possible classes
        classes, class_counts = np.unique(y_np, return_counts=True)
        if self.priors is None:
            # Update priors with sample probability
            self.priors = class_counts / len(y_np)
        self.classes = classes  
        
        # Build likelihood calculator for each attribute
        self.attributes = X_df.columns
        for attr in self.attributes:
            attr_type = X_df.dtypes[attr]
            if (attr_type == np.object) or (attr_type == np.bool): # Categorical 
                self.likelihood_calculators.update({attr: CategoricalLikelihoodCalculator(X_df, y_np, attr, self.classes, self.priors)})
            else: # Numerical
                self.likelihood_calculators.update({attr: NumericalLikelihoodCalculator(X_df, y_np, attr, self.classes, self.priors)})
        pass
    
    def test(self, X, y):
        X_df, y_np = pd.DataFrame(X), np.array(y)
        y_pred = self.predict(X_df)
        right_predictions = (y_pred == y_np)
        accuracy = right_predictions.sum() / len(y_np)
        return accuracy, y_pred
    
    def predict(self, X):
        X_df, y_df = pd.DataFrame(X), pd.DataFrame(np.zeros([len(X), len(self.classes)]))
        y_df.columns = self.classes
        
        # Get the probability for each class
        for index, class_val in enumerate(self.classes):
            # Transform each value of X to corresponding likelihood given current class
            X_likelihood = X_df.apply(lambda row: self.calculate_class_probability_of_row(row, class_val), axis = 1)
            
            # CLass probability 
            y_df.loc[:, class_val] = np.array(X_likelihood.prod(axis = 1)) * self.priors[index]
        
        # Prediction is class that has largest probability
        y_pred = y_df.idxmax(axis=1)
        
        return y_pred
    
    def calculate_class_probability_of_row(self, row, class_val):
        # Go through each cell in row
        for index, val in enumerate(row):
            # Update cell value with corresponding likelihood
            attribute_name = self.attributes[index]
            likelihood_calculator = self.likelihood_calculators[attribute_name]
            likelihood = likelihood_calculator.get_likelihood(val,class_val)
            row[index] = likelihood if likelihood > self.laplace_smoothing_val else self.laplace_smoothing_val
        return row
    
class LikelihoodCalculator(object):
    def __init__(self, X, y, attribute, classes=None, class_probs=None):        
        self.mean_variance_dict = {} # 'y_val': [mean of attribute when y_val, variance of attribute when y_val]   
        self.likelihood_dict = {} # 'X_val|y_val': P(X_val|y_val) 
        
        if (classes is None) or (class_probs is None):
            classes, class_counts = np.unique(y, return_counts=True)        
            class_probs = class_counts / len(y)
        self.calculate_likelihood(X, y, attribute, classes, class_probs)
    
    def calculate_likelihood(self, X, y, attribute, classes, class_probs):
        pass
    
    def get_likelihood(self, X_val, y_val):
        pass
    
class CategoricalLikelihoodCalculator(LikelihoodCalculator):
    def calculate_likelihood(self, X, y, attribute, classes, class_probs):
        # Get all possible value of this attribute
        X_unique_vals = np.unique(X[attribute])
        X_len = len(X)
        
        # Calculate likelihood for each pair value of X and y
        for class_index, class_val in enumerate(classes):
            for X_val in X_unique_vals:
                X_filtered = X.loc[(y == class_val) * (X[attribute] == X_val), attribute]
                probability_X_and_y = len(X_filtered) / X_len
                likelihood = probability_X_and_y / class_probs[class_index]
                
                # Add likelihood of this value pair into dictionary for future retrival
                self.likelihood_dict.update({'{}|{}'.format(X_val, class_val): likelihood})
    
    def get_likelihood(self, X_val, y_val):
        # Get corresponding likelihood based on pair value of X and y
        lookup_key = '{}|{}'.format(X_val, y_val)
        return self.likelihood_dict.get(lookup_key, 0.0)
    
class NumericalLikelihoodCalculator(LikelihoodCalculator):
    def calculate_likelihood(self, X, y, attribute, classes, class_probs):
        # Prepare mean and standard deviation of this attribute for each class
        for class_val in classes:
            X_filtered = X.loc[y == class_val, attribute]
            mean = X_filtered.values.mean()
            std = np.std(X_filtered.values, ddof=1)
            
            # Add the mean and standard deviation pair into dictionary for future retrival
            self.mean_variance_dict.update({class_val: [mean, std]})
    
    def get_likelihood(self, X_val, y_val):
        # Calculate Gaussian probability
        [mean, std] = self.mean_variance_dict[y_val]
        prob = np.exp(-(((X_val - mean) / (std * np.sqrt(2))) ** 2)) / (std * np.sqrt(2 * np.pi)) # Normal's pdf
        return prob