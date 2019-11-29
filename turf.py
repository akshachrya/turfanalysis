import numpy as np
import pandas as pd
from collections import namedtuple
from itertools import combinations
from functools import total_ordering

class Product(object):
    """A class representing each product in the test."""
    def __init__(self, label, reach=None, frequency=None):
        self.label = label
        self.reach = reach
        self.frequency = frequency

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label


@total_ordering
class ProductLine(object):
    """A class representing a set of products in the test"""
    def __init__(self, products, reach, frequency):
        self.products = products
        self.reach = reach
        self.frequency = frequency

    def __str__(self):
        #return "reach: {} | freq: {}".format(self.reach, self.frequency)
        #return self.product_names
        return ",".join([str(p.encode('utf-8')) for p in self.products])

    def __repr__(self):
        return ",".join([str(p.encode('utf-8')) for p in self.products])

    def __eq__(self, other):
        return self.reach == other.reach and self.frequency == other.frequency

    def __lt__(self, other):
        if self.reach == other.reach:
            return self.frequency < other.frequency
        return self.reach < other.reach


def turf_analysis(raw_data, subset_size, objective, product_labels=None):
    # raw_data=raw_data.transpose()
    data = np.array(raw_data)
    # print(data)


    # Check for missing data
    if (np.isnan(data)).any():
        raise ValueError('Input array cannot have missing data.')

    # Number of consumers, number of products
    b, v = data.shape
    

    if objective[0] > objective[1]:
        raise ValueError('The lower bound of the objective range is greater or\
            equal to the upper bound')

    obj_tuple = objective[0], objective[1]

    # Validate subset_size
    if subset_size < 1 or subset_size > v:
        raise ValueError('Invalid subset size: %d.' % subset_size)

    # Product labels
    if product_labels is None:
        product_labels = [str(i) for i in range(v)]
    elif len(product_labels) < v:
        raise ValueError('The are fewer labels than products.')
    else:
        product_labels = product_labels[:v]

    reached = np.logical_and(data > obj_tuple[0], data== obj_tuple[1])
    products = column
    
    top = list()
    product_lines = []
    frequency=[]
    reachh=[]
    
    tures=np.array([True]*subset_size)
    falses=np.array([False]*subset_size)
    

    for p in combinations(range(v), subset_size):
        count=reached[:, p]
        final=list(np.unique(count, axis=0, return_counts=True))
        topic=np.array(final[0])
        value=np.array(final[1])
        countes=[]
        reos=0

        for i in range(len(value)):
           
            checker=np.array_equal(topic[i],tures)
            falser=np.array_equal(topic[i],falses)

            if checker==True:
                freqos=value[i]
                countes.append(freqos)  
                
            if checker==False:
                freqos=0
                countes.append(freqos)
            
            if falser==True:
                reos+=0
             
            if falser==False:
                reos+=value[i]
   
        frequency.append(countes[-1])
        reachh.append(reos)
        product_line = ProductLine([products[i] for i in p],count,frequency)
        product_lines.append(product_line)

    table=[product_lines,frequency]
    table1=[product_lines,reachh]
    return [table,table1]



data=pd.read_csv('TURF_Level1.csv')
dataset = pd.DataFrame(data)
column=list(dataset.columns)
newdataset=dataset.dropna(how='all').replace('yes',1).fillna(0)
totalrespon,columnssize=newdataset.shape


a=turf_analysis(newdataset, subset_size=i, objective=[0,1],product_labels=column)[0]
table=pd.DataFrame(a).T
table.columns=['Configurations','Frequency']
table['Frequency %']=table['Frequency']/len(newdataset)*100
table=table.sort_values(by ='Frequency',ascending=False)
b=turf_analysis(newdataset, subset_size=i, objective=[0,1],product_labels=column)[1]
table1=pd.DataFrame(b).T
table1.columns=['Configurations','Reach']
table1['Reach %']=table1['Reach']/len(newdataset)*100
table1=table1.sort_values(by ='Reach',ascending=False)
print("Frequency Table")
print(table)
print("Reach Table")
print(table1)
print('\n')
