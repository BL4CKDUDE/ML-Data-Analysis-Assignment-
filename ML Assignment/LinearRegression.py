""" the goal of this file was to convert data into classification problem
    then apply linear regression(gradient descent
"""
Data_File = open('Admissions.txt', 'r')  # open the file contents
Data_String = Data_File.read()  # read the contents into a string
full_data = Data_String.splitlines()  # storing each datapoint into a list

full_data2 = []     # initiate a matrix to store the data in a discrete manner
for data_point in full_data:
    data_point = data_point.split(',')

    group = data_point[len(data_point) - 1]  # get the probability of admission

    bias = 0.65
    if float(group) > bias:  # assert whether the probability is above threshold
        group = '1'  # group one represents accepted class
    else:
        group = '0'  # group zero represents rejected class

    data_point.pop(len(data_point) - 1)     # remove the probability value
    data_point.append(group)        # replace it discrete value
    full_data2.append(data_point)

print(full_data2)
print(full_data)

# TODO check appropriate basis functions and modify the design matrix prior to applying linear regression to the data
