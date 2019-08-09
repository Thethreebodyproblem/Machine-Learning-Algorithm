import os
import pandas as pd
import numpy as np
import time

def preprocessing(dataset,column_names):
    dataset[10] = pd.cut(dataset[10],3, labels=['low_gain', 'medium_gain', 'high_gain'])
    dataset[11] = pd.cut(dataset[11],2, labels=['low_loss', 'high_loss'])
    dataset[0] = pd.cut(dataset[0], 4,  labels=['good_age', 'excellent_age', 'bad_age','no_age'])
    
    dataset.columns = column_names
    column_object =list( dataset.columns[dataset.dtypes=='object'])
    column_object.extend(['education-num', 'capital-gain', 'capital-loss', 'hours-per','age'])
    
    edu_num_cat = {i:'edu_num'+str(j)  for j, i in enumerate(set(dataset['education-num']))}
    dataset['education-num'] = dataset['education-num'].map(edu_num_cat)
#     capital_gain_cat = {i:'capital-gain'+str(j)  for j, i in enumerate(set(dataset['capital-gain']))}
#     dataset['capital-gain'] = dataset['capital-gain'].map(capital_gain_cat)
#     capital_loss_cat = {i:'capital-loss'+str(j)  for j, i in enumerate(set(dataset['capital-loss']))}
#     dataset['capital-loss'] = dataset['capital-loss'].map(capital_loss_cat)
    hours_per_cat = {i:'hours-per'+str(j)  for j, i in enumerate(set(dataset['hours-per']))}
    dataset['hours-per'] = dataset['hours-per'].map(hours_per_cat)
    dataset = dataset[column_object]

    return dataset
def scanD(Dataset, Ck, minSupport):
    '''
    Dataset: data in DataFrame
    Ck: candidate in list
    minSupport: minimum support 
    
    Return:
        retList: item in list meet minimum support
        support_data: contain all of the items
    '''
    dict_item = {}
    sample_length = Dataset.shape[0]
    min_support = minSupport*sample_length
    for item in Ck:
        dict_item[tuple(item)] = dict_item.get(tuple(item), 0)
        for sample in Dataset.values:
            if set(item).issubset(set(sample)):
                dict_item[tuple(item)] += 1
                
    return {i:j for i,j in dict_item.items() if j>min_support}


def comb(llist, k):
    Len = len(llist)
    llist=list(llist)
    newList =[]
    for i in range(Len):
        for j in range(i+1,Len):
            L1 = list(llist[i])[:k-2]
            L2 = list(llist[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                newList.append(set(llist[i])|set(llist[j]))
    return newList


def Apriori(dataset, miniSupport):
    C = []
    L = []
    k = 2
    t=time.time()
    C1 = (np.array(dataset).reshape(-1,))
    C1 = [{i} for i in list(set(C1))]
    L1 = scanD(dataset, C1, miniSupport )
    L.append(L1)
    print('begin to do while')
    while len(L[k-2])>0:
        temp_C = comb(L[k-2], k)
        temp_L = scanD(dataset, temp_C, miniSupport)
        k+=1
        L.append(temp_L)
        C.append(temp_C)
    print('Execution time: ', time.time()-t)
    return L, C, sum([len(i) for i in L])


def main():
    train = pd.read_csv('./adult.data.txt', header=None)
    test = pd.read_csv('./adult.test.txt', header=None)

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', \
                'capital-gain', 'capital-loss', 'hours-per', 'native-country', 'income']
    train_data = preprocessing(train, column_names)
    test_data = preprocessing(test, column_names)

    L_full_train, C_train, length_freque_sets_train = Apriori(train_data, 0.23)
    L_full_test, C_test, length_freque_sets_test = Apriori(test_data, 0.23)
    
    for j in L_full_train:
        for i in j:
            j[i] =j[i]/len(train_data)
    for j in L_full_test:
        for i in j:
            j[i] = j[i]/len(test_data)
            
    with open('Apriori_Frequent_items_train.txt', 'w') as f:
        f.write('Frequent Itemsets: Support')
        f.write('\n')
        for i in L_full_train:
            for j in i:
                f.write(str(j)+' : ' + str(i[j]))
                f.write('\n')
            
    with open('Apriori_Frequent_items_test.txt', 'w') as f:
        f.write('Frequent Itemsets: Support')
        f.write('\n')
        for i in L_full_test:
            for j in i:
                f.write(str(j)+' : ' + str(i[j]))
                f.write('\n')
    
if __name__ == '__main__':
    main()