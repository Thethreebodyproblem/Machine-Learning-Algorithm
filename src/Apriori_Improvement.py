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
    dict_item_samples = {}
    sample_length = Dataset.shape[0]
    min_support = minSupport*sample_length
    for item in Ck:
        dict_item[tuple(item)] = dict_item.get(tuple(item), 0)
        dict_item_samples[tuple(item)] = []
        for index, sample in enumerate(Dataset.values):
            if set(item).issubset(set(sample)):
                dict_item[tuple(item)] += 1
                dict_item_samples[tuple(item)].append('sample_'+str(index))
    x = {i:j for i,j in dict_item.items() if j>min_support}
    sample_list = {}
    for select_item in x:
        if select_item in dict_item_samples:
            sample_list[select_item]=dict_item_samples[select_item]
    return x, sample_list


def comb(llist, llist2, k, dataset, miniSupport):
    Len = len(llist)
    llist=list(llist)
    newList =[]
    new_sample_list = []
    for i in range(Len):
        for j in range(i+1,Len):
            L1 = list(llist[i])[:k-2]
            L2 = list(llist[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                newList.append(set(llist[i])|set(llist[j]))
                new_sample_list.append(set(llist2[llist[i]])&set(llist2[llist[j]]))
    ddd = {}
    eee = {}
    key_length = [len(i) for i in new_sample_list]
    miniS = dataset.shape[0]*miniSupport
    for i in range(len(newList)):
        if key_length[i] > miniS:
            ddd[tuple(newList[i])] = key_length[i]
            eee[tuple(newList[i])] = new_sample_list[i]

    return ddd, eee

def Apriori(dataset, miniSupport):
    L1, L2 = [], []
    k = 2
    t = time.time()
    C1 = (np.array(dataset).reshape(-1,))
    C1 = [{i} for i in list(set(C1))]
    L11, L12 = scanD(dataset, C1, miniSupport )
    L1.append(L11)
    L2.append(L12)
    while len(L1[k-2])>0:
        temp_L1, temp_L2 = comb(L1[k-2], L2[k-2], k, dataset, miniSupport)
#         temp_L1, temp_L2 = scanD(dataset, temp_C, miniSupport)
        k+=1
        L1.append(temp_L1)
        L2.append(temp_L2)
    print('Execution time: ', time.time()-t)
    return L1, L2, sum([len(i) for i in L1])


def main():
    train = pd.read_csv('./adult.data.txt', header=None)
    test = pd.read_csv('./adult.test.txt', header=None)

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', \
                'capital-gain', 'capital-loss', 'hours-per', 'native-country', 'income']
    train_data = preprocessing(train, column_names)
    test_data = preprocessing(test, column_names)

    L_full_train, C_full_train, length_freque_sets_train = Apriori(train_data, 0.23)
    L_full_test, C_full_test, length_freque_sets_test = Apriori(test_data, 0.23)
    
    for j in L_full_train:
        for i in j:
            j[i] =j[i]/len(train_data)
    for j in L_full_test:
        for i in j:
            j[i] = j[i]/len(test_data)
            
    with open('Apriori_Improved_Freqitems_train.txt', 'w') as f:
        f.write('Frequent Itemsets: Support')
        f.write('\n')
        for i in L_full_train:
            for j in i:
                f.write(str(j)+' : ' + str(i[j]))
                f.write('\n')
            
    with open('Apriori_Improved_Freqitems_test.txt', 'w') as f:
        f.write('Frequent Itemsets: Support')
        f.write('\n')
        for i in L_full_test:
            for j in i:
                f.write(str(j)+' : ' + str(i[j]))
                f.write('\n')
if __name__ == '__main__':
    main()