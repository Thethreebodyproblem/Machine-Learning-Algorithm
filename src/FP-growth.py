import pandas as pd
import numpy as np
import time

train = pd.read_csv('./adult.data.txt', header=None)
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', \
                'capital-gain', 'capital-loss', 'hours-per', 'native-country', 'income']
test = pd.read_csv('./adult.test.txt', header=None)

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

class TreeNode:
    def __init__(self, nameValue, countValue, parentNode):
        self.name = nameValue
        self.count = countValue
        self.parent = parentNode
        self.children = {}
        
        self.nodeLink = None
    def inc(self, num):
        self.count += num
        
    def disp(self, ind = 1):
        print('  '*ind, self.name, '  ', self.count)
        for children in self.children.values():
            children.disp(ind+1)


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def createTree(dataset, minSup = 3):
    headerTable = {}
    for trans in dataset:
        for item in trans: # traverse of each element for each sample
            headerTable[item] = headerTable.get(item, 0)+dataset[trans]
    for k in list(headerTable):
        if headerTable[k]<minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    
    if len(freqItemSet) == 0:
        return None, None 
    
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
#     print('headerTable', headerTable)
    
    retTreee = TreeNode('Null', 1, None) # initilize
    for tranSet, count in dataset.items():
        localID = {}
        for item in tranSet:
            if item in freqItemSet:
                localID[item] = headerTable[item][0]
        
        if len(localID)>0:
            orderedItems = [v[0] for v in sorted(localID.items(), key = lambda x: x[1], reverse=True)]
            updateTree(orderedItems, retTreee, headerTable, count)
    
    return retTreee, headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = TreeNode(items[0], count, inTree)
#         print(inTree)
#         print('headerTable: ', headerTable[items[0]])
#         print('headerTable: ', headerTable[items[0]][1])
        if headerTable[items[0]][1]==None: # point to the item first time occur 
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items)> 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def ascendTree(leafNode, prefixPath):  
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def ascendTree(leafNode, prefixpath):
    if leafNode.parent != None:
        prefixpath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixpath)



def findPrefixPath(basePat, treeNode):  
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count 
        treeNode = treeNode.nodeLink
    return condPats

# recursively find freqitems
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # sort items ascending order
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    for basePat in bigL:  
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #print ('finalFrequent Item: ',newFreqSet)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #print ('condPattBases :',basePat, condPattBases)

        myCondTree, myHead = createTree(condPattBases, minSup)
        #print ('head from conditional tree: ', myHead)
        if myHead != None: 
#             print ('conditional tree for: ',newFreqSet)
#             myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)     
    
    
def main():
    train = pd.read_csv('./adult.data.txt', header=None)
    test = pd.read_csv('./adult.test.txt', header=None)

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', \
                'capital-gain', 'capital-loss', 'hours-per', 'native-country', 'income']
    train_data = preprocessing(train, column_names)
    test_data = preprocessing(test, column_names)
    initset_train = createInitSet(np.array(train_data).tolist()) # convert DataFrame to list
    len_train = len(initset_train)
    min_supportNo_train = len_train*.23
    initset_test = createInitSet(np.array(test_data).tolist()) # convert DataFrame to list
    len_test = len(initset_test)
    min_supportNo_test = len_test*.23
    
    t=time.time()
    FPtree_train, headerTab_train = createTree(initset_train,min_supportNo_train)
    freqItems_train = [ ] 
    mineTree(FPtree_train, headerTab_train, min_supportNo_train, set([]), freqItems_train)
    print('Executation time: ', time.time()-t)
    FPtree_test, headerTab_test = createTree(initset_test,min_supportNo_test)
    freqItems_test = [ ] 
    mineTree(FPtree_test, headerTab_test, min_supportNo_test, set([]), freqItems_test)
    t=time.time()    
    print('Executation time: ', time.time()-t)

    with open('FP-growth_Freqitems_train.txt', 'w') as f:
        f.write('Frequent Itemsets: Support')
        f.write('\n')
        for i in freqItems_train:
            f.write(str(i))
            f.write('\n')
    print(len(freqItems_train))
    print(len(freqItems_test))
    with open('FP-growth_Freqitems_test.txt', 'w') as f:
        f.write('Frequent Itemsets: Support')
        f.write('\n')
        for i in freqItems_test:
            f.write(str(i))
            f.write('\n')
                
if __name__ == '__main__':
    main()