import pandas as pd
import numpy as np
import math
import sys

arr=[]

class Node(object):
    def __init__(self, attribute):
        self.attr = attribute
        self.left = None
        self.right = None
        self.leaf = False
        self.value = None
        self.pos= None
        self.neg= None
        
def Entropy(df,target_col):
   p_count = df[df[target_col] == 1]
   n_count = df[df[target_col] == 0]
   p = float(p_count.shape[0])
   n = float(n_count.shape[0])
   if p==0 or n==0:
       return 0
   else:
       E = ((-1*p)/(p + n))*math.log(p/(p+n), 2) + ((-1*n)/(p + n))*math.log(n/(p+n), 2)
       return E 

def var(df,target_col):
    k0 =len(df[df[target_col] == 1])
    k1 =len(df[df[target_col] == 0])
    k=len(df[df.columns[-1]])
    if k0==0 or k1==0: 
        return 0
    else:
        V=(k0*k1)/np.square(k)
        return V
    
def Info_gain_entropy(df,split_attribute_name,target_col):
    target_entropy=Entropy(df,target_col)
    vals,counts= np.unique(df[split_attribute_name],return_counts=True)
    attribute_entropy=0
    for i in range(len(vals)):
        attribute_entropy+=(counts[i]/np.sum(counts))*Entropy(df[df[split_attribute_name]==vals[i]],target_col)
    ig=target_entropy-attribute_entropy
    return ig

    
    
def Info_gain_var(df,split_attribute_name,target_col):
    target_entropy=var(df,target_col)
    vals,counts= np.unique(df[split_attribute_name],return_counts=True)
    attribute_entropy=0
    for i in range(len(vals)):
        attribute_entropy+=(counts[i]/np.sum(counts))*var(df[df[split_attribute_name]==vals[i]],target_col)
    ig=target_entropy-attribute_entropy
    return ig


def split_tree(df, target_col ):
	p_df = df[df[target_col] == 1]
	n_df = df[df[target_col] == 0]
	return p_df.shape[0], n_df.shape[0]

def best_attr(df,attributes,target_col,impurity_type):
   max_info_gain = float("-inf")
   best_attr = None
	
   for attr in attributes:
       if impurity_type==1:
           ig = Info_gain_entropy(df, attr,target_col )
       else:
           ig = Info_gain_var(df, attr,target_col )
       if ig > max_info_gain:
           max_info_gain = ig
           best_attr = attr
           			
   return best_attr

def dtree(df,attributes,target_col,impurity_type):
    global arr
   
    p, n = split_tree(df, target_col)
    if p == 0 or n == 0:
        leaf = Node(None)
        leaf.pos=p
        leaf.neg=n
        arr.append(leaf)
        leaf.leaf = True
        if p > n:
            leaf.value = 1
        else:
            leaf.value = 0
        return leaf
    else:
        bestattr= best_attr(df, attributes, target_col,impurity_type)    
        root = Node(bestattr)
        root.pos=p
        root.neg=n
        arr.append(root)
        sub_1 = df[df[bestattr] == 0]
        sub_2 = df[df[bestattr] == 1]
        attributes=attributes.drop([bestattr])
        root.left = dtree(sub_1, attributes, target_col,impurity_type)
        
        root.right = dtree(sub_2, attributes, target_col,impurity_type)
        
        return root

def predict(root, row_df):

	if root.leaf:
		return root.value
	if row_df[root.attr]==0:
		return predict(root.left, row_df)
	elif row_df[root.attr]==1:
		return predict(root.right, row_df)


def test_predictions(root, df):
    total_data = df.shape[0]
    correct_data = 0
    for index,row in df.iterrows():
        prediction = predict(root, row)
        if prediction == row[df.columns[-1]]:
            correct_data += 1
    return (correct_data/total_data)

def prune(root,df_valid,init_acc):
    global arr
    
    best_score=0
    present_score=0
    best_pos=-1
   

    for i in range(len(arr)):
        if arr[i].leaf==True:
            continue
        else:
            arr[i].leaf=True
            
            if arr[i].pos > arr[i].neg:
                arr[i].value = 1
            else:
                arr[i].value = 0
            present_score=test_predictions(root,df_valid)*100.0
            if present_score>best_score:
                best_score=present_score
                best_pos=i
            arr[i].leaf=False

    best_node=arr[best_pos]
    if best_score<init_acc:
        return     
    else:
        best_node.leaf=True
        if best_node.pos > best_node.neg:
            best_node.value = 1
        else:
            best_node.value = 0
        prune(root,df_valid,best_score)

def travel(root,level,d):
    if root==None:
        return
    
    if level==d:
        root.leaf=True
        if root.pos > root.neg:
            root.value = 1
        else:
            root.value = 0
    elif level<d and root.left!=None:
        root.leaf=False
    
    travel(root.left,level+1,d)
    travel(root.right,level+1,d)


def main():
    
    impurity_type=int(sys.argv[1])
    pruning_type=int(sys.argv[2])
    train_dataset=sys.argv[3]
    valid_dataset=sys.argv[4]
    test_dataset=sys.argv[5]
    
    
    df=pd.read_csv(train_dataset,header=None)
    myList=[]
    for i in range(len(df.columns)):
        myList.append('p'+str(i))
    df=df.rename(columns=dict(zip(df.columns,myList)))
    target_col=df.columns[-1]
    ipattributes = df.columns
    ipattributes=ipattributes.drop([target_col])
    root = dtree(df, ipattributes,target_col,impurity_type)
    
   
    if pruning_type!=0:
        print("Accuracy before pruning")
        df_valid=pd.read_csv(valid_dataset,header=None)
        df_valid=df_valid.rename(columns=dict(zip(df_valid.columns,myList)))
        inti_acc=0
        inti_acc= test_predictions(root, df_valid)*100.0
        print(str(inti_acc) + '%')
        
        if pruning_type==1:
    
            print("accuracy after pruning")
            prune(root,df_valid,inti_acc)
            print(str(test_predictions(root, df_valid)*100.0) + '%')
        
        
        elif pruning_type==2:
        
            best_acc=inti_acc
            L=[5,10,15,20,50,100]
            for i in range(len(L)):
                level=0
                d=L[i]
                travel(root,level,d)
                new_acc=test_predictions(root,df_valid)*100.0
                if new_acc>best_acc:
                    best_acc=new_acc
            print('accuracy after pruning')
            print(str(best_acc)+'%')
        

    
    df_test=pd.read_csv(test_dataset,header=None)
    df_test=df_test.rename(columns=dict(zip(df_test.columns,myList)))
    print("Accuracy of test data")
    print(str(test_predictions(root, df_test)*100.0) + '%')
    


if __name__ == '__main__':
	main()

