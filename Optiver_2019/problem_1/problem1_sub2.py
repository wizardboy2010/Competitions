################################################################################
################################################################################
## Template file for problem 1. You have to fill in the function findNumbers  ##
## defined below. The function takes in an input number and return the list   ##
## of numbers that satisfy the problem statement. Please ensure you return a  ##
## list as the submission will be auto evaluated. We have provided a little   ##
## helper to ensure that the return value is correct.                         ##
##                                                                            ##
## You can run this template file to see the output of your function.         ##
## First replace the TEST_NUMBER with correct number.                         ##
## Then simply run: `python problem1_template.py`                             ##
## You should see the output printed once your program runs.                  ##
##                                                                            ##
## DO NOT CHANGE THE NAME OF THE FUNCTION BELOW. ONLY FILL IN THE LOGIC.      ##
## DONT FORGET TO RETURN THE VALUES AS A LIST                                 ##
## IF YOU MAKE ANY IMPORTS PUT THEM IN THE BODY OF THE FUNCTION               ##
##                                                                            ##
## You are free to write additional helper functions but ensure they are all  ##
## in this file else your submission wont work.                               ##
##                                                                            ##
## Good Luck!                                                                 ##
################################################################################
################################################################################


TEST_NUMBER = 1000

def create_prod_pair_dict(inputNumber):
    d = dict()
    for n1 in range(1, inputNumber+1):
        for n2 in range(n1, inputNumber+1):
            if n1*n2 in d:
                d[n1*n2].append([n1,n2])
            else:
                d[n1*n2] = [[n1,n2]]
    return d

def findNumbers(inputNumber):
    ##################################
    ##          FILL ME IN          ##
    ##################################
    
    d_prod_pair = create_prod_pair_dict(inputNumber)
                
    prod_flag = np.zeros((inputNumber, inputNumber))

    for n1 in range(1, inputNumber+1):
        for n2 in range(n1,inputNumber+1):
            if len(d_prod_pair[n1*n2])>1 :
                prod_flag[n1-1,n2-1] = 1
                
    s = np.ones(2*inputNumber)
    s[0] = 0
    
    for sum_pair in range(2,2*inputNumber+1):
        a = max(1,sum_pair - inputNumber)
        b = int(sum_pair/2)
        if a == b:
            n1 = a
            if prod_flag[n1-1,sum_pair - n1-1] == 0 :
                s[sum_pair-1] = 0
        else:
            for n1 in range(a,b):
                if prod_flag[n1-1,sum_pair - n1-1] == 0 :
                    s[sum_pair-1] = 0
                
    for n1 in range(1, inputNumber+1):
        for n2 in range(n1,inputNumber+1):
            if prod_flag[n1-1,n2-1] == 1 :
                l = len([1 for i in d_prod_pair[n1*n2] if s[i[0]+i[1]-1] == 1])
                if l != 1:
                    prod_flag[n1-1, n2-1] = 0
                
                
    while(True):
        temp1 = np.sum(s)
        temp2 = np.sum(prod_flag)

        for sum_pair in range(2,2*inputNumber+1):
            a = max(1,sum_pair - inputNumber)
            b = int(sum_pair/2)
            l = len([1 for n1 in range(a,b) if prod_flag[n1-1,sum_pair - n1 -1] == 1])
            if l != 1:
                s[sum_pair-1] = 0         

        if temp1 == np.sum(s) and temp2 == np.sum(prod_flag):
            break
        else:
            temp1 = np.sum(s)
            temp2 = np.sum(prod_flag)

        for n1 in range(1, inputNumber+1):
            for n2 in range(n1,inputNumber+1):
                if prod_flag[n1-1,n2-1] == 1 :
                    l = len([1 for i in d_prod_pair[n1*n2] if s[i[0]+i[1]-1] == 1])
                    if l != 1:
                        prod_flag[n1-1, n2-1] = 0

        if temp1 == np.sum(s) and temp2 == np.sum(prod_flag):
            break


    cand = []
    for n1 in range(1,inputNumber+1):
        for n2 in range(1,inputNumber+1):
            if prod_flag[n1-1, n2-1] == 1:
                temp = [i for i in d_prod_pair[n1*n2] if s[i[0]+i[1]-1]==1]
                for i in temp:
                    if i not in cand:
                        cand.append(i)
                    
    diff = {}
    for i in cand:
        if i[1]-i[0] in diff:
            diff[i[1]-i[0]].append(i)
        else:
            diff[i[1]-i[0]] = [i]
            
    diff = {i:diff[i] for i in diff if len(diff[i])>1}
    
    max_prob = 0
    #print(cand)
    for key, values in diff.items():
        temp = [j for k in values for j in k]
        prob = [temp.count(i)/len(temp) for i in temp]
        if max_prob < max(prob):
            max_prob = max(prob)
            i = prob.index(max(prob))
            num_guess = temp[i]
            final_list = values
    pair_num  = [i for i in final_list if num_guess not in i][0]    
    
    return pair_num

def ensureNumbers(returnList):
    for num in returnList:
        if type(num) is int:
            continue
        else:
            print(num, ' is not an integer.')
            raise TypeError('The return value is not a list of integers')
    return returnList

def ensureListOfNumbers(returnList):
    if type(returnList) is list:
        return ensureNumbers(returnList)
    else:
        print('Return value is not a list. Please ensure you return a list.')
        raise TypeError('The return value is not a list')



if __name__ == "__main__":
    import numpy as np
    print(ensureListOfNumbers(findNumbers(TEST_NUMBER)))