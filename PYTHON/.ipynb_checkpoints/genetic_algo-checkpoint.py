from bga import BGA
import numpy as np






def values(arr):
    num_1 = int("".join(str(i) for i in arr[0:5]),2)
    num_2 = int("".join(str(i) for i in arr[5:10]),2)
    num_3 = int("".join(str(i) for i in arr[10:15]),2)
    num_4 = int("".join(str(i) for i in arr[15:20]),2)
    
    sol_1 = num_1 + num_2
    sol_2 = num_4 - num_3
    sol_3 = num_2+num_3
    sol_4 = num_1+num_4
    
    final_sol = (((8 - sol_1)**2.0)+((6-sol_2)**2)+((8-sol_3)**2)+((13-sol_4)**2))**0.5
    print(final_sol)
    
    return final_sol
        

        
num_pop = 1000

problem_dimentions = 20

test = BGA(pop_shape=(num_pop, problem_dimentions), method=values, p_c=0.8, p_m=0.8, max_round = 1000000, early_stop_rounds=None, verbose = None, maximum=False)
best_solution, best_fitness = test.run()