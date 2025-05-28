import model
import torch
import matplotlib.pyplot as plt



# Iteration 1200000: Loss = -240277.2188, Positive Balance Years =  20, Premiums = [  93725.19,   84857.21,   27488.98,    7035.36,      -3.03,      -3.47,      -3.14,      -2.97,      -2.96,      -3.12,       0.00,      -3.52,      -3.66,   13313.55,   54829.45,  101742.62,  141815.95,  176011.53,  171574.17,  203483.45]


"""
Best (using linear approach): 240277.2188
Best (using exp approach): 245056.0781, with Positive Balance Years: 20
Single: 238403.1719
245056.07818 / 238403.1719 = 1.0282
"""

premiums_best_from_linear = torch.tensor([  93725.19,   84857.21,   27488.98,    7035.36,      0.00,      0.00,      0.00,      0.00,      0.00,      0.00,       0.00,      0.00,      0.00,   13313.55,   54829.45,  101742.62,  141815.95,  176011.53,  171574.17,  203483.45])

premiums_best_from_exp = torch.tensor([ 206919.20,     203.61,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.60,  137546.92,  152610.50,  168722.52,  186008.00,  203406.31])

206919.20
203.61
0.00
0.00
0.00
0.00
0.00
0.00
0.00
0.00
0.00
0.00
0.00
0.00
0.60
137546.92
152610.50
168722.52
186008.00
203866.31

# print(core_function.calculate_expected_present_value(premiums_bad, use_excel_rounding=False, change_death_benefit=True, print_output=True).item())
print("Best (using linear approach):", model.calculate_expected_present_value(premiums_best_from_exp, use_excel_rounding=True, change_death_benefit=True).item())
print(model.smooth_loss(premiums_best_from_exp, weight=0.5))
print(model.calculate_balance(premiums_best_from_exp))

