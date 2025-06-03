# Gradient Descent for Life Insurance Premiums

import numpy as np
import constants
import torch
import matplotlib.pyplot as plt
from constants import *


# ---- Input ----
premiums = torch.full((YEARS,), 20000.0)
premiums.requires_grad_()


# ---- Compute Policy Premiums ----

# Find ending balance and effect on death rate, given premiums schedule.
def calculate_balance(premiums):
    balance_boy = torch.zeros((YEARS + 1,))
    balance_boy[0] = INITIAL_BALANCE
    PERCENT_PREMIUM_CHARGE = torch.full((YEARS,), 0.02)
    POLICY_FEE = torch.full((YEARS,), 150.0)
    for i in range(YEARS):
        premium_charge = -premiums[i] * PERCENT_PREMIUM_CHARGE[i]
        net_at_risk = DEATH_BENEFIT - premiums[i] - balance_boy[i] - premium_charge + POLICY_FEE[i]
        cost_of_insurance = - COI[i] * net_at_risk / 1000
        partial_sum = balance_boy[i] + premiums[i] + premium_charge - POLICY_FEE[i] + cost_of_insurance
        balance_boy[i+1] = (partial_sum) * (1 + INTEREST_RATE[i])
        
    return balance_boy
    


# ---- Compute Expected Present Value ----


# get boy and eoy discount factors
def discount_factors():
    
    discount_boy = torch.empty((YEARS+1,))
    discount_boy[0] = 1
    for i in range(1, YEARS + 1):
        discount_boy[i] = discount_boy[i-1] / (1 + DISCOUNT[i-1])
        
    return discount_boy, discount_boy[1:]


# get probability of boy survival and eoy death 
def actuarial_factors():
    cumulative_improvement_factor = 1 - ANNUAL_IMPROVEMENT_FACTOR
    adj_rate_one = BASE_MORTALITY * cumulative_improvement_factor
    adj_rate_two = adj_rate_one * (1 + SHOCK_FACTOR)

    survivors_boy = torch.zeros((YEARS + 1,))
    deaths_eoy = torch.zeros((YEARS,))
    survivors_boy[0] = 1.0  # Start with 100% at BOY

    # iteratively find death and survival rates based on previous year results
    for i in range(YEARS):
        deaths_eoy[i] = survivors_boy[i] * adj_rate_two[i]
        survivors_boy[i + 1] = survivors_boy[i] - deaths_eoy[i]

    # we assume all remaining living people die on the last year / collect on the policy
    deaths_eoy[YEARS - 1] = survivors_boy[YEARS - 1]
    return survivors_boy[:-1], deaths_eoy

# inaccurate way that Excel calculates, with too low precision
def excel_actuarial_factors():
    MORTALITY = torch.tensor([1.0000, 0.9960, 0.9895, 0.9782, 0.9617, 0.9410, 0.9153, 0.8829, 0.8427, 0.7934, 0.7336, 0.6625, 0.5874, 0.5151, 0.4470, 0.3843, 0.3271, 0.2751, 0.2285, 0.1872, 0.0])
    survival_boy = MORTALITY[:-1]
    death_eoy = MORTALITY[:-1] - MORTALITY[1:]
    return survival_boy, death_eoy


def positive_balance_years(premiums, premiums_modification_function=torch.nn.ReLU()):

    balance_eoy = calculate_balance(premiums_modification_function(premiums))[1:]
    negative_balance_years = torch.where(balance_eoy < 0)[0]
    
    if len(negative_balance_years) > 0:
        return negative_balance_years[0].item()
    
    return YEARS

# find net expected present value of insurance policy
# use_excel_rounding rounds the actuarial factors to 4 sig figs after each calculation. set True to test against Excel spreadsheet, keep False for higher accuracy
def calculate_expected_present_value(premiums, use_excel_rounding = False, change_death_benefit = False, print_output = False):
    
    death_benefit = torch.full((YEARS,), DEATH_BENEFIT)
        
    # set death benefit to 0 if the policy has ever been in a negative balance
    if change_death_benefit:
        death_benefit[positive_balance_years(premiums):] = 0
    
    discount_boy, discount_eoy = discount_factors()
    survival_boy, death_eoy = excel_actuarial_factors() if use_excel_rounding else actuarial_factors()
    
    premium_cash_flow_boy = premiums * survival_boy
    death_benefit_cash_flow_eoy = death_benefit * death_eoy
    
    if print_output:
        print("Premium Cash Flow (BOY):")
        for idx, val in enumerate(premium_cash_flow_boy):
            print(f"  Year {idx+1:2d}: {val:12,.2f}")
        print("Death Benefit Cash Flow (EOY):")
        for idx, val in enumerate(death_benefit_cash_flow_eoy):
            print(f"  Year {idx+1:2d}: {val:12,.2f}")
        
    # compute net cash flow via offsetting death_benefit_cash_flow_eoy and premium_cash_flow_boy
    net_cash_flow_boy = (
        torch.cat((torch.tensor([0.0]), death_benefit_cash_flow_eoy)) - 
        torch.cat((premium_cash_flow_boy, torch.tensor([0.0]))))
    
    return torch.sum(discount_boy * net_cash_flow_boy)


# ---- Loss Functions ----

# loss is negative balance at the end of the policy or expected present value, depending on whether the policy is in negative balance at the end
def piecewise_loss(premiums):

    balance_boy = calculate_balance(premiums)
    expected_present_value = calculate_expected_present_value(premiums) 
    
    if balance_boy[-1] < -1:
        return -balance_boy[-1]
    
    return -expected_present_value

# loss is expected present value (where policy expires if ever in negative balance) + negative balance at the end of the policy
def expiration_loss(premiums):
    
    balance_boy = calculate_balance(premiums)
    expected_present_value = calculate_expected_present_value(premiums, change_death_benefit=True) 
    return -(expected_present_value + torch.min(balance_boy[-1], torch.tensor(0.0)))

# loss is expected present value of policy (through completion of the policy) + (weighted) negative balance at the end of the policy
def smooth_loss(premiums, weight, balance_activation_function=torch.nn.ReLU(), premiums_modification_function=torch.nn.ReLU()):

    balance_boy = calculate_balance(premiums_modification_function(premiums))
    expected_present_value = calculate_expected_present_value(premiums_modification_function(premiums))

    return -expected_present_value + balance_activation_function(-torch.min(balance_boy)) * weight

