# ---- Constants for Insurance Plan, Mortality Statistics, and Inflation Rate / Discount Rate ----

import torch

YEARS = 20

# discount rate
DISCOUNT = torch.full((YEARS,), 0.05)

# policy values
INITIAL_BALANCE = 209000.0 
DEATH_BENEFIT = 1000000.0
COI = torch.tensor([25.3835, 28.7676, 32.2841, 36.1080, 40.0561, 44.9284, 49.7497, 56.7585, 65.8140, 75.9054, 86.8502, 98.5907, 110.7127, 123.0529, 134.7429, 155.6109, 175.6953, 198.0022, 222.6517, 249.7688])
INTEREST_RATE = torch.full((YEARS,), 0.0564)

# mortality statistics
BASE_MORTALITY = torch.tensor([0.004056, 0.006672, 0.011776, 0.017520, 0.022664, 0.028984, 0.037968, 0.049360, 0.064032, 0.083352, 0.108296, 0.127832, 0.140240, 0.152200, 0.163152, 0.174760, 0.188432, 0.203136, 0.218912, 0.235448])
ANNUAL_IMPROVEMENT_FACTOR = torch.full((YEARS,), 0.01)
SHOCK_FACTOR = torch.zeros((YEARS,))