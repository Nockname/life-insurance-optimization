# --- Show Gains Over Previous Best Premium Schedules ---

"""
Previous Best Constant Strategy: 211010.95
Previous Best Early Strategy: 238397.97
Our Model, with Linear Preprocessing: 240270.94
Our Model, with Exponential Preprocessing: 245053.67

--------------------------------------------------
Flat Gains:
Gain over Constant: 29259.98 (Linear)
Gain over Constant: 34042.72 (Exponential)
Gain over Early: 1872.97 (Linear)
Gain over Early: 6655.70 (Exponential)

--------------------------------------------------
Percentage Gains:
Gain over Constant: 13.87% (Linear)
Gain over Constant: 16.13% (Exponential)
Gain over Early: 0.79% (Linear)
Gain over Early: 2.79% (Exponential)
"""


import model
import torch
import matplotlib.pyplot as plt
import previous_best

premiums_previous_best_early = torch.zeros((20,))
premiums_previous_best_early[0] = 299193.84


# our premium schedules, obtained through train.py
premiums_linear_preprocessing = torch.tensor([  93725.19,   84857.21,   27488.98,    7035.36,      0.00,      0.00,      0.00,      0.00,      0.00,      0.00,       0.00,      0.00,      0.00,   13313.55,   54829.45,  101742.62,  141815.95,  176011.53,  171574.17,  203483.45])
premiums_exponential_preprocessing = torch.tensor([ 206970.72, 99.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24, 137528.3, 152636.11, 168792.05, 185921.27, 204047.02])


epvs = [
    model.calculate_expected_present_value(previous_best.best_const_premiums, change_death_benefit=True).item(),
    model.calculate_expected_present_value(previous_best.best_early_premiums, change_death_benefit=True).item(),
    model.calculate_expected_present_value(premiums_linear_preprocessing, change_death_benefit=True).item(),
    model.calculate_expected_present_value(premiums_exponential_preprocessing, change_death_benefit=True).item()
]

print("\n" + "-" * 50)
print("Premium Schedules:")
print(f"Previous Best Constant Strategy: {[round(x, 2) for x in previous_best.best_const_premiums.tolist()]}")
print(f"Previous Best Early Strategy: {[round(x, 2) for x in previous_best.best_early_premiums.tolist()]}")
print(f"Our Model, with Linear Preprocessing: {[round(x, 2) for x in premiums_linear_preprocessing.tolist()]}")
print(f"Our Model, with Exponential Preprocessing: {[round(x, 2) for x in premiums_exponential_preprocessing.tolist()]}")

print("\n" + "-" * 50)
print("Expected Present Values:")
print(f"Previous Best Constant Strategy: {epvs[0]:.2f}")
print(f"Previous Best Early Strategy: {epvs[1]:.2f}")
print(f"Our Model, with Linear Preprocessing: {epvs[2]:.2f}")
print(f"Our Model, with Exponential Preprocessing: {epvs[3]:.2f}")

print("\n" + "-" * 50)
print("Flat Gains:")
print("Gain over Constant: {:.2f} (Linear)".format(epvs[2] - epvs[0]))
print("Gain over Constant: {:.2f} (Exponential)".format(epvs[3] - epvs[0]))
print("Gain over Early: {:.2f} (Linear)".format(epvs[2] - epvs[1]))
print("Gain over Early: {:.2f} (Exponential)".format(epvs[3] - epvs[1]))

print("\n" + "-" * 50)
print("Percentage Gains:")
print("Gain over Constant: {:.2f}% (Linear)".format((epvs[2] - epvs[0]) / epvs[0] * 100))
print("Gain over Constant: {:.2f}% (Exponential)".format((epvs[3] - epvs[0]) / epvs[0] * 100))
print("Gain over Early: {:.2f}% (Linear)".format((epvs[2] - epvs[1]) / epvs[1] * 100))
print("Gain over Early: {:.2f}% (Exponential)".format((epvs[3] - epvs[1]) / epvs[1] * 100))

labels = ['Previous Best Constant Strategy', 'Previous Best Early Strategy', 'Our Model, with Linear Preprocessing', 'Our Model, with Exponential Preprocessing']
plt.figure(figsize=(10, 6))

plt.plot(labels, epvs, color='tab:blue', marker='o')
plt.ylabel('Expected Present Value')
plt.xlabel('Premium Strategy')
plt.title('Comparison of Policy\'s Expected Present Value Between Previous Best and Our Models')
plt.tight_layout()
plt.show()