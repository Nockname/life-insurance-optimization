import model
import torch
import matplotlib.pyplot as plt

premiums_previous_best = torch.zeros((20,))
premiums_previous_best[0] = 299193.84


# our premium schedules, obtained through train.py
premiums_linear_preprocessing = torch.tensor([  93725.19,   84857.21,   27488.98,    7035.36,      0.00,      0.00,      0.00,      0.00,      0.00,      0.00,       0.00,      0.00,      0.00,   13313.55,   54829.45,  101742.62,  141815.95,  176011.53,  171574.17,  203483.45])
premiums_exponential_preprocessing = torch.tensor([ 206970.72, 99.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24, 137528.3, 152636.11, 168792.05, 185921.27, 204047.02])


epvs = [
    model.calculate_expected_present_value(premiums_previous_best, change_death_benefit=True).item(),
    model.calculate_expected_present_value(premiums_linear_preprocessing, change_death_benefit=True).item(),
    model.calculate_expected_present_value(premiums_exponential_preprocessing, change_death_benefit=True).item()
]
print(epvs)

labels = ['Previous Best Strategy', 'Our Model, with Linear Preprocessing', 'Our Model, with Exponential Preprocessing']
plt.figure(figsize=(10, 6))

plt.plot(labels, epvs, color='tab:blue', marker='o')
plt.ylabel('Expected Present Value')
plt.xlabel('Premium Strategy')
plt.title('Comparison of Policy\'s Expected Present Value Between Previous Best and Our Models')
plt.tight_layout()
plt.show()
