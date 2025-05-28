import model
import torch
import matplotlib.pyplot as plt

premiums_previous_best = torch.zeros((20,))
premiums_previous_best[0] = 299193.84


# our premium schedules, obtained through train.py
premiums_linear_preprocessing = torch.tensor([  93725.19,   84857.21,   27488.98,    7035.36,      0.00,      0.00,      0.00,      0.00,      0.00,      0.00,       0.00,      0.00,      0.00,   13313.55,   54829.45,  101742.62,  141815.95,  176011.53,  171574.17,  203483.45])
premiums_exponential_preprocessing = torch.tensor([ 206919.20,     203.61,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.00,       0.60,  137546.92,  152610.50,  168722.52,  186008.00,  203406.31])


epvs = [
    model.calculate_expected_present_value(premiums_previous_best, change_death_benefit=True).item(),
    model.calculate_expected_present_value(premiums_linear_preprocessing, change_death_benefit=True).item(),
    model.calculate_expected_present_value(premiums_exponential_preprocessing, change_death_benefit=True).item()
]

labels = ['Previous Best Strategy', 'Our Model, with Linear Preprocessing', 'Our Model, with Exponential Preprocessing']
plt.figure(figsize=(10, 6))

plt.plot(labels, epvs, color='tab:blue', marker='o')
plt.ylabel('Expected Present Value')
plt.xlabel('Premium Strategy')
plt.title('Comparison of Policy\'s Expected Present Value Between Previous Best and Our Models')
plt.tight_layout()
plt.show()
