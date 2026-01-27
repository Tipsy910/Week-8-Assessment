from pycaret.datasets import get_data
from pycaret.classification import ClassificationExperiment

data = get_data('diabetes')

exp = ClassificationExperiment()
exp.setup(data, target='Class variable', session_id=123)
base_model = exp.compare_models()
print(base_model)