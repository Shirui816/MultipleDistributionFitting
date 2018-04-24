# MultipleDistributionFitting
Finding optimized number of components from mixed distribution data.

Process:

1. Define target function(s)
2. Create fitting model(s)
3. Evaluation the model by AIC, AICc, BIC
4. Choose the model that minimizes the BIC, AICc or AIC

See the [document](http://multipledistributionfitting.readthedocs.io/en/latest/index.html) here.

## Installation

### pip installation method is on the way

### Directly download

```bash
wget https://github.com/Shirui816/MultipleDistributionFitting/archive/master.zip
```

or

```bash
git clone https://github.com/Shirui816/MultipleDistributionFitting.git
```

### Requirements

```python
numpy >= 1.14
scipy >= 1.0
python3
pandas >= 0.21
```

Install requirements by

```bash
pip install -r requirements.txt
```

### Windows executables is preparing

For Unix/Linux users, you can use `python bin/LorentzianNMR.py <files>`, or
simply `LorentzianNMR.py <files>` if `LorentzianNMR.py` is in your `PATH`.
