
# Naive Bayes - Learning a Model From Data, Then Checking Events Against Our Model

In my [previous Naïve Bayes Notebook](https://github.com/cerebraljam/naive_bayes_notebook), I built a guestimated probability distribution for a imaginary web application. Using that probability distribution, we could update our prior belief to get a good idea if a sequence of events could be associated to a 'buyer' user or not.

How can we come up with the probability distribution from observed data? This notebook demonstrates how we could learn a model from some user session data, and then query that model against a series of events.

For this notebook, I am still using a Naive Bayes approach, which is known to be less efficient than a Bayes Network, but still, it provides quick and dirty results without much efforts.

For more information on how to build a Bayes Network from data, [pgympy](https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb) does have a nice example, given that the complexity of the network isn't too complicated.


```python
import pandas as pd
import numpy as np
import copy
```


```python
%load_ext autoreload
%autoreload 2

# For clarity, I moved the function used to loop through the events and update the probabilities in an external file.
from naive_bayes import analyse_events
```

## Loading The User Session Data

The training file is a text file with "user,action' events. 


```python
data = pd.read_csv('training_events.csv',delimiter=',')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>buyer</td>
      <td>login</td>
    </tr>
    <tr>
      <th>1</th>
      <td>buyer</td>
      <td>view</td>
    </tr>
    <tr>
      <th>2</th>
      <td>buyer</td>
      <td>view</td>
    </tr>
    <tr>
      <th>3</th>
      <td>buyer</td>
      <td>view</td>
    </tr>
    <tr>
      <th>4</th>
      <td>buyer</td>
      <td>view</td>
    </tr>
  </tbody>
</table>
</div>



For the purpose of this notebook, the training events does at have at least one action of each per user. The list of observed actions per user looks like this:


```python
data.groupby(['user'])['action'].unique()
```




    user
    buyer        [login, view, buy, search, address change, log...
    fraudster    [login fail, login, search, view, sell, buy, a...
    seller       [login, address change, search, view, sell, lo...
    Name: action, dtype: object



## Learning the probability distribution from the training data

The idea here is to build a memory representation of **User | Actions | True/False | Positive/Negative** probabilities.

* True Positives are calculated by calculating the ratio between action for a user and that action being used by all users. P(action|user) / P(Action)
* False Positives are calculated similarily: P(action | -user) / P(action)
* True Negatives and False Negative are simply the value of *1 - True Positive* and *1 - False Positive*


```python
template = {'True': { 'Positive': 0.5, 'Negative': 0.5}, 'False': {'Positive': 0.5, 'Negative': 0.5}}
action_template = {}
distribution = {}
for action in data['action'].unique():
    action_template[action] = copy.deepcopy(template)

for user in data['user'].unique():
    distribution[user] = copy.deepcopy(action_template)
    for action in data['action'].unique():
        TP = data.loc[(data['action'] == action) & (data['user'] == user)]['action'].value_counts() / data.loc[data['action'] == action]['action'].value_counts()
        FP = data.loc[(data['action'] == action) & (data['user'] != user)]['action'].value_counts() / data.loc[data['action'] == action]['action'].value_counts()
        TN = 1 - TP
        FN = 1 - FP

        # print("{}\t{}\t{:.2f} {:.2f} {:.2f} {:.2f}".format(user, action, TP[action],FP[action],TN[action],FN[action]))
        distribution[user][action]['True']['Positive'] = TP[action]
        distribution[user][action]['False']['Positive'] = FP[action]
        distribution[user][action]['True']['Negative'] = TN[action]
        distribution[user][action]['False']['Negative'] = FN[action]

```

The end results looks like the following:

* For Buyer, Seller and Fraudster:
    * what is the probability of having a True Positive, what is the probability of a False Positive


```python
print(distribution['buyer']['buy']['True']['Positive'], distribution['buyer']['buy']['False']['Positive'])
print(distribution['seller']['buy']['True']['Positive'], distribution['seller']['buy']['False']['Positive'])
print(distribution['fraudster']['buy']['True']['Positive'], distribution['fraudster']['buy']['False']['Positive'])
```

    0.625 0.375
    0.0625 0.9375
    0.3125 0.6875


Our memory representation is a 4 Dimension memory representation that would look like the following in a Pandas DataFrame


```python
df = pd.DataFrame(distribution)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buyer</th>
      <th>fraudster</th>
      <th>seller</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>address change</th>
      <td>{'False': {'Positive': 0.8, 'Negative': 0.1999...</td>
      <td>{'False': {'Positive': 0.4, 'Negative': 0.6}, ...</td>
      <td>{'False': {'Positive': 0.8, 'Negative': 0.1999...</td>
    </tr>
    <tr>
      <th>buy</th>
      <td>{'False': {'Positive': 0.375, 'Negative': 0.62...</td>
      <td>{'False': {'Positive': 0.6875, 'Negative': 0.3...</td>
      <td>{'False': {'Positive': 0.9375, 'Negative': 0.0...</td>
    </tr>
    <tr>
      <th>login</th>
      <td>{'False': {'Positive': 0.6, 'Negative': 0.4}, ...</td>
      <td>{'False': {'Positive': 0.6, 'Negative': 0.4}, ...</td>
      <td>{'False': {'Positive': 0.8, 'Negative': 0.1999...</td>
    </tr>
    <tr>
      <th>login fail</th>
      <td>{'False': {'Positive': 0.9545454545454546, 'Ne...</td>
      <td>{'False': {'Positive': 0.09090909090909091, 'N...</td>
      <td>{'False': {'Positive': 0.9545454545454546, 'Ne...</td>
    </tr>
    <tr>
      <th>logout</th>
      <td>{'False': {'Positive': 0.8, 'Negative': 0.1999...</td>
      <td>{'False': {'Positive': 0.4, 'Negative': 0.6}, ...</td>
      <td>{'False': {'Positive': 0.8, 'Negative': 0.1999...</td>
    </tr>
    <tr>
      <th>search</th>
      <td>{'False': {'Positive': 0.5, 'Negative': 0.5}, ...</td>
      <td>{'False': {'Positive': 0.8333333333333334, 'Ne...</td>
      <td>{'False': {'Positive': 0.6666666666666666, 'Ne...</td>
    </tr>
    <tr>
      <th>sell</th>
      <td>{'False': {'Positive': 0.9444444444444444, 'Ne...</td>
      <td>{'False': {'Positive': 0.7222222222222222, 'Ne...</td>
      <td>{'False': {'Positive': 0.3333333333333333, 'Ne...</td>
    </tr>
    <tr>
      <th>view</th>
      <td>{'False': {'Positive': 0.23728813559322035, 'N...</td>
      <td>{'False': {'Positive': 0.9322033898305084, 'Ne...</td>
      <td>{'False': {'Positive': 0.8305084745762712, 'Ne...</td>
    </tr>
  </tbody>
</table>
</div>



## Querying our models with a sequence of events

We start with a prior probability of 1 / 3 (Number of user profiles in our model). This could prior probability could be improved by identifying the distribution of our user profiles, but for now, this will do the work.


```python
prior = {}
users = sorted(data['user'].unique())
for u in users:
    prior[u] = 1 / len(data['user'].unique())
    print("{}: {:.2f}%".format(u, prior[u]))
```

    buyer: 0.33%
    fraudster: 0.33%
    seller: 0.33%


# Updating Our Bliefs, Given Evidences

Let's say that we are observing a chain of events. For each of these events, we want to update our belief about the user.

Note that we are doing two things:
* Updating our posterior probability given that we are NOT observing an event
* Updating our posterior probability every time that we observe an event happening.

The reason why we are updating the posterior probability every time we observe an event, even if it was observed earlier, is to amplify events. For example: *login fail* is likely to happen for each of our profile, but it is less likely for the buyer and seller. Having multiple *login fail* should increase our belief that we are dealing with a fraudster.

### The Casual Buyer's Profile


```python
events = ['search', 'view', 'search', 'view', 'view', 'buy']

print("Given the evidences '{}'...".format(",".join(events)))
for user in users:
    print("what is the posterior probability that our user is a {}?".format(user))
    posterior = analyse_events(prior[user], events, distribution[user])
    print("Probability that our user is a {} is: {:.3f}%\n".format(user, 100 * posterior))
```

    Given the evidences 'search,view,search,view,view,buy'...
    what is the posterior probability that our user is a buyer?
    	* "login fail" is False (0.95). Updating 0.33 to 0.91
    	* "login" is False (0.60). Updating 0.91 to 0.94
    	* "address change" is False (0.80). Updating 0.94 to 0.98
    	* "sell" is False (0.94). Updating 0.98 to 1.00
    	* "logout" is False (0.80). Updating 1.00 to 1.00
    	* "search" is True (0.50). Updating 1.00 to 1.00
    	* "view" is True (0.76). Updating 1.00 to 1.00
    	* "search" is True (0.50). Updating 1.00 to 1.00
    	* "view" is True (0.76). Updating 1.00 to 1.00
    	* "view" is True (0.76). Updating 1.00 to 1.00
    	* "buy" is True (0.62). Updating 1.00 to 1.00
    Probability that our user is a buyer is: 100.000%
    
    what is the posterior probability that our user is a fraudster?
    	* "login fail" is False (0.09). Updating 0.33 to 0.05
    	* "login" is False (0.60). Updating 0.05 to 0.07
    	* "address change" is False (0.40). Updating 0.07 to 0.05
    	* "sell" is False (0.72). Updating 0.05 to 0.12
    	* "logout" is False (0.40). Updating 0.12 to 0.08
    	* "search" is True (0.17). Updating 0.08 to 0.02
    	* "view" is True (0.07). Updating 0.02 to 0.00
    	* "search" is True (0.17). Updating 0.00 to 0.00
    	* "view" is True (0.07). Updating 0.00 to 0.00
    	* "view" is True (0.07). Updating 0.00 to 0.00
    	* "buy" is True (0.31). Updating 0.00 to 0.00
    Probability that our user is a fraudster is: 0.000%
    
    what is the posterior probability that our user is a seller?
    	* "login fail" is False (0.95). Updating 0.33 to 0.91
    	* "login" is False (0.80). Updating 0.91 to 0.98
    	* "address change" is False (0.80). Updating 0.98 to 0.99
    	* "sell" is False (0.33). Updating 0.99 to 0.99
    	* "logout" is False (0.80). Updating 0.99 to 1.00
    	* "search" is True (0.33). Updating 1.00 to 0.99
    	* "view" is True (0.17). Updating 0.99 to 0.97
    	* "search" is True (0.33). Updating 0.97 to 0.94
    	* "view" is True (0.17). Updating 0.94 to 0.78
    	* "view" is True (0.17). Updating 0.78 to 0.42
    	* "buy" is True (0.06). Updating 0.42 to 0.05
    Probability that our user is a seller is: 4.544%
    


### The Seller's Profile


```python
events = ['sell', 'sell', 'view', 'sell', 'sell', 'search', 'sell']

print("Given the evidences '{}'...".format(",".join(events)))
for user in users:
    print("what is the posterior probability that our user is a {}?".format(user))
    posterior = analyse_events(prior[user], events, distribution[user])
    print("Probability that our user is a {} is: {:.3f}%\n".format(user, 100 * posterior))
```

    Given the evidences 'sell,sell,view,sell,sell,search,sell'...
    what is the posterior probability that our user is a buyer?
    	* "login fail" is False (0.95). Updating 0.33 to 0.91
    	* "login" is False (0.60). Updating 0.91 to 0.94
    	* "buy" is False (0.38). Updating 0.94 to 0.90
    	* "address change" is False (0.80). Updating 0.90 to 0.97
    	* "logout" is False (0.80). Updating 0.97 to 0.99
    	* "sell" is True (0.06). Updating 0.99 to 0.90
    	* "sell" is True (0.06). Updating 0.90 to 0.34
    	* "view" is True (0.76). Updating 0.34 to 0.63
    	* "sell" is True (0.06). Updating 0.63 to 0.09
    	* "sell" is True (0.06). Updating 0.09 to 0.01
    	* "search" is True (0.50). Updating 0.01 to 0.01
    	* "sell" is True (0.06). Updating 0.01 to 0.00
    Probability that our user is a buyer is: 0.034%
    
    what is the posterior probability that our user is a fraudster?
    	* "login fail" is False (0.09). Updating 0.33 to 0.05
    	* "login" is False (0.60). Updating 0.05 to 0.07
    	* "buy" is False (0.69). Updating 0.07 to 0.14
    	* "address change" is False (0.40). Updating 0.14 to 0.10
    	* "logout" is False (0.40). Updating 0.10 to 0.07
    	* "sell" is True (0.28). Updating 0.07 to 0.03
    	* "sell" is True (0.28). Updating 0.03 to 0.01
    	* "view" is True (0.07). Updating 0.01 to 0.00
    	* "sell" is True (0.28). Updating 0.00 to 0.00
    	* "sell" is True (0.28). Updating 0.00 to 0.00
    	* "search" is True (0.17). Updating 0.00 to 0.00
    	* "sell" is True (0.28). Updating 0.00 to 0.00
    Probability that our user is a fraudster is: 0.001%
    
    what is the posterior probability that our user is a seller?
    	* "login fail" is False (0.95). Updating 0.33 to 0.91
    	* "login" is False (0.80). Updating 0.91 to 0.98
    	* "buy" is False (0.94). Updating 0.98 to 1.00
    	* "address change" is False (0.80). Updating 1.00 to 1.00
    	* "logout" is False (0.80). Updating 1.00 to 1.00
    	* "sell" is True (0.67). Updating 1.00 to 1.00
    	* "sell" is True (0.67). Updating 1.00 to 1.00
    	* "view" is True (0.17). Updating 1.00 to 1.00
    	* "sell" is True (0.67). Updating 1.00 to 1.00
    	* "sell" is True (0.67). Updating 1.00 to 1.00
    	* "search" is True (0.33). Updating 1.00 to 1.00
    	* "sell" is True (0.67). Updating 1.00 to 1.00
    Probability that our user is a seller is: 99.997%
    


### The Heavy Buyer and Seller's Profile

For this one, the user is buying and selling. How does our posterior probability reacts?


```python
events = ['buy', 'sell', 'view', 'sell', 'buy', 'view']

print("Given the evidences '{}'...".format(",".join(events)))
for user in users:
    print("what is the posterior probability that our user is a {}?".format(user))
    posterior = analyse_events(prior[user], events, distribution[user])
    print("Probability that our user is a {} is: {:.3f}%\n".format(user, 100 * posterior))
```

    Given the evidences 'buy,sell,view,sell,buy,view'...
    what is the posterior probability that our user is a buyer?
    	* "search" is False (0.50). Updating 0.33 to 0.33
    	* "login fail" is False (0.95). Updating 0.33 to 0.91
    	* "login" is False (0.60). Updating 0.91 to 0.94
    	* "address change" is False (0.80). Updating 0.94 to 0.98
    	* "logout" is False (0.80). Updating 0.98 to 1.00
    	* "buy" is True (0.62). Updating 1.00 to 1.00
    	* "sell" is True (0.06). Updating 1.00 to 0.96
    	* "view" is True (0.76). Updating 0.96 to 0.99
    	* "sell" is True (0.06). Updating 0.99 to 0.82
    	* "buy" is True (0.62). Updating 0.82 to 0.89
    	* "view" is True (0.76). Updating 0.89 to 0.96
    Probability that our user is a buyer is: 96.157%
    
    what is the posterior probability that our user is a fraudster?
    	* "search" is False (0.83). Updating 0.33 to 0.71
    	* "login fail" is False (0.09). Updating 0.71 to 0.20
    	* "login" is False (0.60). Updating 0.20 to 0.27
    	* "address change" is False (0.40). Updating 0.27 to 0.20
    	* "logout" is False (0.40). Updating 0.20 to 0.14
    	* "buy" is True (0.31). Updating 0.14 to 0.07
    	* "sell" is True (0.28). Updating 0.07 to 0.03
    	* "view" is True (0.07). Updating 0.03 to 0.00
    	* "sell" is True (0.28). Updating 0.00 to 0.00
    	* "buy" is True (0.31). Updating 0.00 to 0.00
    	* "view" is True (0.07). Updating 0.00 to 0.00
    Probability that our user is a fraudster is: 0.003%
    
    what is the posterior probability that our user is a seller?
    	* "search" is False (0.67). Updating 0.33 to 0.50
    	* "login fail" is False (0.95). Updating 0.50 to 0.95
    	* "login" is False (0.80). Updating 0.95 to 0.99
    	* "address change" is False (0.80). Updating 0.99 to 1.00
    	* "logout" is False (0.80). Updating 1.00 to 1.00
    	* "buy" is True (0.06). Updating 1.00 to 0.99
    	* "sell" is True (0.67). Updating 0.99 to 0.99
    	* "view" is True (0.17). Updating 0.99 to 0.97
    	* "sell" is True (0.67). Updating 0.97 to 0.99
    	* "buy" is True (0.06). Updating 0.99 to 0.83
    	* "view" is True (0.17). Updating 0.83 to 0.50
    Probability that our user is a seller is: 49.878%
    


### The Fraudster's Profile

For this example, what skew the probability toward the fraudster, even if there is a lot of *buy*, is mainly the fact that *search* and *view* actions are not observed, which increase the probability that we are dealing with a *fraudster* over a normal user. The *failed logins* clearly doesn't help.


```python
events = ['login fail', 'login fail', 'login', 'address change', 'buy', 'buy', 'buy','logout']

print("Given the evidences '{}'...\n".format(",".join(events)))
for user in users:
    print("what is the posterior probability that our user is a {}?".format(user))
    posterior = analyse_events(prior[user], events, distribution[user])
    print("Probability that our user is a {} is: {:.3f}%\n".format(user, 100 * posterior))
```

    Given the evidences 'login fail,login fail,login,address change,buy,buy,buy,logout'...
    
    what is the posterior probability that our user is a buyer?
    	* "search" is False (0.50). Updating 0.33 to 0.33
    	* "sell" is False (0.94). Updating 0.33 to 0.89
    	* "view" is False (0.24). Updating 0.89 to 0.73
    	* "login fail" is True (0.05). Updating 0.73 to 0.11
    	* "login fail" is True (0.05). Updating 0.11 to 0.01
    	* "login" is True (0.40). Updating 0.01 to 0.00
    	* "address change" is True (0.20). Updating 0.00 to 0.00
    	* "buy" is True (0.62). Updating 0.00 to 0.00
    	* "buy" is True (0.62). Updating 0.00 to 0.00
    	* "buy" is True (0.62). Updating 0.00 to 0.00
    	* "logout" is True (0.20). Updating 0.00 to 0.00
    Probability that our user is a buyer is: 0.116%
    
    what is the posterior probability that our user is a fraudster?
    	* "search" is False (0.83). Updating 0.33 to 0.71
    	* "sell" is False (0.72). Updating 0.71 to 0.87
    	* "view" is False (0.93). Updating 0.87 to 0.99
    	* "login fail" is True (0.91). Updating 0.99 to 1.00
    	* "login fail" is True (0.91). Updating 1.00 to 1.00
    	* "login" is True (0.40). Updating 1.00 to 1.00
    	* "address change" is True (0.60). Updating 1.00 to 1.00
    	* "buy" is True (0.31). Updating 1.00 to 1.00
    	* "buy" is True (0.31). Updating 1.00 to 1.00
    	* "buy" is True (0.31). Updating 1.00 to 1.00
    	* "logout" is True (0.60). Updating 1.00 to 1.00
    Probability that our user is a fraudster is: 99.921%
    
    what is the posterior probability that our user is a seller?
    	* "search" is False (0.67). Updating 0.33 to 0.50
    	* "sell" is False (0.33). Updating 0.50 to 0.33
    	* "view" is False (0.83). Updating 0.33 to 0.71
    	* "login fail" is True (0.05). Updating 0.71 to 0.10
    	* "login fail" is True (0.05). Updating 0.10 to 0.01
    	* "login" is True (0.20). Updating 0.01 to 0.00
    	* "address change" is True (0.20). Updating 0.00 to 0.00
    	* "buy" is True (0.06). Updating 0.00 to 0.00
    	* "buy" is True (0.06). Updating 0.00 to 0.00
    	* "buy" is True (0.06). Updating 0.00 to 0.00
    	* "logout" is True (0.20). Updating 0.00 to 0.00
    Probability that our user is a seller is: 0.000%
    



```python

```
