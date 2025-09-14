# Explanation of any changes made in Q4

* The expected values of `episode_length=200` and `max_timesteps=31000` are used, as indicated by the announcement.
* All the other constants remain unchanged, as required by this question.

* To avoid stopping training too early with a single lucky evaluation (where the mean return happens to exceed 500 by chance, but the model has not learnt enough to perform reliably well), the early stopping rule has been removed, allowing training to continue until reaching `max_timesteps=31000` or `max_time=7200`.
* Reward shaping techniques have been applied to ensure a stable performance, which will be discussed in detail in Question 5 (`Q5.pdf`). 
  - The file `train_ddpg.py` in folder `exercise 4` is the original file, while that in folder `exercise 5` contains all the code for reward shaping implementation.