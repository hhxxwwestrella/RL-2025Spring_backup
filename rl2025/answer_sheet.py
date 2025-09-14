
############################################################################################################
##########################            RL2025 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the Every-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and Every-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) Every-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / Every-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The evaluation returns achieved by Every-Visit Monte Carlo are more sensitive to gamma compared to Q-learning. This is because, in Monte Carlo methods, the earlier a timestep is, the more future rewards we include in its return, each discounted by higher power of gamma. When gamma is small, the distant rewards contribute less, making the return less sensitive to long-term differences between good and bad episodes. In contrast, Q-learning uses bootstrapping to update values based on immediate rewards and estimated future values, thus the estimation of the current state-action value is less affected by the value of gamma."
    # TYPE YOUR ANSWER HERE (100 words max)
    return answer

def question2_5() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) on the differences between the non-slippery and the slippery varian of the problem.
    by [Q-learning / Every-Visit Monte Carlo].
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "In the non-slippery environment, as actions always lead to the intended direction, both Q-learning and Monte Carlo agents can learn reliably and consistently achieve a best mean return of 1.0 over ten random seeds. However, in the slippery environment, the same action can result in different transitions, making the action value estimations less stable and preventing both agents from reaching the goal consistently. Monte Carlo learning is more affected by this randomness, suffering from the high variance of the trajectories. In contrast, Q-learning updates the values gradually and the random effects from slippery transitions might be balanced over time."

    # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In the DiscreteRL algorithm, which learning rate achieves the highest mean returns at the end of training?
    a) 2e-2
    b) 2e-3
    c) 2e-4
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.99
    b) 0.75
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.5
    c) 1e-5
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 0.95?
    a) 0.95
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "When employing an exponential decay strategy with an effective epsilon decay value (i.e., smaller than 1 so that the epsilon value does change over time), the epsilon at the end of training depends on the number of training timesteps. This means that, for those tasks that require fewer timesteps to train, the epsilon could still be relatively large at the end of training, preventing the agent from focusing on exploitation. In contrast, in linear decay strategy, the epsilon can always reach the expected minimum value when a specified fraction of total timesteps have elapsed."
    # TYPE YOUR ANSWER HERE (100 words max)
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "In supervised learning, the aim is to reduce the overall loss on the fixed training data, so the loss typically decreases during the training process. However, in DQN, the training target changes over time because the target network is periodically updated for bootstrapping, which introduces instability. At the early stage of training, as the agent cannot reach the goal, there is little distinction between good and bad actions because all actions yield similarly poor outcomes, which makes the loss relatively small. Later, as the agent learns effective strategies, the Q-values become more sensitive to action quality and any exploration or bad previous episodes replayed by the buffer can cause large TD errors. As a result, over time, the loss can fluctuate and even increase on average."
    # TYPE YOUR ANSWER HERE (150 words max)
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The spikes in the DQN loss plot occur at regular intervals of about 2000 timesteps, which matches the target update frequency. This suggests that they are likely caused by the periodic synchronisation of the target network with the value network. After each update, the target values used for computing the TD error can suddenly change, especially when the value network has changed drastically. This can temporarily increase the TD error, resulting in a spike in the loss plot. "
    # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    answer = "When training models in Q4, I found that the agent tended to learn to spin in place, which led to low returns and high risk of colliding with the blue car. Reward shaping has been introduced to encourage the agent to drive along the lane and avoid the blue car. Although the agent cannot always succeed in avoiding collisions, this trick has significantly improved the model's performance. See Q5.pdf for more details."  # TYPE YOUR ANSWER HERE (200 words max)
    return answer
