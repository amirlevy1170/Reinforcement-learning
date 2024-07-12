import gymnasium as gy
import numpy as np
import matplotlib.pyplot as pl

def GetUniformWeights():
    return np.random.uniform(-1, 1, 4)

def GetBestScoreAndWeight(bestScore, bestWeights, reward, weight):
    if reward > bestScore:
        return reward, weight
    return bestScore, bestWeights

def RunEpisode(env, w):
    reward = 0
    observation, _ = env.reset()
    
    for _ in range(200):
        stepAction = Agent(w, observation)
        observation, currReward, terminated, truncated, _ = env.step(stepAction)
        
        reward += currReward
        
        if terminated or truncated: 
            break
    
    return reward

# Question 2.2
def Agent(ot, w):
    return 1 if np.dot(ot, w) >= 0 else 0

# Question 2.3
def Q2_3():
    # Setup
    _env = gy.make("CartPole-v1", render_mode="human")
    _initialWeights = GetUniformWeights()
    
    # Act
    finalScore = RunEpisode(_env, _initialWeights)
    
    # Show result
    print("final score is: ", finalScore)
    _env.close()
    
# Question 2.4
def RandomSearch(t = 10000, maxScore = 201):
    # Setup
    env = gy.make("CartPole-v1")
    bestScore, bestWeights, episodes = 0, None, 0
    
    #Act
    while episodes <= t and bestScore < maxScore:
        weight = GetUniformWeights()
        reward = RunEpisode(env, weight)
        episodes += 1
        
        bestScore, bestWeights = GetBestScoreAndWeight(bestScore, bestWeights, reward, weight)
    env.close()

    # Results
    return bestScore, bestWeights, episodes

# Question 5
def Q2_5():
    # Setup
    episodesHistogram = []
    
    # Act
    for i in range(1000):
        _, _, episodes = RandomSearch(2000, 200)
        episodesHistogram.append(episodes)
    
    # Show results
    averageEpisodes = "average number of episodes is: " + str(np.average(episodesHistogram))
    print(averageEpisodes)
    pl.hist(episodesHistogram, bins=100, edgecolor='k')
    pl.xlabel('Number of Episodes')
    pl.ylabel('Frequency')
    pl.title(averageEpisodes)
    pl.show()

def Q2():
    Q2_3()
    Q2_5()
    
Q2()