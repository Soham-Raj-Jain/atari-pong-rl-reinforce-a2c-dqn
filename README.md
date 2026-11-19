# Atari Pong Reinforcement Learning: REINFORCE, Actor-Critic, and DQN

A comparative reinforcement learning study on **Atari Pong** using a Monte Carlo policy-gradient method, a Pong-specific advantage actor-critic implementation, and a previously trained DQN baseline.

This project was developed for **CMPE 260** and focuses not only on final reward, but also on convergence behavior, sample efficiency, wall-clock training time, value estimation, and the differences between on-policy and off-policy reinforcement learning.

## Team

- Vineeth Chandra Sai Kandukuri
- Soham Raj Jain
- Arshan Bhanage
- Siddharth Rao Kartik

## Project Overview

The goal of this project is to study how different reinforcement learning paradigms behave on the same Atari Pong task.

We evaluate three approaches:

1. **REINFORCE**  
   A Monte Carlo policy-gradient algorithm that directly optimizes the policy using complete-episode returns.

2. **Advantage Actor-Critic**  
   A Pong-specific long-horizon actor-critic implementation that extends the successful REINFORCE policy with a learned state-value critic.

3. **DQN**  
   The value-based, off-policy baseline from our previous Atari Pong experiment.

The primary comparison considers:

- final evaluation performance
- convergence behavior
- training sample usage
- wall-clock training time
- training versus evaluation performance
- on-policy versus off-policy learning
- actor-critic value estimation
- DQN value overestimation considerations

---

## Atari Pong Environment

The experiments preserve the environment and preprocessing configuration used for the DQN baseline.

### Environment

```text
ALE/Pong-v5
frameskip = 1
repeat_action_probability = 0.0
```

A custom action-repeat wrapper executes each selected action for four Atari frames.

### Preprocessing

```text
RGB Atari frame
        ↓
grayscale
        ↓
resize and crop
        ↓
84 × 84
        ↓
4-frame stack
        ↓
scaled floating-point observation
```

The resulting environment observation has shape:

```text
(4, 84, 84)
```

The policy-gradient agents additionally derive an **80 × 80 frame-difference representation** from consecutive frames to emphasize motion while suppressing the mostly static Pong background.

---

# REINFORCE

The REINFORCE implementation is primarily inspired by the classic Pong policy-gradient approach described by Andrej Karpathy.

Instead of using a large convolutional network, the policy operates on the difference between consecutive processed frames.

### Policy Architecture

```text
80 × 80 frame difference
        ↓
flatten
        ↓
6400 features
        ↓
Linear(6400, 200)
        ↓
ReLU
        ↓
Linear(200, 1)
        ↓
Bernoulli policy
        ↓
LEFT / RIGHT
```

The two learned policy actions are mapped to the meaningful Pong movement actions after the environment performs FIRE reset.

### Final REINFORCE Configuration

```text
gamma                  = 0.99
learning rate          = 2e-4
hidden size            = 200
episodes per update    = 10
RMSProp alpha          = 0.99
RMSProp epsilon        = 1e-5
```

The implementation uses:

- complete-episode Monte Carlo returns
- Pong point-boundary discount reset
- per-episode return normalization
- summed policy-gradient contributions
- stochastic Bernoulli action sampling
- checkpoint-based training continuation

### Final REINFORCE Result

```text
Training decisions:      10,000,000
Last-100 train return:        18.11
Evaluation mean:              18.58
Evaluation std:                2.18
Evaluation episodes:             50
Training time:                ~4.77 h
Maximum episode return:           21
```

REINFORCE achieved the strongest final evaluation result in the experiments.

---

# Advantage Actor-Critic

Several conventional short-horizon Atari A2C configurations were investigated during development but did not learn reliably with the exact Pong environment and preprocessing configuration used in this project.

The final implementation therefore uses a **Pong-specific long-horizon advantage actor-critic formulation**.

It preserves the successful REINFORCE representation and actor while adding a separate learned value-function critic.

## Architecture

```text
                    6400 motion features
                         /       \
                        /         \
                     Actor       Critic
                       ↓            ↓
                 200-unit ReLU  200-unit ReLU
                       ↓            ↓
                  action logit     V(s)
```

The actor and critic use separate parameters and optimizers.

### Advantage

The actor is trained using:

\[
A_t = G_t - \beta V(s_t)
\]

where:

- \(G_t\) is the point-boundary discounted return
- \(V(s_t)\) is the critic's state-value estimate
- \(\beta\) controls how strongly the critic affects the actor

At the beginning of training:

```text
beta = 0
```

so actor learning remains close to REINFORCE.

As the critic learns, beta gradually increases until:

```text
beta = 1
```

at which point the full learned value baseline participates in the policy update.

### Selected Configuration

```text
gamma                       = 0.99
actor learning rate         = 2e-4
critic learning rate        = 5e-4
hidden size                 = 200
episodes per update         = 10
RMSProp alpha               = 0.99
RMSProp epsilon             = 1e-5
baseline warmup updates     = 120
baseline full updates       = 260
critic epochs               = 2
baseline clip               = 1.5
entropy coefficient         = 0.0
```

### Final Actor-Critic Result

```text
Training decisions:      9,050,863
Episodes:                    3,184
Updates:                       318
Last-100 train return:       13.41
Evaluation mean:             14.04
Evaluation std:               3.12
Training time:               ~6.04 h
Critic explained variance:   ~0.72
```

The run stopped because the configured wall-clock limit was reached before the 10-million-decision training budget.

The critic's explained variance near the end of training shows that the value network learned a meaningful representation of the return targets.

---

# DQN Baseline

The DQN results were produced in the previous Atari Pong experiment and are reused here as the off-policy comparison baseline.

Unlike REINFORCE and actor-critic methods, DQN learns an action-value function:

\[
Q(s,a)
\]

and selects actions based on the estimated value of each available action.

DQN also uses:

- experience replay
- bootstrapped Q-learning targets
- target-network updates
- epsilon-greedy action selection
- off-policy data reuse

### DQN Evaluation Results

| Seed | Last-100 Training Return | Evaluation Return |
|---:|---:|---:|
| 42 | 17.52 | 14.72 |
| 123 | 18.47 | 7.00 |
| 2026 | 18.60 | 11.28 |

Average DQN evaluation return:

```text
11.00
```

Each DQN experiment used:

```text
500,000 agent decisions
```

which is substantially fewer environment interactions than either policy-gradient approach.

---

# Final Comparison

| Algorithm | Evaluation Return | Training Decisions | Training Time |
|---|---:|---:|---:|
| **REINFORCE** | **18.58 ± 2.18** | **10.00M** | **~4.77 h** |
| **Actor-Critic** | **14.04 ± 3.12** | **9.05M** | **~6.04 h** |
| **DQN** | **11.00 average** | **0.50M per run** | **~1.20 h average** |

## Key Findings

### Best Final Performance

REINFORCE achieved the highest measured final evaluation return.

```text
REINFORCE > Actor-Critic > DQN average
```

### Best Sample Efficiency

DQN required dramatically fewer environment interactions.

```text
DQN          0.50M
Actor-Critic 9.05M
REINFORCE   10.00M
```

Its replay buffer allows previously collected experience to be reused many times, whereas REINFORCE and the actor-critic implementation are on-policy.

### Actor-Critic Learning

The actor-critic agent successfully learned both:

- a policy for controlling Pong
- a state-value critic

The critic reached approximately **0.72 explained variance** near the end of training.

However, adding a critic did not automatically improve final performance relative to REINFORCE.

### Training vs Evaluation

REINFORCE and actor-critic showed similar late-training and evaluation performance.

DQN showed a substantially larger difference between late-training performance and final evaluation performance.

This difference is reported descriptively and is not treated as direct proof of overfitting.

---

# Value-Based vs Policy-Gradient Learning

The three methods represent different reinforcement-learning strategies.

### DQN

```text
state
  ↓
Q-values for actions
  ↓
choose highest-valued action
```

DQN is primarily **value-based**.

### REINFORCE

```text
state
  ↓
policy
  ↓
action probabilities
  ↓
sample action
```

REINFORCE directly optimizes the **policy**.

### Actor-Critic

```text
                  state
                 /     \
                ↓       ↓
             actor    critic
               ↓        ↓
            policy     V(s)
```

The actor chooses actions while the critic estimates state value and improves the actor's learning signal.

---

# On-Policy vs Off-Policy

REINFORCE and the actor-critic implementation are **on-policy**.

They train primarily using trajectories generated by the current policy.

DQN is **off-policy**.

Its replay buffer allows older transitions to be reused repeatedly, contributing to its much greater sample efficiency in these experiments.

---

# Hyperparameter Tuning

Both policy-gradient methods were tuned before their expensive long-run experiments.

### REINFORCE candidates

The tuning search varied:

- learning rate
- number of complete episodes per update

The winning configuration used:

```text
learning rate = 2e-4
episodes/update = 10
```

### Actor-Critic candidates

The actor-critic search varied:

- critic learning rate
- when the critic begins influencing the actor
- how quickly critic contribution reaches full strength

The winning configuration was the **slow-baseline** variant, which allowed the critic more time to learn before fully influencing policy updates.

---

# Training and Checkpointing

The notebook is designed for long-running Google Colab experiments.

Training progress is stored persistently in Google Drive so interrupted sessions can resume instead of restarting from zero.

Saved training information includes:

- model parameters
- optimizer state
- agent decision count
- episode history
- update history
- elapsed training time
- configuration
- evaluation history

The training functions support:

```python
resume=True
```

to continue from compatible checkpoints.

---

# Notebook Workflow

The notebook follows a staged experimental workflow:

```text
Environment setup
      ↓
Environment validation
      ↓
REINFORCE implementation
      ↓
REINFORCE smoke test
      ↓
REINFORCE tuning
      ↓
REINFORCE long run
      ↓
Actor-Critic implementation
      ↓
Actor-Critic smoke test
      ↓
Actor-Critic tuning
      ↓
Actor-Critic long run
      ↓
Freeze configurations
      ↓
Load DQN reference
      ↓
Final comparison
      ↓
Learning curves and analysis
```

This prevents expensive training runs from being launched before the implementation has been validated.

---

# Running the Notebook

The easiest way to reproduce or inspect the experiments is through **Google Colab**.

1. Upload or open `CMPE260_Project2_Atari_Pong_RL.ipynb`.
2. Enable a GPU runtime.
3. Run the environment/setup cells.
4. Mount Google Drive when requested.
5. Run the smoke tests before starting long training.
6. Training checkpoints will be saved to the configured Drive directory.
7. Existing compatible checkpoints can be resumed.

The full long-run experiments require several hours of GPU runtime.

---

# Experimental Limitations

The comparison is intentionally reported with several limitations.

- REINFORCE and actor-critic use one final long-run training seed (`777`).
- DQN results come from three independently trained seeds (`42`, `123`, and `2026`).
- The policy-gradient agents use the two meaningful Pong movement actions after FIRE reset.
- DQN uses the full six-action minimal Pong action set.
- DQN evaluation uses epsilon `0.02`.
- REINFORCE and actor-critic are evaluated using their learned stochastic policies.
- Training budgets are not identical across algorithms.

Therefore, the results should be interpreted as an empirical comparison of the measured experiments rather than a universal ranking of the algorithms.

---

# Main Conclusion

The experiments demonstrate that reinforcement-learning algorithms can succeed for very different reasons.

**REINFORCE** produced the strongest final Pong policy.

**DQN** required substantially fewer environment interactions because experience replay allows extensive data reuse.

**Actor-Critic** successfully learned a meaningful value-function baseline, but the additional critic did not outperform the simpler REINFORCE policy in this particular environment.

The main takeaway is therefore not that one algorithm is universally superior, but that **final performance, sample efficiency, computational cost, variance reduction, and data reuse represent different tradeoffs in reinforcement learning**.

---

## References

- Andrej Karpathy, *Deep Reinforcement Learning: Pong from Pixels*
- Andrej Karpathy, `pg-pong.py`
- OpenAI Baselines, A2C
- RL Baselines3 Zoo Atari configurations
- Arcade Learning Environment / Atari Pong
