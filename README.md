# 4106-DRL-Final-Project-Group-2

by: Yema, Natkeeran, and Neil

# Where this class project fit's into Yema's broader Solution Architecture

### Master Agent (Bayesian Optimization lives here)
* Talks to the human. Owns the hypothesis. Schedules batches. Decides when a trial is done.
* Optimizes experimental conditions: glucose, peptone, pH, Mg²⁺, inoculation ratio.

### Supervisor Agent
* Coordinates between arms. Allocates tasks. Resolves conflicts. Safety interrupts.

### Arm Agents (this project — DRL)
* Each arm is a DRL agent:
	* State: tip position, target position, fluid level
 	* Action: joint commands
	* Reward: reaching target + delivering correct volume
* Multiple arms = multiple agents, trained independently.

### Data Layer (Kafka / TML / Ollama)
* Streams: humidity, temperature, pH, culture viability, arm telemetry. Retention: 24 hours.

### Physical Layer
* Robotic arms + reservoirs + petri dishes + sensors.

# This Project's Scope: 1 Arm
* Human role: places petri dishes containing Cordyceps militaris anywhere on the table.
* Robot role: one arm that uses a pipette to aspirate nutrient solution from a stationary beaker and dispense the liquid into a randomly located petri dish 

## Training the DRL Agents
* Isaac Lab + teacher-student approach 
    * sidesteps exploration via GPU parallelism and privileged state
* CPU-based PPO/SAC
  	* Yema's tests 
    	* hit the sparse-reward wall mine 
   	 	* provides the diagnostic testbed
   		* reach_env.py — final state: beaker at (0.19, 0.44, 0.82), success threshold 0.08m, linear reward
   	 	* train_reach.py — SAC configuration (buffer_size=100k, batch_size=256)
   		* Training logs: PPO @ 0.14m, PPO @ 0.105m, SAC @ 0.105m
 	* Natkeeran's tests
		* The agent realized that the fastest way to get liquid from the beaker to the Petri dish is to smash through the beaker lol, making the 96% success rate actually a 0% success rate
 
# Next steps:
* Upload our individual docs to repository ✅
* Combine Yema's & Natkeeran's PPO training codes 
